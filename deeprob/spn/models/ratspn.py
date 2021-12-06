# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Tuple, Type

import torch
import torch.nn.functional as F

from deeprob.utils.random import RandomState
from deeprob.utils.region import RegionGraph
from deeprob.torch.base import ProbabilisticModel
from deeprob.torch.constraints import ScaleClipper
from deeprob.spn.layers.ratspn import RegionGraphLayer, GaussianLayer, BernoulliLayer
from deeprob.spn.layers.ratspn import SumLayer, ProductLayer, RootLayer


class RatSpn(ProbabilisticModel):
    def __init__(
        self,
        in_features: int,
        base_cls: Type[RegionGraphLayer],
        base_kwargs: Optional[dict] = None,
        out_classes: int = 1,
        rg_depth: int = 2,
        rg_repetitions: int = 1,
        rg_batch: int = 2,
        rg_sum: int = 2,
        in_dropout: Optional[float] = None,
        sum_dropout: Optional[float] = None,
        random_state: Optional[RandomState] = None
    ):
        """
        Initialize a RAT-SPN.

        :param in_features: The number of input features.
        :param base_cls: The base distribution's class. It must be a sub-class of RegionGraphLayer.
        :param base_kwargs: Optional additiona parameters to pass to the base distribution's class constructor.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param rg_batch: The number of base distribution batches.
        :param rg_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at sum layers. It can be None.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :raises ValueError: If a parameter is out of domain.
        """
        if not issubclass(base_cls, RegionGraphLayer):
            raise ValueError("The base distribution's class must be a sub-class of RegionGraphLayer")
        if in_features <= 0:
            raise ValueError("The number of input features must be positve")
        if out_classes <= 0:
            raise ValueError("The number of output classes must be positive")
        if rg_batch <= 0:
            raise ValueError("The number of base distribution batches must be positive")
        if rg_sum <= 0:
            raise ValueError("The number of sum nodes per region must be positive")
        if in_dropout is not None and (in_dropout <= 0.0 or in_dropout >= 1.0):
            raise ValueError("The dropout rate at base distribution must be in (0, 1)")
        if sum_dropout is not None and (sum_dropout <= 0.0 or sum_dropout >= 1.0):
            raise ValueError("The dropout rate at sum layers must be in (0, 1)")

        super().__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.rg_depth = rg_depth
        self.rg_batch = rg_batch
        self.rg_sum = rg_sum
        self.in_dropout = in_dropout
        self.sum_dropout = sum_dropout
        self.layers = torch.nn.ModuleList()

        # Instantiate the region graph
        region_graph = RegionGraph(self.in_features, self.rg_depth, random_state)

        # Generate the region graph layers
        rg_layers = region_graph.make_layers(rg_repetitions)
        self.rg_layers = list(reversed(rg_layers))

        # Instantiate the base distributions layer
        if base_kwargs is None:
            base_kwargs = dict()
        self.base_layer = base_cls(
            self.in_features, self.rg_batch,
            regions=self.rg_layers[0], rg_depth=self.rg_depth, dropout=self.in_dropout,
            **base_kwargs
        )

        # Alternate between product and sum layer
        in_groups = self.base_layer.in_regions
        in_nodes = self.base_layer.out_channels
        for i in range(1, len(self.rg_layers) - 1):
            if i % 2 == 1:
                layer = ProductLayer(in_groups, in_nodes)
                in_groups = layer.out_partitions
                in_nodes = layer.out_nodes
            else:
                layer = SumLayer(in_groups, in_nodes, self.rg_sum, self.sum_dropout)
                in_groups = layer.out_regions
                in_nodes = layer.out_nodes
            self.layers.append(layer)

        # Instantiate the root layer
        self.root_layer = RootLayer(in_groups, in_nodes, self.out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood given some evidence.
        Random variables can be marginalized using NaN values.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Forward through the inner layers
        for layer in self.layers:
            x = layer(x)

        # Forward through the root layer
        log_prob = self.root_layer(x)
        return log_prob

    @torch.no_grad()
    def mpe(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the maximum at posteriori estimation.
        Random variables can be marginalized using NaN values.

        :param x: The inputs tensor.
        :param y: The target classes tensor. It can be None for unlabeled maximum at posteriori estimation.
        :return: The output of the model.
        """
        lls = []
        inputs = x
        n_samples = x.shape[0]

        # Compute the base distributions log-likelihoods
        x = self.base_layer(x)

        # Compute in forward mode and gather the inner log-likelihoods
        for layer in self.layers:
            lls.append(x)
            x = layer(x)

        # Compute in forward mode through the root layer and get the class index,
        # if no target classes are specified
        if self.out_classes == 1:
            y = torch.zeros(n_samples, dtype=torch.long)
        elif y is None:
            y = torch.argmax(self.root_layer(x), dim=1)

        # Get the root layer indices
        idx_group, idx_offset = self.root_layer.mpe(x, y)

        # Compute in top-down mode through the inner layers
        for i in range(len(self.layers) - 1, -1, -1):
            idx_group, idx_offset = self.layers[i].mpe(lls[i], idx_group, idx_offset)

        # Compute the maximum at posteriori inference at the base layer
        samples = self.base_layer.mpe(inputs, idx_group, idx_offset)
        return samples

    @torch.no_grad()
    def sample(self, n_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute in forward mode through the root layer and get the class index,
        # if no target classes are specified
        if self.out_classes == 1:
            y = torch.zeros(n_samples).long()
        elif y is None:
            y = torch.randint(self.out_classes, [n_samples])

        # Get the root layer indices
        idx_group, idx_offset = self.root_layer.sample(y)

        # Compute in top-down mode through the inner layers
        for i in range(len(self.layers) - 1, -1, -1):
            idx_group, idx_offset = self.layers[i].sample(idx_group, idx_offset)

        # Compute the maximum at posteriori inference at the base layer
        samples = self.base_layer.sample(idx_group, idx_offset)
        return samples

    def loss(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Generative setting, return average negative log-likelihood
        if self.out_classes == 1:
            return -torch.mean(x)

        # Discriminative setting, return cross-entropy loss
        logits = torch.log_softmax(x, dim=1)
        return F.nll_loss(logits, y)


class GaussianRatSpn(RatSpn):
    def __init__(
        self,
        in_features: int,
        out_classes: int = 1,
        rg_depth: int = 2,
        rg_repetitions: int = 1,
        rg_batch: int = 2,
        rg_sum: int = 2,
        in_dropout: Optional[float] = None,
        sum_dropout: Optional[float] = None,
        random_state: Optional[RandomState] = None,
        uniform_loc: Optional[Tuple[float, float]] = None,
        optimize_scale: bool = False
    ):
        """
        Initialize a Gaussian RAT-SPN.

        :param in_features: The number of input features.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param rg_batch: The number of base distributions batches.
        :param rg_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at sum layers. It can be None.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :param uniform_loc: The optional uniform distribution parameters for location initialization.
        :param optimize_scale: Whether to train scale and location jointly.
        """
        super().__init__(
            in_features,
            GaussianLayer, {'uniform_loc': uniform_loc, 'optimize_scale': optimize_scale},
            out_classes, rg_depth, rg_repetitions, rg_batch, rg_sum,
            in_dropout, sum_dropout, random_state
        )

        # Initialize the scale clipper, if specified
        self.optimize_scale = optimize_scale
        if self.optimize_scale:
            self.scale_clipper = ScaleClipper()

    def apply_constraints(self):
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)


class BernoulliRatSpn(RatSpn):
    def __init__(
        self,
        in_features: int,
        out_classes: int = 1,
        rg_depth: int = 2,
        rg_repetitions: int = 1,
        rg_batch: int = 2,
        rg_sum: int = 2,
        in_dropout: Optional[float] = None,
        sum_dropout: Optional[float] = None,
        random_state: Optional[RandomState] = None
    ):
        """
        Initialize a Bernoulli RAT-SPN.

        :param in_features: The number of input features.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param rg_depth: The depth of the region graph.
        :param rg_repetitions: The number of independent repetitions of the region graph.
        :param rg_batch: The number of base distributions batches.
        :param rg_sum: The number of sum nodes per region.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at product layer outputs. It can be None.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        """
        super().__init__(
            in_features,
            BernoulliLayer, None,
            out_classes, rg_depth, rg_repetitions, rg_batch, rg_sum,
            in_dropout, sum_dropout, random_state
        )
