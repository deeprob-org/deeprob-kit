# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd

from deeprob.torch.base import ProbabilisticModel
from deeprob.torch.constraints import ScaleClipper
from deeprob.spn.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer


class DgcSpn(ProbabilisticModel):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        out_classes: int = 1,
        n_batch: int = 8,
        sum_channels: int = 8,
        depthwise: Union[bool, List[bool]] = False,
        n_pooling: int = 0,
        optimize_scale: bool = False,
        in_dropout: Optional[float] = None,
        sum_dropout: Optional[float] = None,
        quantiles_loc: Optional[np.ndarray] = None,
        uniform_loc: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize a Deep Generalized Convolutional Sum-Product Network (DGC-SPN).

        :param in_features: The input size as a (C, D, D) tuple.
        :param out_classes: The number of output classes. Specify 1 in case of plain density estimation.
        :param n_batch: The number of output channels of the base layer.
        :param sum_channels: The number of output channels of spatial sum layers.
        :param depthwise: Whether to use depthwise convolutions as product layers at each depth level.
                          The last flag of the list will be considered as the one for the rest of the network.
                          If a single boolean is passed, it will be used for all the network's product layers.
        :param n_pooling: The number of initial pooling spatial product layers.
        :param optimize_scale: Whether to train scale and location jointly.
        :param in_dropout: The dropout rate for probabilistic dropout at distributions layer outputs. It can be None.
        :param sum_dropout: The dropout rate for probabilistic dropout at sum layers. It can be None.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        :raises ValueError: If a parameter is out of domain.
        """
        if in_features[1] != in_features[2]:
            raise ValueError("The height and width of input size must be the same")
        if out_classes <= 0:
            raise ValueError("The number of output classes must be positive")
        if n_batch <= 0:
            raise ValueError("The number of base distribution batches must be positive")
        if sum_channels <= 0:
            raise ValueError("The number of output channels of spatial sum layers must be positive")
        if in_dropout is not None and (in_dropout <= 0.0 or in_dropout >= 1.0):
            raise ValueError("The dropout rate at base distribution must be in (0, 1)")
        if sum_dropout is not None and (sum_dropout <= 0.0 or sum_dropout >= 1.0):
            raise ValueError("The dropout rate at spatial sum layers must be in (0, 1)")
        if quantiles_loc is not None and uniform_loc is not None:
            raise ValueError("At least one between quantiles_loc and uniform_loc must be None")
        if quantiles_loc is not None and len(quantiles_loc.shape) != 4:
            raise ValueError("The mean quantiles must be a 4D Numpy array")
        if uniform_loc is not None and (len(uniform_loc) != 2 or uniform_loc[0] >= uniform_loc[1]):
            raise ValueError("The uniform range must be a pair (A, B) with A < B")

        # Check depthwise and n_pooling arguments, based on computed network's depth
        depth = int(np.ceil(np.log2(in_features[1])))
        if isinstance(depthwise, bool):
            depthwise = [depthwise] * (depth + 1)
        else:
            if len(depthwise) == 0 or len(depthwise) > depth + 1:
                raise ValueError("The length of depthwise argument must be in [1, ceil(log2(D)) + 1]")
            rest_depthwise = depth + 1 - len(depthwise)
            depthwise.extend([depthwise[-1]] * rest_depthwise)
        if n_pooling < 0 or n_pooling > depth:
            raise ValueError("The number of initial pooling spatial product layers must be in [0, ceil(log2(D))]")

        super().__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.n_batch = n_batch
        self.sum_channels = sum_channels
        self.depthwise = depthwise
        self.n_pooling = n_pooling
        self.optimize_scale = optimize_scale
        self.in_dropout = in_dropout
        self.sum_dropout = sum_dropout
        self.layers = torch.nn.ModuleList()

        # Instantiate the base distribution layer
        self.base_layer = SpatialGaussianLayer(
            self.in_features, self.n_batch,
            optimize_scale=self.optimize_scale, dropout=self.in_dropout,
            quantiles_loc=quantiles_loc, uniform_loc=uniform_loc
        )
        in_features = self.base_layer.out_features

        # Instantiate the inner layers
        for i in range(depth + 1):
            # Check for spatial product pooling layers, and
            # check whether to use depthwise product layer at current depth
            if i < self.n_pooling:
                padding = 'valid'
                stride = (2, 2)
                dilation = (1, 1)
            else:
                padding = 'final' if i == depth else 'full'
                stride = (1, 1)
                k = i - self.n_pooling
                dilation = (2 ** k, 2 ** k)

            # Add a spatial product layer
            spatial_prod = SpatialProductLayer(
                in_features, kernel_size=(2, 2), padding=padding,
                stride=stride, dilation=dilation, depthwise=self.depthwise[i]
            )
            self.layers.append(spatial_prod)
            in_features = spatial_prod.out_features

            if i != depth:
                # Add a spatial sum layer
                spatial_sum = SpatialSumLayer(in_features, self.sum_channels, self.sum_dropout)
                self.layers.append(spatial_sum)
                in_features = spatial_sum.out_features

        # Instantiate the spatial root layer
        self.root_layer = SpatialRootLayer(in_features, self.out_classes)

        # Initialize the scale clipper to apply, if specified
        if self.optimize_scale:
            self.scale_clipper = ScaleClipper()

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

    def mpe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum at posteriori estimation.
        Random variables can be marginalized using NaN values.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the base distribution log-likelihoods
        z = self.base_layer(x)

        # Just in case the inputs don't requires gradients
        if not z.requires_grad:
            z.requires_grad = True

        # Forward through the inner layers
        y = z
        for layer in self.layers:
            y = layer(y)

        # Forward through the root layer
        y = self.root_layer(y)

        # Compute the gradients at distribution leaves
        z_grad, = autograd.grad(y, z, grad_outputs=torch.ones_like(y), only_inputs=True)

        with torch.no_grad():
            # Compute the maximum at posteriori estimate using leaves gradients
            mode = self.base_layer.loc
            estimates = torch.sum(torch.unsqueeze(z_grad, dim=2) * mode, dim=1)
            samples = torch.where(torch.isnan(x), estimates, x)
            return samples

    def sample(self, n_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("Sampling is not implemented for DGC-SPNs")

    def loss(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Generative setting, return average negative log-likelihood
        if self.out_classes == 1:
            return -torch.mean(x)

        # Discriminative setting, return cross-entropy loss
        logits = torch.log_softmax(x, dim=1)
        return F.nll_loss(logits, y)

    def apply_constraints(self):
        # Apply the scale clipper to the base layer, if specified
        if self.optimize_scale:
            self.scale_clipper(self.base_layer)
