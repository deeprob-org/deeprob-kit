# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional

from deeprob.torch.base import DensityEstimator
from deeprob.utils.random import check_random_state, RandomState
from deeprob.flows.utils import BatchNormLayer1d
from deeprob.flows.layers.autoregressive import AutoregressiveLayer
from deeprob.flows.models.base import NormalizingFlow


class MAF(NormalizingFlow):
    def __init__(
        self,
        in_features: int,
        dequantize: bool = False,
        logit: Optional[float] = None,
        in_base: Optional[DensityEstimator] = None,
        n_flows: int = 5,
        depth: int = 1,
        units: int = 128,
        batch_norm: bool = True,
        activation: str = 'relu',
        sequential: bool = True,
        random_state: Optional[RandomState] = None
    ):
        """
        Initialize a Masked Autorefressive Flow (MAF) model.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param n_flows: The number of sequential autoregressive layers.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each autoregressive layer.
        :param activation: The activation function name to use for the flows conditioners hidden layers.
        :param sequential: If True build masks degrees sequentially, otherwise randomly.
        :param random_state: The random state used to generate the masks degrees. Used only if sequential is False.
                             It can be either a seed integer or a np.random.RandomState instance.
        :raises ValueError: If a parameter is out of domain.
        """
        if n_flows <= 0:
            raise ValueError("The number of autoregressive flow layers must be positive")
        if depth <= 0:
            raise ValueError("The number of hidden layers of conditioners must be positive")
        if units <= 0:
            raise ValueError("The number of hidden units per layer must be positive")

        super().__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.activation = activation
        self.sequential = sequential

        # Check the random state, if not using sequential masks
        if not self.sequential:
            random_state = check_random_state(random_state)

        # Build the autoregressive layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                AutoregressiveLayer(
                    self.in_features, self.depth, self.units, self.activation,
                    reverse=reverse, sequential=self.sequential, random_state=random_state
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer1d(self.in_features))

            # Invert the input ordering
            reverse = not reverse
