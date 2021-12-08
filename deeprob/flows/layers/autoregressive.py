# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn

from deeprob.flows.utils import Bijector
from deeprob.torch.utils import ScaledTanh, MaskedLinear, get_activation_class


class AutoregressiveLayer(Bijector):
    def __init__(
        self,
        in_features: int,
        depth: int,
        units: int,
        activation: str,
        reverse: bool = False,
        sequential: bool = True,
        random_state: Optional[np.random.RandomState] = None
    ):
        """
        Build an autoregressive layer as specified in Masked Autoregressive Flow paper.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param activation: The activation used for inner layers of the conditioner.
        :param reverse: Whether to reverse the mask used in the autoregressive layer. Used only if sequential is True.
        :param sequential: Whether to use sequential degrees for inner layers masks.
        :param random_state: The random state used to generate the masks degrees. Used only if sequential is False.
        :raises ValueError: If a parameter is out of domain.
        """
        if depth <= 0:
            raise ValueError("The depth value must be positive")
        if units <= 0:
            raise ValueError("The units value must be positive")
        if not sequential and not isinstance(random_state, np.random.RandomState):
            raise ValueError("A Numpy RandomState is required if sequential is False")
        activation_cls = get_activation_class(activation)

        super().__init__(in_features)
        self.layers = nn.ModuleList()
        self.scale_act = ScaledTanh()

        # Create the masks of the masked linear layers
        if sequential:
            degrees = self.build_degrees_sequential(depth, units, reverse)
        else:
            degrees = self.build_degrees_random(depth, units, random_state)
        masks = self.build_masks(degrees)

        # Preserve the input ordering
        self.ordering = degrees[0]

        # Initialize the conditioner neural network
        layers = []
        out_features = units
        for mask in masks[:-1]:
            layers.extend([
                MaskedLinear(in_features, out_features, mask),
                activation_cls()
            ])
            in_features = out_features
        out_features = self.in_features * 2
        layers.append(MaskedLinear(in_features, out_features, np.tile(masks[-1], reps=(2, 1))))
        self.network = nn.Sequential(*layers)

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the parameters and apply the affine transformation (backward mode)
        z = self.network(x)
        t, s = torch.chunk(z, chunks=2, dim=1)
        s = self.scale_act(s)
        u = (x - t) * torch.exp(-s)
        inv_log_det_jacobian = -torch.sum(s, dim=1)
        return u, inv_log_det_jacobian

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize arbitrarily
        x = torch.zeros_like(u)
        log_det_jacobian = torch.zeros_like(u)

        # This requires D iterations where D is the number of features
        # Get the parameters and apply the affine transformation (forward mode)
        for i in range(self.in_features):
            z = self.network(x)
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_act(s)
            idx = np.argwhere(self.ordering == i).item()
            x[:, idx] = u[:, idx] * torch.exp(s[:, idx]) + t[:, idx]
            log_det_jacobian[:, idx] = s[:, idx]
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1)
        return x, log_det_jacobian

    def build_degrees_sequential(self, depth: int, units: int, reverse: bool) -> List[np.ndarray]:
        """
        Build sequential degrees for the linear layers of the autoregressive network.

        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param reverse: Whether to reverse the mask used in the autoregressive layer. Used only if sequential is True.
        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        # Initialize the input degrees sequentially
        degrees = []
        if reverse:
            degrees.append(np.arange(self.in_features - 1, -1, -1))
        else:
            degrees.append(np.arange(self.in_features))

        # Add the degrees of the hidden layers
        for _ in range(depth):
            degrees.append(np.arange(units) % (self.in_features - 1))
        return degrees

    def build_degrees_random(self, depth: int, units: int, random_state: np.random.RandomState) -> List[np.ndarray]:
        """
        Create random degrees for the linear layers of the autoregressive network.

        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param random_state: The random state.
        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        # Initialize the input degrees randomly
        degrees = []
        ordering = np.arange(self.in_features)
        random_state.shuffle(ordering)
        degrees.append(ordering)

        # Add the degrees of the hidden layers
        for _ in range(depth):
            min_prev_degree = np.min(degrees[-1])
            degrees.append(random_state.randint(min_prev_degree, self.in_features - 1, units))
        return degrees

    @staticmethod
    def build_masks(degrees: List[np.ndarray]) -> List[np.ndarray]:
        """
        Build masks from degrees.

        :param degrees: A sequence of units degrees for each hidden layer.
        :return: The masks to use for each hidden layer of the autoregressive network.
        """
        masks = []
        for (d1, d2) in zip(degrees[:-1], degrees[1:]):
            d1 = np.expand_dims(d1, axis=0)
            d2 = np.expand_dims(d2, axis=1)
            masks.append(np.less_equal(d1, d2))
        d1 = np.expand_dims(degrees[-1], axis=0)
        d2 = np.expand_dims(degrees[0], axis=1)
        masks.append(np.less(d1, d2))
        return masks
