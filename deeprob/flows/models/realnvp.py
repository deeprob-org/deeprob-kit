# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deeprob.torch.base import DensityEstimator
from deeprob.flows.utils import BatchNormLayer1d
from deeprob.flows.layers.coupling import CouplingLayer1d, CouplingBlock2d
from deeprob.flows.models.base import NormalizingFlow


class RealNVP1d(NormalizingFlow):
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
        affine: bool = True
    ):
        """
        Real Non-Volume-Preserving (RealNVP) 1D normalizing flow model.

        :param in_features: The number of input features.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param n_flows: The number of sequential coupling flows.
        :param depth: The number of hidden layers of flows conditioners.
        :param units: The number of hidden units per layer of flows conditioners.
        :param batch_norm: Whether to apply batch normalization after each coupling layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :raises ValueError: If a parameter is out of scope.
        """
        if n_flows <= 0:
            raise ValueError("The number of coupling flow layers must be positive")
        if depth <= 0:
            raise ValueError("The number of hidden layers of conditioners must be positive")
        if units <= 0:
            raise ValueError("The number of hidden units per layer must be positive")

        super().__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        self.n_flows = n_flows
        self.depth = depth
        self.units = units
        self.batch_norm = batch_norm
        self.affine = affine

        # Build the coupling layers
        reverse = False
        for _ in range(self.n_flows):
            self.layers.append(
                CouplingLayer1d(
                    self.in_features, self.depth, self.units,
                    affine=self.affine, reverse=reverse
                )
            )

            # Append batch normalization after each layer, if specified
            if self.batch_norm:
                self.layers.append(BatchNormLayer1d(self.in_features))

            # Invert the input ordering
            reverse = not reverse


class RealNVP2d(NormalizingFlow):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        dequantize: bool = False,
        logit: Optional[float] = None,
        in_base: Optional[DensityEstimator] = None,
        network: str = 'resnet',
        n_flows: int = 1,
        n_blocks: int = 2,
        channels: int = 32,
        affine: bool = True
    ):
        """
        Real Non-Volume-Preserving (RealNVP) 2D normalizing flow model.

        :param in_features: The input size as a (C, H, W) tuple.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :param network: The neural network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_flows: The number of sequential multi-scale architectures.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        """
        if network not in ['resnet', 'densenet']:
            raise ValueError("The neural network conditioner must be either 'resnet' or 'densenet'")
        if n_flows <= 0:
            raise ValueError("The number of coupling flow layers must be positive")
        if n_blocks <= 0:
            raise ValueError("The number of conditioners blocks must be positive")
        if channels <= 0:
            raise ValueError("The number of channels must be positive")

        super().__init__(in_features, dequantize=dequantize, logit=logit, in_base=in_base)
        self.n_flows = n_flows
        self.network = network
        self.n_blocks = n_blocks
        self.channels = channels
        self.affine = affine
        self.perm_matrices = torch.nn.ParameterList()

        # Build the coupling blocks
        channels = self.channels
        in_features = self.in_features
        for _ in range(self.n_flows):
            self.layers.append(CouplingBlock2d(
                in_features, self.network, self.n_blocks, channels,
                affine=self.affine, last_block=False
            ))

            # Initialize the order matrix for down-scaling and up-scaling
            self.perm_matrices.append(nn.Parameter(
                self.build_permutation_matrix(in_features[0]), requires_grad=False
            ))

            # Halve the number of channels due to multi-scale architecture
            in_features = (in_features[0] * 2, in_features[1] // 2, in_features[2] // 2)

            # Double the number of channels
            channels *= 2

        # Add the last coupling block
        self.layers.append(CouplingBlock2d(
            in_features, self.network, self.n_blocks, channels,
            affine=self.affine, last_block=True
        ))

    @staticmethod
    def build_permutation_matrix(channels: int) -> torch.Tensor:
        """
        Build the permutation matrix that defines (a non-trivial) variables ordering
        when downscaling or upscaling as in RealNVP.

        :param channels: The number of input channels.
        :return: The permutation matrix tensor.
        """
        weights = np.zeros([channels * 4, channels, 2, 2], dtype=np.float32)
        ordering = np.array([
            [[[1., 0.],
              [0., 0.]]],
            [[[0., 0.],
              [0., 1.]]],
            [[[0., 1.],
              [0., 0.]]],
            [[[0., 0.],
              [1., 0.]]]
        ], dtype=np.float32)
        for i in range(channels):
            weights[4*i:4*i+4, i:i+1] = ordering
        permutation = np.array([4 * i + j for j in [0, 1, 2, 3] for i in range(channels)])
        return torch.tensor(weights[permutation], dtype=torch.float32)

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the backward transformation.

        :param x: The inputs.
        :return: The transformed samples and the backward log-det-jacobian.
        """
        inv_log_det_jacobian = 0.0

        # Apply the coupling block layers
        slices = []
        for i, layer in enumerate(self.layers):
            x, ildj = layer.apply_backward(x)
            inv_log_det_jacobian += ildj

            if i != len(self.layers) - 1:
                # Downscale the results and split them in half (i.e. multi-scale architecture)
                x = F.conv2d(x, self.perm_matrices[i], stride=2)
                x, z = torch.chunk(x, chunks=2, dim=1)
                slices.append(z)

        # Re-concatenate all the chunks in reverse order and upscale the results
        for i in range(len(self.layers) - 2, -1, -1):
            x = torch.cat([x, slices[i]], dim=1)
            x = F.conv_transpose2d(x, self.perm_matrices[i], stride=2)

        return x, inv_log_det_jacobian

    def apply_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward transformation.

        :param x: the inputs.
        :return: The transformed samples and the forward log-det-jacobian.
        """
        log_det_jacobian = 0.0

        # Collect the chunks in and upscale the results
        slices = []
        for i in range(len(self.layers) - 1):
            # Downscale the results and split them in half (i.e. multi-scale architecture)
            x = F.conv2d(x, self.perm_matrices[i], stride=2)
            x, z = torch.chunk(x, chunks=2, dim=1)
            slices.append(z)

        # Apply the normalizing flows in forward mode
        for i, layer in reversed(list(enumerate(self.layers))):
            if i != len(self.layers) - 1:
                x = torch.cat([x, slices[i]], dim=1)
                x = F.conv_transpose2d(x, self.perm_matrices[i], stride=2)
            x, ldj = layer.apply_forward(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian
