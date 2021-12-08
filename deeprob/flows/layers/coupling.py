# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Tuple

import numpy as np
import torch
from torch import nn

from deeprob.torch.utils import ScaledTanh
from deeprob.flows.utils import squeeze_depth2d, unsqueeze_depth2d, Bijector, BatchNormLayer2d
from deeprob.flows.layers.densenet import DenseNetwork
from deeprob.flows.layers.resnet import ResidualNetwork


class CouplingLayer1d(Bijector):
    def __init__(
        self,
        in_features: int,
        depth: int,
        units: int,
        affine: bool = True,
        reverse: bool = False
    ):
        """
        Build a RealNVP (or NICE) 1D coupling layer.

        :param in_features: The number of input features.
        :param depth: The number of hidden layers of the conditioner.
        :param units: The number of units of each hidden layer of the conditioner.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super().__init__(in_features)
        self.affine = affine
        self.reverse = reverse

        # Initialize the coupling masks
        mask, inv_mask = self.build_alternating_masks()
        if reverse:
            mask, inv_mask = inv_mask, mask
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        self.register_buffer('inv_mask', torch.tensor(inv_mask, dtype=torch.float32))

        # Build the conditioner neural network
        layers = []
        in_features = self.in_features
        out_features = units
        for _ in range(depth):
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True)
            ])
            in_features = out_features
        out_features = self.in_features * 2 if affine else self.in_features
        layers.append(nn.Linear(in_features, out_features))
        self.network = nn.Sequential(*layers)

        # Build the activation function for the scale of affine transformation
        if affine:
            self.scale_act = ScaledTanh()

    def build_alternating_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the alterning masks.

        :return: The alterning mask and its inverse.
        """
        mask = np.arange(self.in_features) % 2
        inv_mask = 1.0 - mask
        return mask, inv_mask

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Evaluate the neural network conditioner
        z = self.network(self.mask * x)

        if self.affine:
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_act(s)
            t = self.inv_mask * t
            s = self.inv_mask * s
            u = (x - t) * torch.exp(-s)
            inv_log_det_jacobian = -torch.sum(s, dim=1)
        else:
            t = self.inv_mask * z
            u = x - t
            inv_log_det_jacobian = 0.0
        return u, inv_log_det_jacobian

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Evaluate the neural network conditioner
        z = self.network(self.mask * u)

        if self.affine:
            t, s = torch.chunk(z, chunks=2, dim=1)
            s = self.scale_act(s)
            t = self.inv_mask * t
            s = self.inv_mask * s
            x = u * torch.exp(s) + t
            log_det_jacobian = torch.sum(s, dim=1)
        else:
            t = self.inv_mask * z
            x = u + t
            log_det_jacobian = 0.0
        return x, log_det_jacobian


class CouplingLayer2d(Bijector):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        network: str,
        n_blocks: int,
        channels: int,
        affine: bool = True,
        channelwise: bool = False,
        reverse: bool = False
    ):
        """
        Build a RealNVP (or NICE) 2D coupling layer.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :param channelwise: Whether to use channel-wise coupling mask.
                            Defaults to False, i.e. checkerboard coupling mask.
        :param reverse: Whether to reverse the mask used in the coupling layer. Useful for alternating masks.
        """
        super().__init__(in_features)
        self.affine = affine
        self.channelwise = channelwise
        self.reverse = reverse

        if not channelwise:
            # Initialize the coupling masks
            mask, inv_mask = self.build_checkerboard_masks()
            if reverse:
                mask, inv_mask = inv_mask, mask
            self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
            self.register_buffer('inv_mask', torch.tensor(inv_mask, dtype=torch.float32))

        # Build the conditioner neural network
        in_channels = self.in_channels // 2 if channelwise else self.in_channels
        out_channels = in_channels * 2 if affine else in_channels
        if network == 'resnet':
            self.network = ResidualNetwork(in_channels, channels, out_channels, n_blocks)
        elif network == 'densenet':
            self.network = DenseNetwork(in_channels, channels, out_channels, n_blocks)
        else:
            raise NotImplementedError("Unknown network conditioner {}".format(network))

        # Build the activation function for the scale of affine transformation
        if affine:
            self.scale_act = ScaledTanh([in_channels, 1, 1])

    @property
    def in_channels(self) -> int:
        return self.in_features[0]

    @property
    def in_height(self) -> int:
        return self.in_features[1]

    @property
    def in_width(self) -> int:
        return self.in_features[2]

    def build_checkerboard_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the checkerboard coupling masks.

        :return: The checkerboard mask and its inverse.
        """
        mask = np.sum(np.indices([1, self.in_height, self.in_width]), axis=0) % 2
        inv_mask = 1.0 - mask
        return mask, inv_mask

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        if self.channelwise:
            if self.reverse:
                mx, my = torch.chunk(x, chunks=2, dim=1)
            else:
                my, mx = torch.chunk(x, chunks=2, dim=1)

            # Evaluate the neural network conditioner
            z = self.network(mx)

            if self.affine:
                # Apply the affine transformation (backward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_act(s)
                my = (my - t) * torch.exp(-s)
                inv_log_det_jacobian = -torch.sum(s.view(batch_size, -1), dim=1)
            else:
                # Apply the translation-only transformation (backward mode)
                my = my - z
                inv_log_det_jacobian = 0.0

            if self.reverse:
                u = torch.cat([mx, my], dim=1)
            else:
                u = torch.cat([my, mx], dim=1)
        else:
            # Evaluate the neural network conditioner
            mx = self.mask * x
            z = self.network(mx)

            if self.affine:
                # Apply the affine transformation (backward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_act(s)
                t = self.inv_mask * t
                s = self.inv_mask * s
                u = (x - t) * torch.exp(-s)
                inv_log_det_jacobian = -torch.sum(s.view(batch_size, -1), dim=1)
            else:
                # Apply the translation-only transformation (backward mode)
                t = self.inv_mask * z
                u = x - t
                inv_log_det_jacobian = 0.0
        return u, inv_log_det_jacobian

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]

        if self.channelwise:
            if self.reverse:
                mu, mv = torch.chunk(u, chunks=2, dim=1)
            else:
                mv, mu = torch.chunk(u, chunks=2, dim=1)

            # Get the parameters
            z = self.network(mu)

            if self.affine:
                # Apply the affine transformation (forward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_act(s)
                mv = mv * torch.exp(s) + t
                log_det_jacobian = torch.sum(s.view(batch_size, -1), dim=1)
            else:
                # Apply the translation-only transformation (forward mode)
                mv = mv + z
                log_det_jacobian = 0.0

            if self.reverse:
                x = torch.cat([mu, mv], dim=1)
            else:
                x = torch.cat([mv, mu], dim=1)
        else:
            # Get the parameters
            mu = self.mask * u
            z = self.network(mu)

            if self.affine:
                # Apply the affine transformation (forward mode)
                t, s = torch.chunk(z, chunks=2, dim=1)
                s = self.scale_act(s)
                t = self.inv_mask * t
                s = self.inv_mask * s
                x = u * torch.exp(s) + t
                log_det_jacobian = torch.sum(s.view(batch_size, -1), dim=1)
            else:
                # Apply the translation-only transformation (forward mode)
                t = self.inv_mask * z
                x = u + t
                log_det_jacobian = 0.0

        return x, log_det_jacobian


class CouplingBlock2d(Bijector):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        network: str,
        n_blocks: int,
        channels: int,
        affine: bool = True,
        last_block: bool = False
    ):
        """
        Build a RealNVP (or NICE) 2D coupling block,
        consisting of check-board/channel-wise couplings and squeeze operation.

        :param in_features: The size of the input.
        :param network: The network conditioner to use. It can be either 'resnet' or 'densenet'.
        :param n_blocks: The number of residual blocks or dense blocks.
        :param channels: The number of output channels of each convolutional layer.
        :param affine: Whether to use affine transformation. If False then use only translation (as in NICE).
        :param last_block: Whether it is the last block (i.e. no channelwise-masked couplings) or not.
        """
        super().__init__(in_features)
        self.last_block = last_block

        # Build the input couplings (consisting of 3 checkerboard-masked couplings)
        self.in_couplings = nn.ModuleList([
            CouplingLayer2d(
                self.in_features, network, n_blocks, channels, affine,
                channelwise=False, reverse=False
            ),
            BatchNormLayer2d(self.in_channels),
            CouplingLayer2d(
                self.in_features, network, n_blocks, channels, affine,
                channelwise=False, reverse=True
            ),
            BatchNormLayer2d(self.in_channels),
            CouplingLayer2d(
                self.in_features, network, n_blocks, channels, affine,
                channelwise=False, reverse=False
            ),
            BatchNormLayer2d(self.in_channels)
        ])

        if self.last_block:
            # Add an additional checkerboard-masked coupling layer
            self.in_couplings.extend([
                CouplingLayer2d(
                    self.in_features, network, n_blocks, channels, affine,
                    channelwise=False, reverse=True
                ),
                BatchNormLayer2d(self.in_channels)
            ])
        else:
            # Compute the size of the input (after squeezing operation)
            squeezed_channels = self.in_channels * 4
            squeezed_features = (squeezed_channels, self.in_height // 2, self.in_width // 2)

            # Double the number of channels
            channels *= 2

            # Build the output couplings (consisting of 3 channel-wise-masked couplings)
            self.out_couplings = nn.ModuleList([
                CouplingLayer2d(
                    squeezed_features, network, n_blocks, channels, affine,
                    channelwise=True, reverse=False
                ),
                BatchNormLayer2d(squeezed_channels),
                CouplingLayer2d(
                    squeezed_features, network, n_blocks, channels, affine,
                    channelwise=True, reverse=True
                ),
                BatchNormLayer2d(squeezed_channels),
                CouplingLayer2d(
                    squeezed_features, network, n_blocks, channels, affine,
                    channelwise=True, reverse=False
                ),
                BatchNormLayer2d(squeezed_channels)
            ])

    @property
    def in_channels(self) -> int:
        return self.in_features[0]

    @property
    def in_height(self) -> int:
        return self.in_features[1]

    @property
    def in_width(self) -> int:
        return self.in_features[2]

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_log_det_jacobian = 0.0

        # Pass through the checkerboard-masked couplings
        for layer in self.in_couplings:
            x, ildj = layer.apply_backward(x)
            inv_log_det_jacobian += ildj

        if not self.last_block:
            # Squeeze the inputs
            x = squeeze_depth2d(x)

            # Pass through the channel-wise-masked couplings
            for layer in self.out_couplings:
                x, ildj = layer.apply_backward(x)
                inv_log_det_jacobian += ildj

            # Un-squeeze the outputs
            x = unsqueeze_depth2d(x)

        return x, inv_log_det_jacobian

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_jacobian = 0.0

        if not self.last_block:
            # Squeeze the inputs
            u = squeeze_depth2d(u)

            # Pass through the channel-wise-masked couplings
            for layer in reversed(self.out_couplings):
                u, ldj = layer.apply_forward(u)
                log_det_jacobian += ldj

            # Un-squeeze the inputs
            u = unsqueeze_depth2d(u)

        # Pass through the checkerboard-masked couplings
        for layer in reversed(self.in_couplings):
            u, ldj = layer.apply_forward(u)
            log_det_jacobian += ldj

        return u, log_det_jacobian
