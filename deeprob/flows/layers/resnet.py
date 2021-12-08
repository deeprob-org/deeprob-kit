# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import torch
from torch import nn

from deeprob.torch.utils import WeightNormConv2d


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int):
        """
        Build a basic residual block as in ResNet.

        :param n_channels: The number of channels.
        """
        super().__init__()

        # Build the residual block
        self.block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the residual block.

        :param x: The inputs.
        :return: The outputs.
        """
        return x + self.block(x)


class ResidualNetwork(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, n_blocks: int):
        """
        Initialize a residual network (ResNet) with skip connections.

        :param in_channels: The number of input channels.
        :param mid_channels: The number of mid channels.
        :param out_channels: The number of output channels.
        :param n_blocks: The number of residual blocks.
        :raises ValueError: If a parameter is out of domain.
        """
        if n_blocks <= 0:
            raise ValueError("The number of residual blocks must be positve")

        super().__init__()
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        # Build the input convolutional layer and input skip layer
        self.in_conv = WeightNormConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_skip = WeightNormConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        # Build the lists of residual blocks and skip connections
        for _ in range(n_blocks):
            self.blocks.append(ResidualBlock(mid_channels))
            self.skips.append(WeightNormConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True))

        # Build the output network
        self.out_network = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the residual network.

        :param x: The inputs.
        :return: The outputs.
        """
        # Pass through the input layers
        x = self.in_conv(x)
        z = self.in_skip(x)

        # Pass through the residual blocks
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            z += skip(x)

        # Pass through the output network
        x = self.out_network(z)
        return x
