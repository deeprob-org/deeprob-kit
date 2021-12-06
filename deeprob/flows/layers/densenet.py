# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from deeprob.torch.utils import WeightNormConv2d


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_checkpoint: bool = False):
        """
        Initialize a dense layer as in DenseNet.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                               (by increasing training time caused by re-computations).
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Build the bottleneck network
        # Use 4 * out_channels as number of mid features channels
        mid_channels = 4 * out_channels
        self.bottleneck_network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        )

        # Build the main dense layer
        self.network = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def bottleneck(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Pass through the bottleneck layer.

        :param inputs: A list of previous feature maps.
        :return: The outputs of the bottleneck.
        """
        x = torch.cat(inputs, dim=1)
        return self.bottleneck_network(x)

    def checkpoint_bottleneck(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Pass through the bottleneck layer (by using a checkpoint).

        :param inputs: A list of previous feature maps.
        :return: The outputs of the bottleneck.
        """
        def closure(*inputs):
            return self.bottleneck(inputs)
        return checkpoint(closure, *inputs)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the dense layer.

        :param inputs: A list of previous feature maps.
        :return: The outputs of the layer.
        """
        # Pass through the bottleneck
        if self.use_checkpoint and any(map(lambda t: t.requires_grad, inputs)):
            x = self.checkpoint_bottleneck(inputs)
        else:
            x = self.bottleneck(inputs)

        # Pass through the main dense layer
        x = self.network(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, n_layers: int, in_channels: int, out_channels: int, use_checkpoint: bool = False):
        """
        Initialize a dense block as in DenseNet.

        :param n_layers: The number of dense layers.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                              (by increasing training time caused by re-computations).
        """
        super().__init__()
        self.layers = nn.ModuleList()

        # Build the dense layers
        for i in range(n_layers):
            self.layers.append(DenseLayer(
                in_channels + i * out_channels, out_channels, use_checkpoint=use_checkpoint
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dense block.

        :param x: The inputs.
        :return: The outputs.
        """
        outputs = [x]
        for layer in self.layers:
            x = layer(outputs)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        """
        Initialize a transition layer as in DenseNet.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param bias: Whether to use bias in the last convolutional layer.
        """
        super().__init__()

        # Build the transition layer
        self.network = torch.nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            WeightNormConv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        )

    def forward(self, x):
        """
        Evaluate the layer.

        :param x: The inputs.
        :return: The outputs of the layer.
        """
        return self.network(x)


class DenseNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        n_blocks: int,
        use_checkpoint: bool = False
    ):
        """
        Initialize a dense network (DenseNet) with only one dense block.

        :param in_channels: The number of input channels.
        :param mid_channels: The number of mid channels.
        :param out_channels: The number of output channels.
        :param n_blocks: The number of dense blocks.
        :param use_checkpoint: Whether to use a checkpoint in order to reduce memory usage
                              (by increasing training time caused by re-computations).
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        # Build the input convolutional layer
        self.in_conv = WeightNormConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        # Build the list of dense blocks and transition layers
        # Use four dense layer for each dense block
        for i in range(n_blocks):
            self.blocks.append(DenseBlock(4, mid_channels, mid_channels, use_checkpoint=use_checkpoint))
            if i == n_blocks - 1:
                self.blocks.append(Transition(5 * mid_channels, out_channels, bias=True))
            else:
                self.blocks.append(Transition(5 * mid_channels, mid_channels, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dense network.

        :param x: The inputs.
        :return: The outputs.
        """
        # Pass through the input convolutional layer
        x = self.in_conv(x)

        # Pass through the dense blocks
        for block in self.blocks:
            x = block(x)
        return x
