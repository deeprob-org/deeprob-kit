# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


def get_activation_class(name: str):
    """
    Get the activation function class by its name.

    :param name: The activation function's name.
                 It can be one of: 'relu', 'leaky-relu', 'softplus', 'tanh', 'sigmoid'.
    :return: The activation function class.
    :raises ValueError: If the activation function's name is not known.
    """
    try:
        return {
            'relu': nn.ReLU,
            'leaky-relu': nn.LeakyReLU,
            'softplus': nn.Softplus,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }[name]
    except KeyError as ex:
        raise ValueError from ex


def get_optimizer_class(name: str):
    """
    Get the optimizer class by its name.

    :param name: The optimizer's name. It can be 'sgd', 'rmsprop', 'adagrad', 'adam'.
    :return: The optimizer class.
    :raises ValueError: If the optimizer's name is not known.
    """
    try:
        return {
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adam': optim.Adam
        }[name]
    except KeyError as ex:
        raise ValueError from ex


class ScaledTanh(nn.Module):
    """Scaled Tanh activation module."""
    def __init__(self, weight_size: Union[int, tuple, list] = 1):
        """
        Build the module.

        :param weight_size: The size of the weight parameter.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(weight_size), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the scaled tanh function.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        return self.weight * torch.tanh(x)


class MaskedLinear(nn.Linear):
    """Masked version of linear layer."""
    def __init__(self, in_features: int, out_features: int, mask: np.ndarray):
        """
        Build a masked linear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param mask: The mask to apply to the weights of the layer.
        :raises ValueError: If the mask parameter is not consistent with the number of input and output features.
        """
        super().__init__(in_features, out_features)
        if mask.shape[0] != out_features or mask.shape[1] != in_features:
            raise ValueError("Inconsistent mask shape")
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        return F.linear(x, self.mask * self.weight, self.bias)


class WeightNormConv2d(nn.Module):
    """Conv2D with weight normalization."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True
    ):
        """
        Initialize a Conv2d layer with weight normalization.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param kernel_size: The convolving kernel size.
        :param stride: The stride of convolution.
        :param padding: The padding to apply.
        :param bias: Whether to use bias parameters.
        """
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the weight-normalized convolutional layer.

        :param x: The inputs.
        :return: The outputs of the module.
        """
        return self.conv(x)
