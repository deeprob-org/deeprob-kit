# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import abc
from typing import Union, Tuple

import numpy as np
import torch
from torch import nn


def squeeze_depth2d(x: torch.Tensor) -> torch.Tensor:
    """
    Squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C, H, W].
    :return: The output tensor of size [N, C * 4, H // 2, W // 2].
    """
    # This is literally 6D tensor black magic
    n, c, h, w = x.size()
    x = x.reshape(n, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(n, c * 4, h // 2, w // 2)
    return x


def unsqueeze_depth2d(x: torch.Tensor) -> torch.Tensor:
    """
    Un-squeeze operation (as in RealNVP).

    :param x: The input tensor of size [N, C * 4, H // 2, W // 2].
    :return: The output tensor of size [N, C, H, W].
    """
    # This is literally 6D tensor black magic
    n, c, h, w = x.size()
    x = x.reshape(n, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(n, c // 4, h * 2, w * 2)
    return x


class Bijector(abc.ABC, nn.Module):
    """Bijector abastract class."""
    def __init__(self, in_features: Union[int, Tuple[int, int, int]]):
        """
        Initialize a bijector module.

        :param in_features: The number of input features.
        :raises ValueError: If the number of input features is invalid.
        """
        if not isinstance(in_features, int):
            if not isinstance(in_features, tuple) or len(in_features) != 3:
                raise ValueError("The number of input features must be either an int or a (C, H, W) tuple")

        super().__init__()
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x: torch.Tensor, backward: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the bijector transformation.

        :param x: The inputs.
        :param backward: Whether to apply the backward transformation.
        :return: The transformed samples and the corresponding log-det-jacobian.
        """
        if backward:
            return self.apply_backward(x)
        return self.apply_forward(x)

    @abc.abstractmethod
    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the backward transformation.

        :param x: The inputs.
        :return: The transformed samples and the backward log-det-jacobian.
        """

    @abc.abstractmethod
    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward transformation.

        :param u: The inputs.
        :return: The transformed samples and the forward log-det-jacobian.
        """


class BatchNormLayer1d(Bijector):
    def __init__(self, in_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        Build a Batch Normalization 1D layer.

        :param in_features: The number of input features.
        :param momentum: The momentum used to update the running parameters.
        :param eps: Epsilon value, an arbitrarily small value.
        :raises ValueError: If a parameter is out of domain.
        """
        if momentum <= 0.0 or momentum >= 1.0:
            raise ValueError("The momentum value must be in (0, 1)")
        if eps <= 0.0:
            raise ValueError("The epsilon value must be positive")

        super().__init__(in_features)
        self.momentum = momentum
        self.eps = eps

        # Initialize the learnable parameters (used for training)
        self.weight = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=True)

        # Initialize the running parameters (used for inference)
        self.register_buffer('running_var', torch.ones(1, self.in_features))
        self.register_buffer('running_mean', torch.zeros(1, self.in_features))

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # Check if the module is training
        if self.training:
            # Get the mini batch statistics
            var, mean = torch.var_mean(x, dim=0, keepdim=True)

            # Update the running parameters
            self.running_var.mul_(self.momentum).add_(var.data * (1.0 - self.momentum))
            self.running_mean.mul_(self.momentum).add_(mean.data * (1.0 - self.momentum))
        else:
            # Get the running parameters as batch mean and variance
            mean = self.running_mean
            var = self.running_var

        # Apply the transformation
        var = var + self.eps
        u = (x - mean) / torch.sqrt(var)
        u = u * torch.exp(self.weight) + self.bias
        inv_log_det_jacobian = torch.sum(self.weight - 0.5 * torch.log(var))
        return u, inv_log_det_jacobian.expand(batch_size)

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]

        # Get the running parameters as batch mean and variance
        mean = self.running_mean
        var = self.running_var

        # Apply the transformation
        var = var + self.eps
        u = (u - self.bias) * torch.exp(-self.weight)
        x = u * torch.sqrt(var) + mean
        log_det_jacobian = torch.sum(-self.weight + 0.5 * torch.log(var))
        return x, log_det_jacobian.expand(batch_size)


class BatchNormLayer2d(Bijector):
    def __init__(self, in_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        Build a Batch Normalization 2D layer.

        :param in_features: The number of input features.
        :param momentum: The momentum used to update the running parameters.
        :param eps: An arbitrarily small value.
        :raises ValueError: If a parameter is out of domain.
        """
        if momentum <= 0.0 or momentum >= 1.0:
            raise ValueError("The momentum value must be in (0, 1)")
        if eps <= 0.0:
            raise ValueError("The epsilon value must be positive")

        super().__init__(in_features)
        self.momentum = momentum
        self.eps = eps

        # Initialize the learnable parameters (used for training)
        self.weight = nn.Parameter(torch.zeros(1, self.in_features, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, self.in_features, 1, 1), requires_grad=True)

        # Initialize the running parameters (used for inference)
        self.register_buffer('running_var', torch.ones(1, self.in_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, self.in_features, 1, 1))

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        grid_size = x.shape[2] * x.shape[3]

        # Check if the module is training
        if self.training:
            # Get the mini batch statistics
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.mean((x - mean) ** 2.0, dim=[0, 2, 3], keepdim=True)

            # Update the running parameters
            self.running_var.mul_(self.momentum).add_(var.data * (1.0 - self.momentum))
            self.running_mean.mul_(self.momentum).add_(mean.data * (1.0 - self.momentum))
        else:
            # Get the running parameters as batch mean and variance
            mean = self.running_mean
            var = self.running_var

        # Apply the transformation
        var = var + self.eps
        u = (x - mean) / torch.sqrt(var)
        u = u * torch.exp(self.weight) + self.bias
        inv_log_det_jacobian = torch.sum(self.weight - 0.5 * torch.log(var)) * grid_size
        return u, inv_log_det_jacobian.expand(batch_size)

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]
        grid_size = u.shape[2] * u.shape[3]

        # Get the running parameters as batch mean and variance
        mean = self.running_mean
        var = self.running_var

        # Apply the transformation
        var = var + self.eps
        u = (u - self.bias) * torch.exp(-self.weight)
        x = u * torch.sqrt(var) + mean
        log_det_jacobian = torch.sum(0.5 * torch.log(var) - self.weight) * grid_size
        return x, log_det_jacobian.expand(batch_size)


class DequantizeLayer(Bijector):
    def __init__(self, in_features: Union[int, Tuple[int, int, int]], n_bits: int = 8):
        """
        Build a Dequantization transformation layer.

        :param in_features: The number of input features.
        :param n_bits: The number of bits to use.
        :raises ValueError: If a parameter is out of domain.
        """
        if n_bits <= 0:
            raise ValueError("The number of bits must be positive")

        super().__init__(in_features)
        self.n_bits = n_bits
        self.bins = 2 ** self.n_bits

        # Cache the log-det-jacobian as a constant
        dims = np.prod(self.in_features)
        self.register_buffer('ldj', torch.tensor(dims * np.log(self.bins), dtype=torch.float32))

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        u = x * (self.bins - 1)  # In PyTorch the images are often normalized (see ToTensor()).
        u = (u + torch.rand_like(x)) / self.bins
        return u, -self.ldj.expand(batch_size)

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]
        x = torch.floor(u * self.bins)
        x = torch.clamp(x, min=0, max=self.bins - 1) / (self.bins - 1)
        return x, self.ldj.expand(batch_size)


class LogitLayer(Bijector):
    def __init__(self, in_features: Union[int, Tuple[int, int, int]], alpha: float = 0.05):
        """
        Build a Logit transformation layer.

        :param in_features: The number of input features.
        :param alpha: The alpha parameter for logit transformation.
        :raises ValueError: If a parameter is out of domain.
        """
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("The alpha logit parameter must be in (0, 1)")

        super().__init__(in_features)
        self.alpha = alpha

        # Cache part of the log-det-jacobian as a constant
        dims = np.prod(self.in_features)
        self.register_buffer('ldj', torch.tensor(-dims * np.log(1.0 - 2.0 * self.alpha), dtype=torch.float32))

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.alpha + (1.0 - 2.0 * self.alpha) * x
        lx = torch.log(x)
        rx = torch.log(1.0 - x)
        u = lx - rx
        v = lx + rx
        log_det_jacobian = torch.sum(v.view(batch_size, -1), dim=1) + self.ldj
        return u, -log_det_jacobian

    def apply_forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]
        u = torch.sigmoid(u)
        x = (u - self.alpha) / (1.0 - 2.0 * self.alpha)
        lu = torch.log(u)
        ru = torch.log(1.0 - u)
        v = lu + ru
        log_det_jacobian = torch.sum(v.view(batch_size, -1), dim=1) + self.ldj
        return x, log_det_jacobian
