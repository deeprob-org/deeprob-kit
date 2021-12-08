# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Union, Optional, List, Tuple

import abc
import torch
import torchvision.transforms.functional as F


class Transform(abc.ABC):
    """Generic data transform function."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate in forward mode the transformation.
        Equivalent to forward(x).

        :param x: The inputs.
        :return: The outputs.
        """
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate in forward mode the transformation.

        :param x: The inputs.
        :return: The outputs.
        """

    @abc.abstractmethod
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate in backward mode the transformation.

        :param x: The inputs.
        :return: The outputs.
        """


class TransformList(Transform, list):
    """A list of transformations."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self:
            x = transform.forward(x)
        return x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in reversed(self):
            x = transform.backward(x)
        return x


class Normalize(Transform):
    def __init__(
        self,
        mean: Union[float, torch.Tensor],
        std: Union[float, torch.Tensor],
        eps: float = 1e-7
    ):
        """
        Initialize a normalization transformation.
        This transformation computes the following equations:

        | y = (x - mean) / (std + eps)
        | x = y * (std + eps) + mean

        :param mean: The mean values. One for each channel.
        :param std: The standard deviation values.
        :param eps: The epsilon value (used to avoid divisions by zero).
        :raises ValueError: If the epsilon value is out of domain.
        """
        if eps <= 0.0:
            raise ValueError("The epsilon value must be positive")
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


class Quantize(Transform):
    def __init__(self, n_bits: int = 8):
        """
        Initialize a quantization transformation.
        This transformation computes the following equations:

        | y = clamp(floor(x * 2 ** n_bits), 0, 2 ** n_bits - 1) / (2 ** n_bits - 1)
        | x = ((x * (2 ** n_bits - 1)) + u) / (2 ** n_bits)
        | with u ~ Uniform(0, 1)

        :param n_bits: The number of bits.
        :raises ValueError: If the number of bits is not positive.
        """
        if n_bits <= 0:
            raise ValueError("The number of bits must be positive")
        self.n_bits = n_bits
        self.bins = 2 ** self.n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.floor(x * self.bins)
        x = torch.clamp(x, min=0, max=self.bins - 1) / (self.bins - 1)
        return x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (self.bins - 1)
        x = (x + torch.rand(x.size())) / self.bins
        return x


class Flatten(Transform):
    def __init__(
        self,
        shape: Optional[Union[torch.Size, List[int], Tuple[int, ...]]] = None
    ):
        """
        Initialize a flatten transformation.

        :param shape: The original tensor shape.
                      It can be None to enable only forward transformation.
        """
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shape is None:
            raise NotImplementedError("Flatten backward method not implemented because shape is None")
        return torch.reshape(x, self.shape)


class Reshape(Transform):
    def __init__(
        self,
        target_shape: Union[torch.Size, List[int], Tuple[int, ...]],
        shape: Union[torch.Size, List[int], Tuple[int, ...]] = None
    ):
        """
        Initialize a reshape transformation.

        :param target_shape: The target tensor shape.
        :param shape: The input tensor shape.
        """
        self.target_shape = target_shape
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, self.target_shape)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shape is None:
            raise NotImplementedError("Reshape backward method not implemented because shape is None")
        return torch.reshape(x, self.shape)


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        """
        Initialize a random horizontal flip transformation.

        :param p: The probability of flipping.
        :raises ValueError: If the probability of flipping is out of domain.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("The probability of flipping must be in (0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hflip(x) if torch.rand(1) < self.p else x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return x
