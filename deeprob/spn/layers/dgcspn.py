# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from itertools import product
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deeprob.torch.initializers import dirichlet_


class SpatialGaussianLayer(nn.Module):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        out_channels: int,
        optimize_scale: bool = False,
        dropout: Optional[float] = None,
        quantiles_loc: Optional[np.ndarray] = None,
        uniform_loc: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of output channels.
        :param optimize_scale: Whether to optimize scale.
        :param dropout: The leaf nodes dropout rate. It can be None.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        :raises ValueError: If both quantiles_loc and uniform_loc are specified.
        """
        if quantiles_loc is not None and uniform_loc is not None:
            raise ValueError("At most one between quantiles_loc and uniform_loc can be specified")

        super().__init__()
        self.in_features = in_features
        self.out_features = (out_channels, self.in_height, self.in_width)
        self.dropout = dropout

        # Instantiate the location parameter
        if quantiles_loc is not None:
            self.loc = nn.Parameter(
                torch.tensor(quantiles_loc, dtype=torch.float32),
                requires_grad=True
            )
        elif uniform_loc is not None:
            low, high = uniform_loc
            linspace = torch.linspace(low, high, steps=self.out_channels).view(-1, 1, 1, 1)
            self.loc = nn.Parameter(
                linspace.repeat(1, *self.in_features),
                requires_grad=True
            )
        else:
            self.loc = nn.Parameter(
                torch.randn(self.out_channels, *self.in_features),
                requires_grad=True
            )

        # Instantiate the scale parameter
        if optimize_scale:
            self.scale = torch.nn.Parameter(
                0.5 + 0.1 * torch.tanh(torch.randn(self.out_channels, *self.in_features)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(self.out_channels, *self.in_features),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale, validate_args=False)

    @property
    def in_channels(self) -> int:
        return self.in_features[0]

    @property
    def in_height(self) -> int:
        return self.in_features[1]

    @property
    def in_width(self) -> int:
        return self.in_features[2]

    @property
    def out_channels(self) -> int:
        return self.out_features[0]

    @property
    def out_height(self) -> int:
        return self.out_features[1]

    @property
    def out_width(self) -> int:
        return self.out_features[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1)
        x = self.distribution.log_prob(x)

        # Apply the input dropout, if specified
        if self.training and self.dropout is not None:
            x[torch.lt(torch.rand_like(x), self.dropout)] = np.nan

        # Marginalize missing values (denoted with NaNs)
        torch.nan_to_num_(x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)


class SpatialProductLayer(nn.Module):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        kernel_size: Union[int, Tuple[int, int]],
        padding: str,
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        depthwise: bool = True
    ):
        """
        Initialize a Spatial Product layer.

        :param in_features: The number of input features.
        :param kernel_size: The size of the kernels.
        :param stride: The strides to use.
        :param padding: The padding mode to use. It can be 'valid', 'full' or 'final'.
                        Valid padding means no padding used. Full padding means padding is used based on effective
                        kernel size. Final padding means one-side padding (

        :param dilation: The space between the kernel points.
        :param depthwise: Whether to use depthwise convolutions. If False, random sparse kernels are used.
        :raises ValueError: If a parameter is out of domain.
        """
        super().__init__()
        self.in_features = in_features
        self.groups = self.in_channels if depthwise else 1

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        # Compute the effective kernel size, due to dilation
        kh, kw = kernel_size
        keh = (kh - 1) * self.dilation[0] + 1
        kew = (kw - 1) * self.dilation[1] + 1

        # Initialize the padding to apply
        if padding == 'valid':
            self.pad = [0, 0, 0, 0]
        elif padding == 'full':
            self.pad = [kew - 1, kew - 1, keh - 1, keh - 1]
        elif padding == 'final':
            self.pad = [0, (kew - 1) * 2 - self.in_width, 0, (keh - 1) * 2 - self.in_height]
        else:
            raise ValueError("Padding mode must be either 'valid', 'full' or 'final'")

        # Compute the number of output features
        kernel_dim = kh * kw
        out_h = self.pad[2] + self.pad[3] + self.in_height - keh + 1
        out_w = self.pad[0] + self.pad[1] + self.in_width - kew + 1
        out_h = int(np.ceil(out_h / self.stride[0]))
        out_w = int(np.ceil(out_w / self.stride[1]))
        out_c = self.in_channels if depthwise else self.in_channels ** kernel_dim
        self.out_features = (out_c, out_h, out_w)

        # Build the convolution kernels
        if not depthwise:
            # Consider all the possible combinations of previous layer's node ids (along the channel dimension)
            kernel_ids = np.array(list(product(range(self.in_channels), repeat=kernel_dim)))
            kernel_ids = np.reshape(kernel_ids, [self.out_channels, 1, kh, kw])
            channel_ids = np.expand_dims(np.arange(self.in_channels), axis=[0, 2, 3])
            channel_ids = np.tile(channel_ids, [self.out_channels, 1, kh, kw])
            weight = torch.tensor(np.equal(channel_ids, kernel_ids), dtype=torch.float32)
        else:
            weight = torch.ones(self.out_channels, 1, kh, kw)

        # Initialize the weight buffer
        self.register_buffer('weight', weight)

    @property
    def in_channels(self) -> int:
        return self.in_features[0]

    @property
    def in_height(self) -> int:
        return self.in_features[1]

    @property
    def in_width(self) -> int:
        return self.in_features[2]

    @property
    def out_channels(self) -> int:
        return self.out_features[0]

    @property
    def out_height(self) -> int:
        return self.out_features[1]

    @property
    def out_width(self) -> int:
        return self.out_features[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Pad the inputs and compute the log-likelihoods
        x = F.pad(x, self.pad)
        return F.conv2d(
            x, self.weight,
            stride=self.stride, dilation=self.dilation, groups=self.groups
        )


class SpatialSumLayer(nn.Module):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        out_channels: int,
        dropout: Optional[float] = None
    ):
        """
        Initialize a Spatial Sum layer.

        :param in_features: The number of input features.
        :param out_channels: The number of output channels.
        :param dropout: The input nodes dropout rate. It can be None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = (out_channels, self.in_height, self.in_width)
        self.dropout = dropout

        # Initialize the weight tensor
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, *self.in_features),
            requires_grad=True
        )
        dirichlet_(self.weight, alpha=1.0, dim=1)

    @property
    def in_channels(self) -> int:
        return self.in_features[0]

    @property
    def in_height(self) -> int:
        return self.in_features[1]

    @property
    def in_width(self) -> int:
        return self.in_features[2]

    @property
    def out_channels(self) -> int:
        return self.out_features[0]

    @property
    def out_height(self) -> int:
        return self.out_features[1]

    @property
    def out_width(self) -> int:
        return self.out_features[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Apply the dropout, if specified
        if self.training and self.dropout is not None:
            x[torch.lt(torch.rand_like(x), self.dropout)] = -np.inf

        # Compute the log-likelihood using the "logsumexp" trick
        w = torch.log_softmax(self.weight, dim=1)  # (out_channels, in_channels, in_height, in_width)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_channels, in_height, in_width)
        x = torch.logsumexp(x + w, dim=2)          # (-1, out_channels, in_height, in_width)
        return x


class SpatialRootLayer(nn.Module):
    def __init__(
        self,
        in_features: Tuple[int, int, int],
        out_channels: int,
    ):
        """
        Initialize a Spatial Root layer.

        :param in_features: The number of input features.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels

        # Initialize the weight tensor
        in_flatten_size = np.prod(self.in_features).item()
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, in_flatten_size),
            requires_grad=True
        )
        dirichlet_(self.weight, alpha=1.0)

    @property
    def in_channels(self) -> int:
        return self.in_size[0]

    @property
    def in_height(self) -> int:
        return self.in_size[1]

    @property
    def in_width(self) -> int:
        return self.in_size[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the log-likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)
        w = torch.log_softmax(self.weight, dim=1)  # (out_channels, in_flatten_size)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_flatten_size)
        x = torch.logsumexp(x + w, dim=2)          # (-1, out_channels)
        return x
