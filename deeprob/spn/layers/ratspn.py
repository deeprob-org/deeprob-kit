# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import abc
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch import distributions

from deeprob.torch.initializers import dirichlet_


class RegionGraphLayer(abc.ABC, nn.Module):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        regions: List[tuple],
        rg_depth: int,
        dropout: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize a Region Graph-based base distribution.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param dropout: The leaf nodes dropout rate. It can be None.
        """
        super().__init__()
        self.in_features = in_features
        self.in_regions = len(regions)
        self.out_channels = out_channels
        self.rg_depth = rg_depth
        self.dropout = dropout
        self.distribution = None

        # Compute the padding and the number of features for each base distribution batch
        self.pad = -self.in_features % (2 ** self.rg_depth)
        in_features_pad = self.in_features + self.pad
        self.dimension = in_features_pad // (2 ** self.rg_depth)

        # Append dummy variables to regions orderings and update the pad mask
        mask = regions.copy()
        if self.pad > 0:
            pad_mask = np.zeros(shape=(len(regions), 1, self.dimension), dtype=np.bool_)
            for i, region in enumerate(regions):
                n_dummy = self.dimension - len(region)
                if n_dummy > 0:
                    pad_mask[i, :, -n_dummy:] = True
                    mask[i] = mask[i] + (mask[i][-1],) * n_dummy
            self.register_buffer('pad_mask', torch.tensor(pad_mask))
        self.register_buffer('mask', torch.tensor(mask))

        # Build the flatten inverse mask
        inv_mask = torch.argsort(torch.reshape(self.mask, [-1, in_features_pad]), dim=1)
        self.register_buffer('inv_mask', inv_mask)

        # Build the flatten inverted pad mask
        if self.pad > 0:
            inv_pad_mask = torch.reshape(self.pad_mask, [-1, in_features_pad])
            inv_pad_mask = torch.gather(inv_pad_mask, dim=1, index=self.inv_mask)
            self.register_buffer('inv_pad_mask', inv_pad_mask)

    def unpad_samples(self, x: torch.Tensor, idx_group: torch.Tensor) -> torch.Tensor:
        """
        Reorder and unpad some samples.

        :param x: The samples.
        :param idx_group: The group indices.
        :return: The reordered samples with padding dummy variables removed.
        """
        n_samples = idx_group.shape[0]

        # Reorder the samples
        idx_repetitions = torch.div(idx_group[:, 0], 2 ** self.rg_depth, rounding_mode='floor')
        samples = torch.gather(x, dim=1, index=self.inv_mask[idx_repetitions])

        # Remove the padding, if required
        if self.pad > 0:
            samples = samples[self.inv_pad_mask[idx_repetitions]].view(n_samples, self.in_features)
        return samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the layer on some inputs.

        :param x: The inputs.
        :return: The log-likelihoods of each distribution leaf.
        """
        # Gather the inputs and compute the log-likelihoods
        x = torch.unsqueeze(x[:, self.mask], dim=2)
        x = self.distribution.log_prob(x)

        # Apply the input dropout, if specified
        if self.training and self.dropout is not None:
            x[torch.lt(torch.rand_like(x), self.dropout)] = np.nan

        # Marginalize missing values (denoted with NaNs)
        torch.nan_to_num_(x)

        # Pad to zeros
        if self.pad > 0:
            x.masked_fill_(self.pad_mask, 0.0)
        return torch.sum(x, dim=-1)

    @abc.abstractmethod
    def distribution_mode(self) -> torch.Tensor:
        """
        Get the mode of the distribution.

        :return: The mode of the distribution.
        """

    @torch.no_grad()
    def mpe(self, x: torch.Tensor, idx_group: torch.Tensor, idx_offset: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs. Random variables can be marginalized using NaN values.
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The samples having maximum at posteriori estimates on marginalized random variables.
        """
        # Get the maximum at posteriori estimation of the base distribution
        # and filter the base samples by the region and offset indices
        mode = self.distribution_mode()
        samples = torch.flatten(mode[idx_group, idx_offset], start_dim=1)

        # Reorder the variables and remove the padding (if required)
        samples = self.unpad_samples(samples, idx_group)

        # Assign the maximum at posteriori estimation to NaN random variables
        return torch.where(torch.isnan(x), samples, x)

    @torch.no_grad()
    def sample(self, idx_group: torch.Tensor, idx_offset: torch.Tensor) -> torch.Tensor:
        """
        Sample from a base distribution.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The computed samples.
        """
        n_samples = idx_group.shape[0]

        # Sample from the base distribution and filter the samples
        samples = self.distribution.sample([n_samples])
        samples = samples[torch.unsqueeze(torch.arange(n_samples), dim=1), idx_group, idx_offset]
        samples = torch.flatten(samples, start_dim=1)

        # Reorder the variables and remove the padding (if required)
        samples = self.unpad_samples(samples, idx_group)
        return samples


class GaussianLayer(RegionGraphLayer):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        regions: List[tuple],
        rg_depth: int,
        dropout: Optional[float] = None,
        uniform_loc: Optional[Tuple[float, float]] = None,
        optimize_scale: bool = False
    ):
        """
        Initialize a Gaussian distributions input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param dropout: The leaf nodes dropout rate. It can be None.
        :param uniform_loc: The optional uniform distribution parameters for location initialization.
        :param optimize_scale: Whether to optimize scale and location jointly.
        """
        super().__init__(in_features, out_channels, regions, rg_depth, dropout)

        # Instantiate the location variable
        if uniform_loc is None:
            self.loc = nn.Parameter(
                torch.randn(self.in_regions, self.out_channels, self.dimension),
                requires_grad=True
            )
        else:
            a, b = uniform_loc
            self.loc = nn.Parameter(
                a + (b - a) * torch.rand(self.in_regions, self.out_channels, self.dimension),
                requires_grad=True
            )

        # Instantiate the scale variable
        if optimize_scale:
            self.scale = nn.Parameter(
                0.5 + 0.1 * torch.tanh(torch.randn(self.in_regions, self.out_channels, self.dimension)),
                requires_grad=True
            )
        else:
            self.scale = nn.Parameter(
                torch.ones(self.in_regions, self.out_channels, self.dimension),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = distributions.Normal(self.loc, self.scale, validate_args=False)

    def distribution_mode(self) -> torch.Tensor:
        return self.distribution.mean


class BernoulliLayer(RegionGraphLayer):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        regions: List[tuple],
        rg_depth: int,
        dropout: Optional[float] = None
    ):
        """
        Initialize a Bernoulli distributions input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param dropout: The leaf nodes dropout rate. It can be None.
        """
        super().__init__(in_features, out_channels, regions, rg_depth, dropout)

        # Instantiate the logit distribution parameters
        self.logits = nn.Parameter(
            torch.randn(self.in_regions, self.out_channels, self.dimension),
            requires_grad=True
        )

        # Instantiate the multi-batch Bernoulli distribution
        self.distribution = distributions.Bernoulli(logits=self.logits, validate_args=False)

    def distribution_mode(self) -> torch.Tensor:
        probs = self.distribution.mean
        return (probs >= 0.5).float()


class ProductLayer(nn.Module):
    def __init__(
        self,
        in_regions: int,
        in_nodes: int
    ):
        """
        Initialize the Product layer.

        :param in_regions: The number of input regions.
        :param in_nodes: The number of input nodes per region.
        """
        super().__init__()
        self.in_regions = in_regions
        self.in_nodes = in_nodes
        self.out_partitions = in_regions // 2
        self.out_nodes = in_nodes ** 2

        # Initialize the mask used to compute the outer product
        mask = [True, False] * self.out_partitions
        self.register_buffer('mask', torch.tensor(mask))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the outer product (the "outer sum" in log domain)
        x1 = x[:,  self.mask]                                # (-1, out_partitions, in_nodes)
        x2 = x[:, ~self.mask]                                # (-1, out_partitions, in_nodes)
        x1 = torch.unsqueeze(x1, dim=3)                      # (-1, out_partitions, in_nodes, 1)
        x2 = torch.unsqueeze(x2, dim=2)                      # (-1, out_partitions, 1, in_nodes)
        x = x1 + x2                                          # (-1, out_partitions, in_nodes, in_nodes)
        x = x.view(-1, self.out_partitions, self.out_nodes)  # (-1, out_partitions, out_nodes)
        return x

    @torch.no_grad()
    def mpe(
        self,
        x: torch.Tensor,
        idx_group: torch.Tensor,
        idx_offset: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor]:
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs (not used here).
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        return self.sample(idx_group, idx_offset)

    @torch.no_grad()
    def sample(
        self,
        idx_group: torch.Tensor,
        idx_offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from a product layer.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        idx_offset0 = torch.div(idx_offset, self.in_nodes, rounding_mode='floor')
        idx_offset1 = torch.remainder(idx_offset, self.in_nodes)

        # Compute the corresponding group and offset indices
        idx_group = torch.flatten(
            torch.stack([idx_group * 2, idx_group * 2 + 1], dim=2),
            start_dim=1
        )
        idx_offset = torch.flatten(
            torch.stack([idx_offset0, idx_offset1], dim=2),
            start_dim=1
        )
        return idx_group, idx_offset


class SumLayer(nn.Module):
    def __init__(
        self,
        in_partitions: int,
        in_nodes: int,
        out_nodes: int,
        dropout: Optional[float] = None
    ):
        """
        Initialize the sum layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_nodes: The number of output nodes per region.
        :param dropout: The input nodes dropout rate. It can be None.
        """
        super().__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_regions = in_partitions
        self.out_nodes = out_nodes
        self.dropout = dropout

        # Instantiate the weights
        self.weight = nn.Parameter(
            torch.empty(self.out_regions, self.out_nodes, self.in_nodes),
            requires_grad=True
        )
        dirichlet_(self.weight, alpha=1.0)

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
        w = torch.log_softmax(self.weight, dim=2)  # (out_regions, out_nodes, in_nodes)
        x = torch.unsqueeze(x, dim=2)              # (-1, in_partitions, 1, in_nodes) with in_partitions = out_regions
        x = torch.logsumexp(x + w, dim=3)          # (-1, out_regions, out_nodes)
        return x

    @torch.no_grad()
    def mpe(
        self,
        x: torch.Tensor,
        idx_group: torch.Tensor,
        idx_offset: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor]:
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        # Compute the offset indices evaluating the sum nodes as an argmax
        x = x[torch.unsqueeze(torch.arange(x.shape[0]), dim=1), idx_group]
        w = torch.log_softmax(self.weight[idx_group, idx_offset], dim=2)
        idx_offset = torch.argmax(x + w, dim=2)
        return idx_group, idx_offset

    @torch.no_grad()
    def sample(
        self,
        idx_group: torch.Tensor,
        idx_offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from a sum layer.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        # Compute the indices by sampling from a categorical distribution that is parametrized by sum layer's weights
        w = torch.log_softmax(self.weight[idx_group, idx_offset], dim=2)
        idx_offset = distributions.Categorical(logits=w).sample()
        return idx_group, idx_offset


class RootLayer(nn.Module):
    def __init__(
        self,
        in_partitions: int,
        in_nodes: int,
        out_classes: int
    ):
        """
        Initialize the root layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_classes: The number of output nodes.
        """
        super().__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_classes = out_classes

        # Instantiate the weights
        self.weight = nn.Parameter(
            torch.empty(self.out_classes, self.in_partitions * self.in_nodes),
            requires_grad=True
        )
        dirichlet_(self.weight, alpha=1.0)

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The outputs.
        """
        # Compute the log-likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)          # (-1, in_partitions * in_nodes)
        w = torch.log_softmax(self.weight, dim=1)  # (out_classes, in_partitions * in_nodes)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_partitions * in_nodes)
        x = torch.logsumexp(x + w, dim=2)          # (-1, out_classes)
        return x

    @torch.no_grad()
    def mpe(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param y: The target classes.
        :return: The group and offset indices.
        """
        # Compute the layer top-down and get the group and offset indices
        x = torch.flatten(x, start_dim=1)
        w = torch.log_softmax(self.weight, dim=1)
        idx = torch.argmax(x + w[y], dim=1, keepdim=True)
        idx_group = torch.div(idx, self.in_nodes, rounding_mode='floor')
        idx_offset = torch.remainder(idx, self.in_nodes)
        return idx_group, idx_offset

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the root layer.

        :param y: The target classes.
        :return: The group and offset indices.
        """
        # Compute the layer top-down and get the indices
        w = torch.log_softmax(self.weight, dim=1)
        idx = distributions.Categorical(logits=w[y]).sample().unsqueeze(dim=1)
        idx_group = torch.div(idx, self.in_nodes, rounding_mode='floor')
        idx_offset = torch.remainder(idx, self.in_nodes)
        return idx_group, idx_offset
