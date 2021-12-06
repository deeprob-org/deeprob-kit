# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import abc
from typing import Optional, Union

import torch
from torch import nn
from torch import distributions


class ProbabilisticModel(abc.ABC, nn.Module):
    """Abstract Probabilistic Model base class."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood of a batched sample.
        Note that the nn.Module.forward method of sub-classes must implement log-likelihood evaluation.

        :param x: The batched sample.
        :return: The batched log-likelihoods.
        """
        return self.__call__(x)

    @abc.abstractmethod
    def sample(self, n_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample some values from the modeled distribution.

        :param n_samples: The number of samples.
        :param y: The samples labels. It can be None.
        :return: The samples.
        """

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the loss of the model.

        :param x: The outputs of the model.
        :param y: The ground-truth. It can be None.
        :return: The loss.
        """

    def apply_constraints(self):
        """
        Apply the constraints specified by the model.
        """


#: A density estimator is either a DeeProb-kit probabilistic model or a Torch distribution.
DensityEstimator = Union[ProbabilisticModel, distributions.Distribution]
