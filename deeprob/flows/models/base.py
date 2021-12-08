# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Tuple

import torch
from torch import nn
from torch import distributions

from deeprob.torch.base import ProbabilisticModel, DensityEstimator
from deeprob.flows.utils import DequantizeLayer, LogitLayer


class NormalizingFlow(ProbabilisticModel):
    def __init__(
        self,
        in_features,
        dequantize: bool = False,
        logit: Optional[float] = None,
        in_base: Optional[DensityEstimator] = None
    ):
        """
        Initialize an abstract Normalizing Flow model.

        :param in_features: The input size.
        :param dequantize: Whether to apply the dequantization transformation.
        :param logit: The logit factor to use. Use None to disable the logit transformation.
        :param in_base: The input base distribution to use. If None, the standard Normal distribution is used.
        :raises ValueError: If the number of input features is invalid.
        :raises ValueError: If the logit value is invalid.
        """
        if not isinstance(in_features, int):
            if not isinstance(in_features, tuple) or len(in_features) != 3:
                raise ValueError("The number of input features must be either an int or a (C, H, W) tuple")

        super().__init__()
        self.in_features = in_features

        # Build the dequantization layer
        if dequantize:
            self.dequantize = DequantizeLayer(in_features)
        else:
            self.dequantize = None

        # Build the logit layer
        if logit is not None:
            if logit <= 0.0 or logit >= 1.0:
                raise ValueError("The logit factor must be in (0, 1)")
            self.logit = LogitLayer(in_features, alpha=logit)
        else:
            self.logit = None

        # Build the base distribution, if necessary
        if in_base is None:
            self.in_base_loc = nn.Parameter(torch.zeros(in_features), requires_grad=False)
            self.in_base_scale = nn.Parameter(torch.ones(in_features), requires_grad=False)
            self.in_base = distributions.Normal(self.in_base_loc, self.in_base_scale)
        else:
            self.in_base = in_base

        # Initialize the normalizing flow layers
        self.layers = nn.ModuleList()

    def train(self, mode: bool = True, base_mode: bool = True):
        """
        Set the training mode.

        :param mode: The training mode for the flows layers.
        :param base_mode: The training mode for the in_base distribution.
        :return: Itself.
        """
        self.training = mode
        self.layers.train(mode)
        if isinstance(self.in_base, torch.nn.Module):
            self.in_base.train(base_mode)
        return self

    def eval(self):
        """
        Turn off the training mode for both the flows layers and the in_base distribution.

        :return: Itself.
        """
        return self.train(False, False)

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the data batch before feeding it to the probabilistic model (forward mode).

        :param x: The input data batch.
        :return: The preprocessed data batch and the inv-log-det-jacobian.
        """
        inv_log_det_jacobian = 0.0
        if self.dequantize is not None:
            x, ildj = self.dequantize.apply_backward(x)
            inv_log_det_jacobian += ildj
        if self.logit is not None:
            x, ildj = self.logit.apply_backward(x)
            inv_log_det_jacobian += ildj
        return x, inv_log_det_jacobian

    def unpreprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the data batch before feeding it to the probabilistic model (backward mode).

        :param x: The input data batch.
        :return: The unpreprocessed data batch and the log-det-jacobian.
        """
        log_det_jacobian = 0.0
        if self.logit is not None:
            x, ldj = self.logit.apply_forward(x)
            log_det_jacobian += ldj
        if self.dequantize is not None:
            x, ldj = self.dequantize.apply_forward(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood given complete evidence.

        :param x: The inputs.
        :return: The log-likelihoods.
        """
        # Preprocess the samples
        batch_size = x.shape[0]
        x, inv_log_det_jacobian = self.preprocess(x)

        # Apply backward transformations
        x, ildj = self.apply_backward(x)
        inv_log_det_jacobian += ildj

        # Compute the prior log-likelihood
        base_lls = self.in_base.log_prob(x)
        prior = torch.sum(base_lls.view(batch_size, -1), dim=1)

        # Return the final log-likelihood
        return prior + inv_log_det_jacobian

    @torch.no_grad()
    def sample(self, n_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sample from the base distribution
        x = self.in_base.sample([n_samples])

        # Apply forward transformations
        x, _ = self.apply_forward(x)

        # Apply reversed preprocessing transformation
        x, _ = self.unpreprocess(x)
        return x

    def apply_backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the backward transformation.

        :param x: The inputs.
        :return: The transformed samples and the backward log-det-jacobian.
        """
        inv_log_det_jacobian = 0.0
        for layer in self.layers:
            x, ildj = layer.apply_backward(x)
            inv_log_det_jacobian += ildj
        return x, inv_log_det_jacobian

    def apply_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward transformation.

        :param x: the inputs.
        :return: The transformed samples and the forward log-det-jacobian.
        """
        log_det_jacobian = 0.0
        for layer in reversed(self.layers):
            x, ldj = layer.apply_forward(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def loss(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute the loss as the average negative log-likelihood
        return -torch.mean(x)
