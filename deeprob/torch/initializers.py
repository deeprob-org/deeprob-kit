# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import torch
from torch import distributions


def dirichlet_(tensor: torch.Tensor, alpha: float = 1.0, log_space: bool = True, dim: int = -1):
    """
    Initialize a tensor using the symmetric Dirichlet distribution.

    :param tensor: The tensor to initialize.
    :param alpha: The concentration parameter.
    :param log_space: Whether to initialize the tensor in the logarithmic space.
    :param dim: The dimension over which to sample.
    """
    shape = tensor.shape
    if len(shape) == 0:
        raise ValueError("Singleton tensors are not valid")
    min_dim, max_dim = -len(shape), len(shape) - 1
    if dim not in range(min_dim, max_dim):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(min_dim, max_dim, dim)
        )
    idx = (len(shape) + dim) % len(shape)
    with torch.no_grad():
        concentration = torch.full([shape[idx]], alpha)
        dirichlet = distributions.Dirichlet(concentration)
        samples = dirichlet.sample([d for i, d in enumerate(shape) if i != idx])
        if log_space:
            samples = torch.log(samples)
        tensor.copy_(torch.transpose(samples, idx, -1))
