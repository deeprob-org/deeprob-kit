# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional

import numpy as np
from tqdm import tqdm

from deeprob.context import ContextState
from deeprob.utils.random import RandomState, check_random_state
from deeprob.spn.utils.filter import filter_nodes_by_type
from deeprob.spn.utils.validity import check_spn
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.algorithms.gradient import eval_backward


def expectation_maximization(
    root: Node,
    data: np.ndarray,
    num_iter: int = 100,
    batch_perc: float = 0.1,
    step_size: float = 0.5,
    random_init: bool = True,
    random_state: Optional[RandomState] = None,
    verbose: bool = True
) -> Node:
    """
    Learn the parameters of a SPN by batch Expectation-Maximization (EM).
    See https://arxiv.org/abs/1604.07243 and https://arxiv.org/abs/2004.06231 for details.

    :param root: The spn structure.
    :param data: The data to use to learn the parameters.
    :param num_iter: The number of iterations.
    :param batch_perc: The percentage of data to use for each step.
    :param step_size: The step size for batch EM.
    :param random_init: Whether to random initialize the weights of the SPN.
    :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
    :param verbose: Whether to enable verbose learning.
    :return: The spn with learned parameters.
    :raises ValueError: If a parameter is out of domain.
    """
    if num_iter <= 0:
        raise ValueError("The number of iterations must be positive")
    if batch_perc <= 0.0 or batch_perc >= 1.0:
        raise ValueError("The batch percentage must be in (0, 1)")
    if step_size <= 0.0 or step_size >= 1.0:
        raise ValueError("The step size must be in (0, 1)")

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    # Compute the batch size
    n_samples = len(data)
    batch_size = int(batch_perc * n_samples)

    # Compute a list-based cache for accessing nodes
    cached_nodes = {
        'sum': filter_nodes_by_type(root, Sum),
        'leaf': filter_nodes_by_type(root, Leaf)
    }

    # Check the random state
    random_state = check_random_state(random_state)

    # Random initialize the parameters of the SPN, if specified
    if random_init:
        # Initialize the sum parameters
        for node in cached_nodes['sum']:
            node.em_init(random_state)

        # Initialize the leaf parameters
        for node in cached_nodes['leaf']:
            node.em_init(random_state)

    # Initialize the tqdm bar, if verbose is specified
    iterator = range(num_iter)
    if verbose:
        iterator = tqdm(
            iterator, leave=None, unit='batch',
            bar_format='{desc}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    for _ in iterator:
        # Sample a batch of data randomly with uniform distribution
        batch_indices = random_state.choice(n_samples, size=batch_size, replace=False)
        batch_data = data[batch_indices]

        # Prevent checking the SPN at every forward inference step, we already did that!
        with ContextState(check_spn=False):
            # Forward step, obtaining the LLs at each node
            root_ll, lls = log_likelihood(root, batch_data, return_results=True)
        mean_ll = np.mean(root_ll)

        # Backward step, compute the log-gradients required to compute the sufficient statistics
        grads = eval_backward(root, lls)

        # Update the weights of each sum node
        for node in cached_nodes['sum']:
            children_ll = lls[list(map(lambda c: c.id, node.children))]
            stats = np.exp(children_ll - root_ll + grads[node.id])
            node.em_step(stats, step_size)

        # Update the parameters of each leaf node
        for node in cached_nodes['leaf']:
            stats = np.exp(lls[node.id] - root_ll + grads[node.id])
            node.em_step(stats, batch_data[:, node.scope], step_size)

        # Update the progress bar
        if verbose:
            iterator.set_description('Batch Avg. LL: {:.4f}'.format(mean_ll))

    return root
