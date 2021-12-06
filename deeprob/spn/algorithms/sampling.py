# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import numpy as np
from scipy import stats

from deeprob.context import ContextState
from deeprob.spn.structure.node import Node, Sum
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.algorithms.evaluation import eval_top_down


def sample(root: Node, x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Sample some features from the distribution represented by the SPN.

    :param root: The root of the SPN.
    :param x: The inputs with possible NaN values to fill with sampled values.
    :param inplace: Whether to make inplace assignments.
    :return: The inputs that are NaN-filled with samples from appropriate distributions.
    """
    # First evaluate the SPN bottom-up, then top-down
    _, lls = log_likelihood(root, x, return_results=True)
    with ContextState(check_spn=False):  # We already checked the SPN in forward mode
        return eval_top_down(
            root, x, lls,
            leaf_func=leaf_sample,
            sum_func=sum_sample,
            inplace=inplace
        )


def leaf_sample(node: Leaf, x: np.ndarray) -> np.ndarray:
    """
    Sample some values from the distribution leaf.

    :param node: The distribution leaf node.
    :param x: The inputs with possible NaN values to fill with sampled values.
    :return: The completed samples.
    """
    return node.sample(x)


def sum_sample(node: Sum, lls: np.ndarray) -> np.ndarray:
    """
    Choose the sub-distribution from which sample.

    :param node: The sum node.
    :param lls: The log-likelihoods of the children nodes.
    :return: The index of the sub-distribution to follow.
    """
    n_samples, n_features = lls.shape
    gumbel = stats.gumbel_l.rvs(0.0, 1.0, size=(n_samples, n_features))
    weighted_lls = lls + np.log(node.weights) + gumbel
    return np.argmax(weighted_lls, axis=1)
