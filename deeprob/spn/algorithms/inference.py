# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Union, Tuple

import numpy as np

from deeprob.context import ContextState
from deeprob.spn.structure.node import Node, Sum
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.algorithms.evaluation import eval_bottom_up, eval_top_down


def likelihood(
    root: Node,
    x: np.ndarray,
    return_results: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :param return_results: A flag indicating if this function must return the likelihoods of each node of the SPN.
    :return: The likelihood values. Additionally it returns the likelihood values of each node.
    """
    return eval_bottom_up(
        root, x,
        leaf_func=node_likelihood,
        node_func=node_likelihood,
        return_results=return_results
    )


def log_likelihood(
    root: Node,
    x: np.ndarray,
    return_results: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the logarithmic likelihoods of the SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :return: The log likelihood values. Additionally it returns the log likelihood values of each node.
    """
    return eval_bottom_up(
        root, x,
        leaf_func=node_log_likelihood,
        node_func=node_log_likelihood,
        return_results=return_results
    )


def mpe(root: Node, x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Compute the Maximum Posterior Estimate of a SPN given some inputs.

    :param root: The root of the SPN.
    :param x: The inputs. They can be marginalized using NaNs.
    :param inplace: Whether to make inplace assignments.
    :return: The NaN-filled inputs.
    """
    _, lls = log_likelihood(root, x, return_results=True)
    with ContextState(check_spn=False):  # We already checked the SPN in forward mode
        return eval_top_down(
            root, x, lls,
            leaf_func=leaf_mpe,
            sum_func=sum_mpe,
            inplace=inplace
        )


def node_likelihood(node: Node, x: np.ndarray) -> np.ndarray:
    """
    Compute the likelihood of a node given the list of likelihoods of its children.

    :param node: The internal node.
    :param x: The array of likelihoods of the children.
    :return: The likelihoods of the node given the inputs.
    """
    ls = node.likelihood(x)
    return np.squeeze(ls, axis=1)


def node_log_likelihood(node: Node, x: np.ndarray) -> np.ndarray:
    """
    Compute the log-likelihood of a node given the list of log-likelihoods of its children.

    :param node: The internal node.
    :param x: The array of log-likelihoods of the children.
    :return: The log-likelihoods of the node given the inputs.
    """
    lls = node.log_likelihood(x)
    return np.squeeze(np.maximum(lls, -1e31), axis=1)


def leaf_mpe(node: Leaf, x: np.ndarray) -> np.ndarray:
    """
    Compute the maximum likelihood estimate of a leaf node.

    :param node: The leaf node.
    :param x: The inputs with some NaN values.
    :return: The most proable explanation.
    """
    return node.mpe(x)


def sum_mpe(node: Sum, lls: np.ndarray) -> np.ndarray:
    """
    Choose the branch that maximize the posterior estimate likelihood.

    :param node: The sum node.
    :param lls: The log-likelihoods of the children nodes.
    :return: The branch that maximize the posterior estimate likelihood.
    """
    weighted_lls = lls + np.log(node.weights)
    return np.argmax(weighted_lls, axis=1)
