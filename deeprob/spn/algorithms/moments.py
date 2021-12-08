# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import numpy as np

from deeprob.spn.structure.node import Node
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.algorithms.inference import node_likelihood
from deeprob.spn.algorithms.evaluation import eval_bottom_up


def moment(root: Node, order: int = 1) -> np.ndarray:
    """
    Compute non-central moments of a given order of a smooth and decomposable SPN.

    :param root: The root of the SPN.
    :param order: The order of the moment. If scalar, it will be used for all the random variables.
    :return: The non-central moments with respect to each variable in the scope.
    :raises ValueError: If the order of the moment is negative.
    """
    scope = root.scope
    if order < 0:
        raise ValueError("The order of the moment must be non-negative")
    if order == 0:  # Completely skip computation for 0-order moments
        return np.ones(len(scope), dtype=np.float32)

    # Compute the moments w.r.t. each random variable by proceeding bottom-up
    moments = np.ones(shape=[len(scope), len(scope)], dtype=np.float32)
    return eval_bottom_up(
        root, moments,
        leaf_func=leaf_moment,
        node_func=node_likelihood,
        leaf_func_kwargs={'order': order}
    )


def leaf_moment(node: Leaf, x: np.ndarray, order: int) -> np.ndarray:
    """
    Compute the moment of a leaf node.

    :param node: The leaf node.
    :param x: The inputs of the leaf. Actually, it's used only to infer the output shape.
    :param order: The order of the moment.
    :return: The moment of the leaf node.
    """
    m = np.ones(len(x), dtype=np.float32)
    m[node.scope] = node.moment(k=order)
    return m


def expectation(root: Node) -> np.ndarray:
    """
    Compute the expectation values of a SPN w.r.t. each of the random variables.

    :param root: The root of the SPN.
    :return: The expectation w.r.t. each of the random variables.
    """
    return moment(root, order=1)


def variance(root: Node) -> np.ndarray:
    """
    Compute the variance values of a SPN w.r.t. each of the random variables.

    :param root: The root of the SPN.
    :return: The variance w.r.t. each of the random variables.
    """
    fst_moment = moment(root, order=1)
    snd_moment = moment(root, order=2)
    return snd_moment - fst_moment ** 2.0


def skewness(root: Node) -> np.ndarray:
    """
    Compute the skewness values of a SPN w.r.t. each of the random variables.

    :param root: The root of the SPN.
    :return: The skewness w.r.t. each of the random variables.
    """
    # This implementation is derived by expanding the third central moment
    # and obtaining a definition based on non-central moments
    fst_moment = moment(root, order=1)
    snd_moment = moment(root, order=2)
    thd_moment = moment(root, order=3)
    g1 = fst_moment ** 2.0
    g2 = snd_moment - g1
    g3 = 3.0 * snd_moment + 2.0 * g1
    return (thd_moment - fst_moment * g3) / (g2 ** 1.5)


def kurtosis(root: Node) -> np.ndarray:
    """
    Compute the kurtosis values of a SPN w.r.t. each of the random variables.
    This function returns the kurtosis based on Fisher's definition, i.e.
    3.0 is subtracted from the result to give 0.0 for a normal distribution.

    :param root: The root of the SPN.
    :return: The kurtosis w.r.t. each of the random variables.
    """
    # This implementation is derived from Moors' interpretation
    # (More @ https://en.wikipedia.org/wiki/Kurtosis#Moors'_interpretation)
    # by expanding Var[Z^2] + 1 and obtaining a definition based on non-central moments
    fst_moment = moment(root, order=1)
    snd_moment = moment(root, order=2)
    thd_moment = moment(root, order=3)
    fhd_moment = moment(root, order=4)
    g1 = fst_moment ** 2.0
    g2 = snd_moment - g1
    g3 = 4.0 * (g1 ** 2.0 + fst_moment * thd_moment)
    g4 = snd_moment * (8.0 * g1 - snd_moment)
    return -2.0 + (fhd_moment - g3 + g4) / (g2 ** 2.0)
