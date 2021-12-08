# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, List

import numpy as np

from deeprob.spn.structure.leaf import Leaf, LeafType


def entropy_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Entropy based column splitting method.

    :param data: The data.
    :param distributions: Distributions of the features.
    :param domains: Range of values of the features.
    :param e: Threshold of the considered entropy to be signficant.
    :param alpha: laplacian alpha to apply at frequence.
    :return: A partitioning of features.
    """
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=np.int64)

    # Compute entropy for each variable
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:
            bins = domains[i] + [len(domains[i])]
            hist, _ = np.histogram(data[:, i], bins=bins)
            probs = (hist + alpha) / (len(data) + len(hist) * alpha)
            entropy = -np.sum(probs * np.log2(probs))
        elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:
            hist, _ = np.histogram(data[:, i], bins='scott')
            probs = (hist + alpha) / (len(data) + len(hist) * alpha)
            entropy = -np.sum(probs * np.log2(probs)) / np.log2(len(hist))
        else:
            raise ValueError("Leaves distributions must be either discrete or continuous")

        # Add to cluster if entropy is less than the threshold
        if entropy < e:
            partition[i] = 1

    return partition


def entropy_adaptive_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1,
    size: int = None
) -> np.ndarray:
    """
    Adaptive Entropy based column splitting method.

    :param data: The data.
    :param distributions: Distributions of the features.
    :param domains: Range of values of the features.
    :param e: Threshold of the considered entropy to be signficant.
    :param alpha: laplacian alpha to apply at frequence.
    :param size: Size of whole dataset.
    :return: A partitioning of features.
    :raises ValueError: If the size of the data is missing.
    """
    if size is None:
        raise ValueError("Missing data size for Adaptive Entropy column splitting method")

    return entropy_cols(
        data, distributions, domains, random_state,
        e=max(e * (len(data) / size), np.finfo(np.float32).eps),
        alpha=alpha
    )
