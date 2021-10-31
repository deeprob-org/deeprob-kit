from typing import Union, Type, List

import numpy as np

from deeprob.spn.structure.leaf import Leaf, LeafType
from deeprob.utils.statistics import compute_gini


def gini_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Compute Gini's index based splitting.

    :param data: The data.
    :param distributions: Distributions of the features.
    :param domains: Range of values of the features.
    :param e: Threshold of the considered entropy to be signficant.
    :param alpha: laplacian alpha to apply at frequence.
    :return: A partitioning of features.
    """
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)

    # Compute entropy for each variable
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:  # discrete
            gini = compute_gini(data[:, i], domains[i], 'discrete', alpha)
        elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:  # continuous
            gini = compute_gini(data[:, i], domains[i], 'continuous', alpha)
        else:
            raise ValueError('Leaves distributions must be either discrete or continuous')

        # Add to cluster if entropy less than threshold
        if gini < e:
            partition[i] = 1

    return partition


def gini_adaptive_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1,
    size: int = None
) -> np.ndarray:
    """
    Compute Adaptive Gini's index based splitting.

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
        raise ValueError("Missing size input for entropy adaptive computation")

    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)

    # compute entropy for each variable
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:  # discrete
            gini = compute_gini(data[:, i], domains[i], 'discrete', alpha)
        elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:  # continuous
            gini = compute_gini(data[:, i], domains[i], 'continuous', alpha)
        else:
            raise ValueError('Leaves distributions must be either discrete or continuous')

        # Adaptive gini
        e = max(e * (data.shape[0] / size), 1e-07)

        # add to cluster if gini is less than threshold
        if gini < e:
            partition[i] = 1

    return partition
