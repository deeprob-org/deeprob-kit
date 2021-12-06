# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, Tuple, List, Callable, Any

import numpy as np

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.learning.splitting.cluster import gmm, kmeans, kmeans_mb, dbscan, wald
from deeprob.spn.learning.splitting.rdc import rdc_rows
from deeprob.spn.learning.splitting.random import random_rows

#: A signature for a rows splitting function.
SplitRowsFunc = Callable[
    [np.ndarray,                # The data
     List[Type[Leaf]],          # The distributions
     List[Union[list, tuple]],  # The domains
     np.random.RandomState,     # The random state
     Any],                      # Other arguments
    np.ndarray                  # The rows ids
]


def split_rows_clusters(
    data: np.ndarray,
    clusters: np.ndarray
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Split the data horizontally given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :return: (slices, weights) where slices is a list of partial data and
             weights is a list of proportions of the local data in respect to the original data.
    """
    slices = list()
    weights = list()
    n_samples = len(data)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        local_data = data[clusters == c, :]
        slices.append(local_data)
        weights.append(len(local_data) / n_samples)
    return slices, weights


def get_split_rows_method(split_rows: str) -> SplitRowsFunc:
    """
    Get the rows splitting method given a string.

    :param split_rows: The string of the method do get.
    :return: The corresponding rows splitting function.
    :raises ValueError: If the rows splitting method is unknown.
    """
    if split_rows == 'kmeans':
        return kmeans
    if split_rows == 'kmeans_mb':
        return kmeans_mb
    if split_rows == 'dbscan':
        return dbscan
    if split_rows == 'wald':
        return wald
    if split_rows == 'gmm':
        return gmm
    if split_rows == 'rdc':
        return rdc_rows
    if split_rows == 'random':
        return random_rows
    raise ValueError("Unknown split rows method called {}".format(split_rows))
