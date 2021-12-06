# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, Tuple, List, Callable, Any

import numpy as np

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.learning.splitting.gvs import gvs_cols, rgvs_cols, wrgvs_cols
from deeprob.spn.learning.splitting.entropy import entropy_cols, entropy_adaptive_cols
from deeprob.spn.learning.splitting.gini import gini_cols, gini_adaptive_cols
from deeprob.spn.learning.splitting.rdc import rdc_cols
from deeprob.spn.learning.splitting.random import random_cols

#: A signature for a columns splitting function.
SplitColsFunc = Callable[
    [np.ndarray,                # The data
     List[Type[Leaf]],          # The distributions
     List[Union[list, tuple]],  # The domains
     np.random.RandomState,     # The random state
     Any],                      # Other arguments
    np.ndarray                  # The columns ids
]


def split_cols_clusters(
    data: np.ndarray,
    clusters: np.ndarray,
    scope: List[int]
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Split the data vertically given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :param scope: The original scope.
    :return: (slices, scopes) where slices is a list of partial data and
             scopes is a list of partial scopes.
    """
    slices = list()
    scopes = list()
    scope = np.asarray(scope)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        cols = (clusters == c)
        slices.append(data[:, cols])
        scopes.append(scope[cols].tolist())
    return slices, scopes


def get_split_cols_method(split_cols: str) -> SplitColsFunc:
    """
    Get the columns splitting method given a string.

    :param split_cols: The string of the method do get.
    :return: The corresponding columns splitting function.
    :raises ValueError: If the columns splitting method is unknown.
    """
    if split_cols == 'gvs':
        return gvs_cols
    if split_cols == 'rgvs':
        return rgvs_cols
    if split_cols == 'wrgvs':
        return wrgvs_cols
    if split_cols == 'ebvs':
        return entropy_cols
    if split_cols == 'ebvs_ae':
        return entropy_adaptive_cols
    if split_cols == 'gbvs':
        return gini_cols
    if split_cols == 'gbvs_ag':
        return gini_adaptive_cols
    if split_cols == 'rdc':
        return rdc_cols
    if split_cols == 'random':
        return random_cols
    raise ValueError("Unknown split rows method called {}".format(split_cols))
