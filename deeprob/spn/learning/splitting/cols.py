import numpy as np

from typing import Union, Type, Tuple, List, Callable, Any

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.learning.splitting.gvs import gvs_cols
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
    elif split_cols == 'rdc':
        return rdc_cols
    elif split_cols == 'random':
        return random_cols
    else:
        raise ValueError("Unknown split rows method called {}".format(split_cols))
