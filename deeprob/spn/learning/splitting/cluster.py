# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

import warnings
from typing import Union, Type, List

import numpy as np
from sklearn import mixture, cluster
from sklearn.exceptions import ConvergenceWarning

from deeprob.spn.structure.leaf import Leaf, LeafType
from deeprob.utils.data import mixed_ohe_data


def gmm(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2
) -> np.ndarray:
    """
    Execute GMM clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any(len(d) > 2 for d in domains):
        data = mixed_ohe_data(data, domains)

    # Apply GMM
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings
        return mixture.GaussianMixture(n, n_init=3, random_state=random_state).fit_predict(data)


def kmeans(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2
) -> np.ndarray:
    """
    Execute K-Means clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any(len(d) > 2 for d in domains):
        data = mixed_ohe_data(data, domains)

    # Apply K-Means
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings
        return cluster.KMeans(n, n_init=5, random_state=random_state).fit_predict(data)


def kmeans_mb(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2
) -> np.ndarray:
    """
    Execute MiniBatch K-Means clustering on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    """
    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any(len(d) > 2 for d in domains):
        data = mixed_ohe_data(data, domains)

    # Apply K-Means MiniBatch
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings
        warnings.simplefilter(action='ignore', category=UserWarning)  # Ignore user warnings
        return cluster.MiniBatchKMeans(n, n_init=5, random_state=random_state).fit_predict(data)


def dbscan(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2
) -> np.ndarray:
    """
    Execute DBSCAN clustering on some data (only on discrete data).

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    :raises ValueError: If the leaf distributions are NOT discrete.
    """
    # Control if distribution are binary
    if not all(x.LEAF_TYPE == LeafType.DISCRETE for x in distributions):
        raise ValueError('DBScan clustering can be applied only on discrete attributes')

    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any(len(d) > 2 for d in domains):
        data = mixed_ohe_data(data, domains)

    # Apply DBSCAN
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings
        return cluster.DBSCAN(eps = 0.25, n_jobs=-1).fit_predict(data)


def wald(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2
) -> np.ndarray:
    """
    Execute Ward (Hierarchical) clustering on some data (only discrete data).

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters.
    :return: An array where each element is the cluster where the corresponding data belong.
    :raises ValueError: If the leaf distributions are NOT discrete.
    """
    # Control if distribution are binary
    if not all(x.LEAF_TYPE == LeafType.DISCRETE for x in distributions):
        raise ValueError('DBScan clustering can be applied only on discrete attributes')

    # Convert the data using One Hot Encoding, in case of non-binary discrete features
    if any(len(d) > 2 for d in domains):
        data = mixed_ohe_data(data, domains)

    # Apply Wald
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings
        return cluster.AgglomerativeClustering(n, linkage='ward').fit_predict(data)
