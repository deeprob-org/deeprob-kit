# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import warnings
from typing import Union, Type, List, Callable
from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn import cluster, cross_decomposition
from sklearn.exceptions import ConvergenceWarning

from deeprob.spn.structure.leaf import LeafType, Leaf
from deeprob.utils.data import ohe_data, ecdf_data


def rdc_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    d: float = 0.3,
    k: int = 20,
    s: float = 1.0 / 6.0,
    nl: Callable[[np.ndarray], np.ndarray] = np.sin
) -> np.ndarray:
    """
    Split the features using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param d: The threshold value that regulates the independence tests among the features.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non linear function to use.
    :return: A features partitioning.
    """
    # Compute the RDC scores matrix
    rdc_matrix = rdc_scores(data, distributions, domains, random_state, k=k, s=s, nl=nl)

    # Compute the adjacency matrix
    adj_matrix = (rdc_matrix > d).astype(np.int32)

    # Compute the connected components of the adjacency matrix
    adj_matrix = sparse.csr_matrix(adj_matrix)
    _, clusters = sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True)
    return clusters


def rdc_rows(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    n: int = 2,
    k: int = 20,
    s: float = 1.0 / 6.0,
    nl: Callable[[np.ndarray], np.ndarray] = np.sin
) -> np.ndarray:
    """
    Split the samples using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters for KMeans.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non linear function to use.
    :return: A samples partitioning.
    """
    # Transform the samples by RDC
    rdc_samples = np.concatenate(
        rdc_transform(data, distributions, domains, random_state, k, s, nl), axis=1
    )

    # Apply K-Means to the transformed samples
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n, n_init=5, random_state=random_state).fit_predict(rdc_samples)


def rdc_scores(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    k: int = 20,
    s: float = 1.0 / 6.0,
    nl: Callable[[np.ndarray], np.ndarray] = np.sin
) -> np.ndarray:
    """
    Compute the RDC (Randomized Dependency Coefficient) score for each pair of features.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non linear function to use.
    :return: The RDC score matrix.
    """
    # Apply RDC transformation to the features
    _, n_features = data.shape
    rdc_features = rdc_transform(data, distributions, domains, random_state, k, s, nl)
    pairwise_comparisons = list(combinations(range(n_features), 2))

    # Run Canonical Component Analysis (CCA) on RDC-transformed features
    rdc_matrix = np.empty(shape=(n_features, n_features), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for CCA
        for i, j in pairwise_comparisons:
            score = rdc_cca(i, j, rdc_features)
            rdc_matrix[i, j] = rdc_matrix[j, i] = score
    np.fill_diagonal(rdc_matrix, 1.0)
    return rdc_matrix


def rdc_cca(i: int, j: int, features: List[np.ndarray]) -> float:
    """
    Compute the RDC (Randomized Dependency Coefficient) using CCA (Canonical Correlation Analysis).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    cca = cross_decomposition.CCA(n_components=1, max_iter=128, tol=1e-3)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    x_cca, y_cca = x_cca.squeeze(), y_cca.squeeze()
    return np.corrcoef(x_cca, y_cca)[0, 1]


def rdc_transform(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    k: int = 20,
    s: float = 1.0 / 6.0,
    nl: Callable[[np.ndarray], np.ndarray] = np.sin
) -> List[np.ndarray]:
    """
    Execute the RDC (Randomized Dependency Coefficient) pipeline on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non-linear function to use.
    :return: The transformed data.
    :raises ValueError: If an unknown distribution type is found.
    """
    features = []
    for i, dist in enumerate(distributions):
        if dist.LEAF_TYPE == LeafType.DISCRETE:
            feature_matrix = ohe_data(data[:, i], domains[i])
        elif dist.LEAF_TYPE == LeafType.CONTINUOUS:
            feature_matrix = np.expand_dims(data[:, i], axis=-1)
        else:
            raise ValueError("Unknown distribution type {}".format(dist.LEAF_TYPE))
        x = np.apply_along_axis(ecdf_data, 0, feature_matrix)
        features.append(x.astype(np.float32))

    samples = []
    for x in features:
        stddev = np.sqrt(s / x.shape[1])
        w = stddev * random_state.randn(x.shape[1], k).astype(np.float32)
        b = stddev * random_state.randn(k).astype(np.float32)
        y = nl(np.dot(x, w) + b)
        samples.append(y)
    return samples
