import warnings
import numpy as np

from typing import Union, Type, List

from sklearn import mixture, cluster
from sklearn.exceptions import ConvergenceWarning
from deeprob.spn.structure.leaf import Leaf
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
    if any([len(d) > 2 for d in domains]):
        data = mixed_ohe_data(data, domains)

    # Apply GMM
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for GMM
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
    if any([len(d) > 2 for d in domains]):
        data = mixed_ohe_data(data, domains)

    # Apply K-Means
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n, n_init=5, random_state=random_state).fit_predict(data)
