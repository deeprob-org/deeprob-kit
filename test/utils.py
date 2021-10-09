import numpy as np

from typing import List
from itertools import product


def binary_data_ids(data: np.ndarray) -> np.ndarray:
    """
    Given some binary data, compute the id of each sample.
    This is done by converting the binary features into a decimal number.

    :param data: The data.
    :return: The id of each data sample.
    """
    return np.dot(data, 1 << np.arange(data.shape[-1] - 1, -1, -1)).astype(np.int64)


def compute_mpe_ids(complete_mpe_data: np.ndarray, complete_lls: np.ndarray) -> List[int]:
    """
    Compute the maximum at posterior samples ids.

    :param complete_mpe_data: The complete posterior data.
    :param complete_lls: The log-likelihoods of each data sample.
    :return: A list of maximum at posterior sample ids.
    """
    sample_ids = binary_data_ids(complete_mpe_data)
    sample_lls = complete_lls[sample_ids]
    mpe_idx = np.argmax(sample_lls, axis=1)
    mpe_sample_ids = sample_ids[np.arange(len(sample_ids)), mpe_idx]
    return mpe_sample_ids.tolist()


def resample_data(data: np.ndarray, n_samples: int, random_state: np.random.RandomState) -> np.ndarray:
    """
    Resample data with replacement.

    :param data: The original data.
    :param n_samples: The number of samples to extract.
    :param random_state: The random state.
    :return: The resampled data.
    """
    return data[random_state.choice(len(data), size=n_samples, replace=True)]


def marginalize_data(data: np.ndarray, mar_features: List[int]) -> np.ndarray:
    """
    Marginalize data, given a list of features to marginalize.

    :param data: The original data.
    :param mar_features: A list of features to marginalize.
    :return: The marginalized features (using NaNs).
    """
    data = data.astype(np.float32, copy=True)
    data[:, mar_features] = np.nan
    return data


def random_marginalize_data(data: np.ndarray, p: float, random_state: np.random.RandomState) -> np.ndarray:
    """
    Marginalize data sample-wise randomly.

    :param data: The original data.
    :param p: The probability of marginalize a feature value of a single sample.
    :param random_state: The random state.
    :return: The marginalized data (using NaNs).
    """
    data = data.astype(np.float32, copy=True)
    data[random_state.rand(*data.shape) <= p] = np.nan
    return data


def complete_binary_data(n_features: int) -> np.ndarray:
    """
    Build a data set with complete assignments of binary features.

    :param n_features: The number of features.
    :return: A data array with shape (2 ** n_features, n_features).
    """
    return np.array([list(i) for i in product([0, 1], repeat=n_features)], dtype=np.float32)


def complete_marginalized_binary_data(n_features: int, mar_features: List[int]) -> np.ndarray:
    """
    Build a data set with complete assignments of binary features, with marginalized features.

    :param n_features: The number of features.
    :param mar_features: A list of features to marginalize.
    :return: A marginalized data array with shape (2 ** n_features, n_features).
    """
    evi_features = [i for i in range(n_features) if i not in mar_features]
    data = np.empty(shape=(2 ** len(evi_features), n_features), dtype=np.float32)
    data[:, evi_features] = complete_binary_data(len(evi_features))
    data[:, mar_features] = np.nan
    return data


def complete_posterior_binary_data(n_features: int, mar_features: List[int]) -> np.ndarray:
    """
    Build a data set with complete assignments of binary features, having another dimension
    for all the possible assignments of marginalized features.

    :param n_features: The number of features.
    :param mar_features: A list of features for which consider all the possible assignment combinations.
    :return: A data array with shape (2 ** (n_features - n_mar_features), 2 ** n_mar_features, n_features).
    """
    evi_features = [i for i in range(n_features) if i not in mar_features]
    data = np.empty(shape=(2 ** len(evi_features), 2 ** len(mar_features), n_features), dtype=np.float32)
    data[:, :, evi_features] = np.expand_dims(complete_binary_data(len(evi_features)), axis=1)
    data[:, :, mar_features] = complete_binary_data(len(mar_features))
    return data
