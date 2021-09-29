import numpy as np

from typing import List
from itertools import product


def binary_samples_ids(samples: np.ndarray) -> np.ndarray:
    n_features = samples.shape[-1]
    return np.dot(samples, 1 << np.arange(n_features - 1, -1, -1)).astype(np.int64)


def compute_mpe_ids(data: np.ndarray, mpe_data: np.ndarray, lls: np.ndarray) -> List[int]:
    sample_ids = binary_samples_ids(data)
    mpe_sample_ids = binary_samples_ids(mpe_data)
    lls = dict(zip(sample_ids, lls))
    expected_mpe_ids = list(map(
        # Take the ids of the completions having maximum LL
        lambda x: max(map(lambda i: (i, lls[i]), x), key=lambda y: y[1])[0],
        mpe_sample_ids
    ))
    return expected_mpe_ids


def build_resampled_data(data: np.ndarray, n_samples: int, random_state: np.random.RandomState) -> np.ndarray:
    return data[random_state.choice(len(data), size=n_samples, replace=True)]


def build_mar_data(data: np.ndarray, mar_features: List[int]) -> np.ndarray:
    data = data.copy()
    data[:, mar_features] = np.nan
    return data


def build_random_mar_data(data: np.ndarray, p: float, random_state: np.random.RandomState) -> np.ndarray:
    data = data.copy()
    data[random_state.rand(*data.shape) <= p] = np.nan
    return data


def build_complete_data(n_features: int) -> np.ndarray:
    return np.array([list(i) for i in product([0, 1], repeat=n_features)], dtype=np.float32)


def build_complete_mar_data(n_features: int, mar_features: List[int]) -> np.ndarray:
    data = build_complete_data(n_features)
    data[:, mar_features] = np.nan
    return data


def build_complete_mpe_data(n_features: int, mar_features: List[int]) -> np.ndarray:
    complete_mar_features = build_complete_data(len(mar_features))
    data = np.expand_dims(build_complete_data(n_features), axis=1)
    data = np.repeat(data, repeats=len(complete_mar_features), axis=1)
    data[:, :, mar_features] = complete_mar_features
    return data
