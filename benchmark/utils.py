from typing import Tuple, List

import numpy as np

from experiments.datasets import load_binary_dataset


def compute_mean_stddev_times(elapsed_times: List[float]) -> Tuple[float, float]:
    mean_time = np.mean(elapsed_times).item()
    stddev_time = 2.0 * np.std(elapsed_times).item() / np.sqrt(len(elapsed_times))
    return mean_time, stddev_time


def load_benchmark_data(
    name: str,
    mar_p: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0.0 < mar_p < 1.0
    data_train, _, data_test = load_binary_dataset('experiments/datasets', name, raw=True)
    data_train, data_test = data_train.astype(np.float32), data_test.astype(np.float32)
    mar_data = data_test.copy()
    random_state = np.random.RandomState(seed)
    mar_data[random_state.rand(*mar_data.shape) <= mar_p] = np.nan
    data_test = data_test.astype(np.int64)
    return data_train, data_test, mar_data
