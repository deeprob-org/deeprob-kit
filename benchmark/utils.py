import numpy as np

from typing import Tuple, List
from experiments.datasets import load_binary_dataset


def compute_mean_stddev_times(elapsed_times: List[float]) -> Tuple[float, float]:
    mean_time = np.mean(elapsed_times).item()
    stddev_time = 2.0 * np.std(elapsed_times).item() / np.sqrt(len(elapsed_times))
    return mean_time, stddev_time


def load_dataset(
    name: str,
    mar_p: float = 0.5,
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0.0 < mar_p < 1.0
    random_state = np.random.RandomState(seed)
    data_train, _, data_test = load_binary_dataset('experiments/datasets', name, raw=True)
    indices = random_state.choice(len(data_test), size=n_samples, replace=True)  # Resample the test set
    evi_data = data_test[indices]
    data_train, evi_data = data_train.astype(np.float32), evi_data.astype(np.float32)
    mar_data = evi_data.copy()
    mar_data[random_state.rand(*mar_data.shape) <= mar_p] = np.nan
    return data_train, evi_data, mar_data
