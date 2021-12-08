# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import abc
from typing import Optional, Union, Tuple, List, Type

import numpy as np
from scipy import stats

from deeprob.context import is_check_dtype_enabled


class DataTransform(abc.ABC):
    """Abstract data transformation."""
    @abc.abstractmethod
    def fit(self, data: np.ndarray):
        """
        Fit the data transform with some data.

        :param data: The data for fitting.
        """

    @abc.abstractmethod
    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the data transform to some data.

        :param data: The data to transform.
        :return: The transformed data.
        """

    @abc.abstractmethod
    def backward(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the backward data transform to some data.

        :param data: The data to transform.
        :return: The transformed data.
        """


class DataFlatten(DataTransform):
    def __init__(self):
        """
        Build the data flatten transformation.
        """
        self.shape = None

    def fit(self, data: np.ndarray):
        self.shape = data.shape[1:]

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.reshape(data, [len(data), -1])

    def backward(self, data: np.ndarray) -> np.ndarray:
        return np.reshape(data, [len(data), *self.shape])


class DataNormalizer(DataTransform):
    def __init__(
        self,
        interval: Optional[Tuple[float, float]] = None,
        clip: bool = False,
        dtype=np.float32
    ):
        """
        Build the data normalizer transformation.

        :param interval: The normalizing interval. If None data will be normalized in [0, 1].
        :param clip: Whether to clip data if out of interval.
        :param dtype: The type for type conversion.
        :raises ValueError: If the normalizing interval is out of domain.
        """
        if interval is None:
            interval = (0.0, 1.0)
        elif interval[0] >= interval[1]:
            raise ValueError("The normalizing interval must be (a, b) with a < b")

        self.interval = interval
        self.clip = clip
        self.dtype = dtype
        self.prev_dtype = None
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        self.prev_dtype = data.dtype
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def forward(self, data: np.ndarray) -> np.ndarray:
        a, b = self.interval
        data = (data - self.min) / (self.max - self.min)
        data = data * (b - a) + a
        if self.clip:
            data = np.clip(data, a, b)
        return data.astype(self.dtype)

    def backward(self, data: np.ndarray) -> np.ndarray:
        a, b = self.interval
        data = (data - a) / (b - a)
        data = (self.max - self.min) * data + self.min
        return data.astype(self.prev_dtype)


class DataStandardizer(DataTransform):
    def __init__(self, sample_wise: bool = True, eps: float = 1e-7, dtype=np.float32):
        """
        Build the data standardizer transformation.

        :param sample_wise: Whether to apply sample wise standardization.
        :param eps: The epsilon value for standardization.
        :param dtype: The type for type conversion.
        :raises ValueError: If the epsilon value is out of domain.
        """
        if eps <= 0.0:
            raise ValueError("The epsilon value must be positive")
        self.sample_wise = sample_wise
        self.eps = eps
        self.dtype = dtype
        self.prev_dtype = None
        self.mean = None
        self.stddev = None

    def fit(self, data: np.ndarray):
        self.prev_dtype = data.dtype
        axis = 0 if self.sample_wise else None
        self.mean = np.mean(data, axis=axis)
        self.stddev = np.std(data, axis=axis)

    def forward(self, data: np.ndarray) -> np.ndarray:
        data = (data - self.mean) / (self.stddev + self.eps)
        return data.astype(self.dtype)

    def backward(self, data: np.ndarray) -> np.ndarray:
        data = (self.stddev + self.eps) * data + self.mean
        return data.astype(self.prev_dtype)


def ohe_data(data: np.ndarray, domain: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    One-Hot-Encoding function.

    :param data: The 1D data to encode.
    :param domain: The domain to use.
    :return: The One Hot encoded data.
    """
    ohe = np.zeros((len(data), len(domain)), dtype=np.float32)
    ohe[np.equal.outer(data, domain)] = 1.0
    return ohe


def mixed_ohe_data(data: np.ndarray, domains: List[Union[list, tuple]]) -> np.ndarray:
    """
    One-Hot-Encoding function, applied on mixed data (both continuous and non-binary discrete).
    Note that One-Hot-Encoding is applied only on categorical random variables having more than two values.

    :param data: The data matrix to encode.
    :param domains: The domains to use.
    :return: The One Hot encoded data.
    :raises ValueError: If there are inconsistencies between the data and domains.
    """
    _, n_features = data.shape
    if len(domains) != n_features:
        raise ValueError("Each data column should correspond to a random variable having a domain")

    ohe = []
    for i in range(n_features):
        if len(domains[i]) > 2:
            ohe.append(ohe_data(data[:, i], domains[i]))
        else:
            ohe.append(data[:, i])
    return np.column_stack(ohe)


def ecdf_data(data: np.ndarray) -> np.ndarray:
    """
    Empirical Cumulative Distribution Function (ECDF).

    :param data: The data.
    :return: The result of the ECDF on data.
    """
    return stats.rankdata(data, method='max') / len(data)


def check_data_dtype(data: np.ndarray, dtype: Type[np.dtype] = np.float32):
    """
    Check whether the data is compatible with a given dtype (defaults to np.float32).
    If the data dtype is not compatible, then cast it.

    :param data: The data.
    :param dtype: The desidered dtype compatibility (defaults to np.float32).
    :return: The casted data if necessary, otherwise returns data itself.
    """
    if not is_check_dtype_enabled():
        # Skip data dtype check and casting
        return data

    # Get flags for floating point data and type
    is_data_fp = data.dtype in [np.float32, np.float64]
    is_dtype_fp = dtype in [np.float32, np.float64]

    if is_dtype_fp:
        if not is_data_fp or data.dtype.itemsize < np.dtype(dtype).itemsize:
            # If dtype is FP and data is not FP or it is a "smaller" FP, then cast it
            return data.astype(dtype)
    elif is_data_fp or data.dtype.itemsize < np.dtype(dtype).itemsize:
        # If dtype is integral and data is FP or it is a "smaller" integral, then cast it
        return data.astype(dtype)

    # Data is compatible w.r.t. dtype
    # i.e. it is FP if dtype is FP and integral if dtype is integral, and it is at least as "big" as dtype
    return data
