# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import abc
from enum import Enum
from typing import Optional, Union, List

import numpy as np
import scipy.stats as ss

from deeprob.utils.data import check_data_dtype
from deeprob.spn.structure.node import Node


class LeafType(Enum):
    """
    The type of the distribution leaf.
    It can be either discrete or continuous.
    """
    DISCRETE = 1
    CONTINUOUS = 2


class Leaf(Node):
    LEAF_TYPE = None

    def __init__(self, scope: Union[int, List[int]]):
        """
        Initialize a leaf node given its scope.

        :param scope: The scope of the leaf.
        :param kwargs: Additional arguments.
        """
        super().__init__([scope] if isinstance(scope, int) else scope)

    @abc.abstractmethod
    def em_init(self, random_state: np.random.RandomState):
        """
        Random initialize the leaf's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """

    @abc.abstractmethod
    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        """
        Compute a batch Expectation-Maximization (EM) step.

        :param stats: The sufficient statistics of each sample.
        :param data: The data regarding random variables of the leaf.
        :param step_size: The step size of update.
        """

    @abc.abstractmethod
    def fit(self, data: np.ndarray, domain: Union[list, tuple], **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param kwargs: Optional parameters.
        :raises ValueError: If a parameter is out of domain.
        """

    @abc.abstractmethod
    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihoods.
        """

    @abc.abstractmethod
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log-likelihoods.
        """

    @abc.abstractmethod
    def mpe(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the maximum at posteriori values.

        :param x: The inputs.
        :return: The distribution's maximum at posteriori values.
        """

    @abc.abstractmethod
    def sample(self, x: np.ndarray) -> np.ndarray:
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """

    @abc.abstractmethod
    def moment(self, k: int = 1) -> float:
        """
        Compute the moment of a given order.

        :param k: The order of the moment.
        :return: The moment of order k.
        """

    @abc.abstractmethod
    def params_count(self) -> int:
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """

    @abc.abstractmethod
    def params_dict(self) -> dict:
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """


class Bernoulli(Leaf):
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope: int, p: float = 0.5):
        """
        Initialize a Bernoulli leaf node given its scope.

        :param scope: The scope of the leaf.
        :param p: The Bernoulli probability.
        :raises ValueError: If a parameter is out of domain.
        """
        super().__init__(scope)
        if p < 0.0 or p > 1.0:
            raise ValueError("The Bernoulli probability must be in [0, 1]")

        self.p = p

    def fit(self, data: np.ndarray, domain: list, alpha: float = 0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: The Laplace smoothing factor.
        :param kwargs: Optional parameters.
        :raises ValueError: If a parameter is out of domain.
        """
        if domain != [0, 1]:
            raise ValueError("The domain must be binary for a Bernoulli distribution")
        if alpha < 0.0:
            raise ValueError("The Laplace smoothing factor must be non-negative")

        # Check the data dtype
        data = check_data_dtype(data, dtype=np.float32)

        # Estimate using Laplace smoothing
        self.p = (np.sum(data) + alpha) / (len(data) + 2 * alpha)

    def em_init(self, random_state: np.random.RandomState):
        self.p = random_state.rand()

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        alpha = np.finfo(np.float16).eps  # Use a very small Laplace smoothing factor
        data = np.squeeze(data, axis=1)
        total_stats = np.sum(stats)
        p = (np.dot(stats, data) + alpha) / (total_stats + 2 * alpha)

        # Update the parameters
        self.p = (1.0 - step_size) * self.p + step_size * p

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ls[~mask] = ss.bernoulli.pmf(x[~mask], self.p)
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.zeros([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        lls[~mask] = ss.bernoulli.logpmf(x[~mask], self.p)
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = 0 if self.p < 0.5 else 1
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = ss.bernoulli.rvs(self.p, size=np.count_nonzero(mask))
        return x

    def moment(self, k: int = 1) -> float:
        return ss.bernoulli.moment(k, self.p)

    def params_count(self):
        return 1

    def params_dict(self):
        return {'p': self.p}


class Categorical(Leaf):
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(
        self,
        scope: int,
        categories: Optional[Union[List, np.ndarray]] = None,
        probabilities: Optional[Union[List, np.ndarray]] = None
    ):
        """
        Initialize a Categorical leaf node given its scope.

        :param scope: The scope of the leaf.
        :param categories: The possible categories.
        :param probabilities: The probabilities associated to each category.
        """
        super().__init__(scope)
        self.categories = None
        self.probabilities = None
        self.distribution = None

        if categories is not None and probabilities is not None:
            if len(categories) != len(probabilities):
                raise ValueError("Each category must be associated a probability")
            if not np.isclose(np.sum(probabilities), 1.0):
                raise ValueError("Probabilities parameter must sum up to 1")
            if isinstance(categories, list):
                categories = np.array(categories, np.int64)
            if isinstance(probabilities, list):
                probabilities = np.array(probabilities, np.float32)
            self.categories = np.array(categories, np.int64)
            self.probabilities = np.array(probabilities, np.float32)
            self.distribution = ss.rv_discrete(values=(self.categories, self.probabilities))
        elif categories is not None or probabilities is not None:
            raise ValueError("Partial defined parameters (categories, probabilities) are not handled")

    def fit(self, data: np.ndarray, domain: list, alpha: float = 0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: The Laplace smoothing factor.
        :param kwargs: Optional parameters.
        :raises ValueError: If a parameter is out of domain.
        """
        if not isinstance(domain, list):
            raise ValueError("The domain must be categorical for a Categorical distribution")
        if alpha < 0.0:
            raise ValueError("The Laplace smoothing factor must be non-negative")

        self.probabilities = np.empty(len(domain), np.float32)
        for i, d in enumerate(domain):
            self.probabilities[i] = (len(data[data == d]) + alpha) / (len(data) + len(domain) * alpha)
        self.categories = np.array(domain, np.int64)
        self.distribution = ss.rv_discrete(values=(self.categories, self.probabilities))

    def em_init(self, random_state: np.random.RandomState):
        """
        Random initialize the leaf's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        :raises ValueError: If the categories are not initialized.
        """
        if self.categories is None:
            raise ValueError("Categorical leaf distribution is not initialized")

        # Initialize the categories probabilities using a dirichlet distribution
        self.probabilities = random_state.dirichlet(np.ones(len(self.categories)))
        self.distribution = ss.rv_discrete(values=(self.categories, self.probabilities))

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        alpha = np.finfo(np.float16).eps  # Use a very small Laplace smoothing factor
        data = np.squeeze(data, axis=1)
        total_stats = np.sum(stats)

        # Compute the probabilities for each category
        probabilities = np.empty(len(self.categories), np.float32)
        for i, d in enumerate(self.categories):
            probabilities[i] = (np.sum(stats[data == d]) + alpha) / (total_stats + len(self.categories) * alpha)

        # Update the parameters
        self.probabilities = (1.0 - step_size) * self.probabilities + step_size * probabilities
        self.distribution = ss.rv_discrete(values=(self.categories, self.probabilities))

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ls[~mask] = self.distribution.pmf(x[~mask].astype(np.int64, copy=False))
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.zeros([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        lls[~mask] = self.distribution.logpmf(x[~mask].astype(np.int64, copy=False))
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.categories[self.probabilities.argmax()]
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.distribution.rvs(size=np.count_nonzero(mask))
        return x

    def moment(self, k: int = 1) -> float:
        return self.distribution.moment(k)

    def params_count(self) -> int:
        return 2 * len(self.categories)

    def params_dict(self) -> dict:
        if self.distribution is None:
            return {'categories': None, 'probabilities': None}
        return {'categories': self.categories, 'probabilities': self.probabilities}


class Isotonic(Leaf):
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(
        self,
        scope: int,
        densities: Optional[Union[List[float], np.ndarray]] = None,
        breaks: Optional[Union[List[float], np.ndarray]] = None
    ):
        """
        Initialize a histogram-Isotonic leaf node given its scope.

        :param scope: The scope of the leaf.
        :param densities: The densities. They must sum up to one.
        :param breaks: The breaks values, such that len(breaks) == len(densities) + 1.
        :raises ValueError: If a parameter is out of domain.
        """
        super().__init__(scope)
        self.densities = None
        self.breaks = None
        self.distribution = None

        if densities is not None and breaks is not None:
            if len(breaks) != len(densities) + 1:
                raise ValueError("Invalid histogram parameters shapes")
            if not np.isclose(np.sum(densities), 1.0):
                raise ValueError("Densities parameter must sum up to 1")
            if isinstance(densities, list):
                densities = np.array(densities, np.float32)
            if isinstance(breaks, list):
                breaks = np.array(breaks, np.float32)
            self.densities = densities
            self.breaks = breaks
            self.distribution = ss.rv_histogram(histogram=(densities, breaks))
        elif densities is not None or breaks is not None:
            raise ValueError("Partial defined parameters (densities, breaks) are not handled")

    def fit(self, data: np.ndarray, domain: tuple, alpha: float = 0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: The Laplace smoothing factor.
        :param kwargs: Optional parameters.
        :raises ValueError: If a parameter is out of domain.
        """
        if not isinstance(domain, tuple):
            raise ValueError("The domain must be continuous for an Isotonic distribution")
        if alpha < 0.0:
            raise ValueError("The Laplace smoothing factor must be non-negative")
        histogram, breaks = np.histogram(data, bins='fd')

        # Apply Laplace smoothing and obtain the densities
        densities = (histogram + alpha) / (len(data) + len(histogram) * alpha)
        densities = densities.astype(np.float32, copy=False)
        breaks = breaks.astype(np.float32, copy=False)

        # Build the distribution
        self.densities = densities
        self.breaks = breaks
        self.distribution = ss.rv_histogram(histogram=(densities, breaks))

    def em_init(self, random_state: np.random.RandomState):
        raise NotImplementedError("EM parameters initialization not yet implemented for Isotonic distributions")

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        raise NotImplementedError("EM step not yet implemented for Isotonic distributions")

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ood_mask = ~mask & ((x <= self.distribution.a) | (x >= self.distribution.b))
        ls[~mask] = self.distribution.pdf(x[~mask])
        ls[ood_mask] = np.finfo(np.float32).eps
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.zeros([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ood_mask = ~mask & ((x <= self.distribution.a) | (x >= self.distribution.b))
        lls[~mask] = self.distribution.logpdf(x[~mask])
        lls[ood_mask] = np.log(np.finfo(np.float64).eps)
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        idx = np.argmax(self.densities)
        x[mask] = (self.breaks[idx] + self.breaks[idx + 1]) / 2.0
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.distribution.ppf(q=np.random.rand(np.count_nonzero(mask)))
        return x

    def moment(self, k: int = 1) -> np.ndarray:
        return self.distribution.moment(k)

    def params_count(self) -> int:
        return 2 * len(self.densities) + 1

    def params_dict(self) -> dict:
        if self.distribution is None:
            return {'densities': None, 'breaks': None}
        return {'densities': self.densities, 'breaks': self.breaks}


class Uniform(Leaf):
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope: int, start: float = 0.0, width: float = 1.0):
        """
        Initialize an Uniform leaf node given its scope.

        :param scope: The scope of the leaf.
        :param start: The start of the uniform distribution.
        :param width: The width of the uniform distribution.
        """
        super().__init__(scope)
        self.start = start
        self.width = width

    def fit(self, data: np.ndarray, domain: tuple, **kwargs):
        if not isinstance(domain, tuple):
            raise ValueError("The domain must be continuous for an Uniform distribution")

        # Estimate the parameters of a uniform distribution
        self.start, self.width = ss.uniform.fit(data)

    def em_init(self, random_state: np.random.RandomState):
        raise NotImplementedError("EM parameters initialization not yet implemented for Uniform distributions")

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        raise NotImplementedError("EM step not yet implemented for Uniform distributions")

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ls[~mask] = ss.uniform.pdf(x[~mask], self.start, self.width)
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.zeros([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        lls[~mask] = ss.uniform.logpdf(x[~mask], self.start, self.width)
        return lls

    def mpe(self, x: np.ndarray) -> np.array:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.start
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = ss.uniform.rvs(self.start, self.width, size=np.count_nonzero(mask))
        return x

    def moment(self, k: int = 1) -> float:
        return ss.uniform.moment(k, self.start, self.width)

    def params_count(self) -> int:
        return 2

    def params_dict(self) -> dict:
        return {
            'start': self.start,
            'width': self.width
        }


class Gaussian(Leaf):
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope: int, mean: float = 0.0, stddev: float = 1.0):
        """
        Initialize a Gaussian leaf node given its scope.

        :param scope: The scope of the leaf.
        :param mean: The mean parameter.
        :param stddev: The standard deviation parameter.
        :raises ValueError: If a parameter is out of domain.
        """
        super().__init__(scope)
        if stddev <= 1e-5:
            raise ValueError("The standard deviation of a Gaussian must be greater than 1e-5")

        self.mean = mean
        self.stddev = stddev

    def fit(self, data: np.ndarray, domain: tuple, **kwargs):
        if not isinstance(domain, tuple):
            raise ValueError("The domain must be continuous for a Gaussian distribution")

        self.mean, self.stddev = ss.norm.fit(data)
        self.stddev = max(self.stddev, 1e-5)

    def em_init(self, random_state: np.random.RandomState):
        self.mean = 1e-1 * random_state.randn()
        self.stddev = 0.5 + 1e-1 * np.tanh(random_state.randn())

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        data = np.squeeze(data, axis=1)
        total_stats = np.sum(stats) + np.finfo(np.float32).eps
        mean = np.sum(stats * data) / total_stats
        stddev = np.sqrt(np.sum(stats * (data - mean) ** 2.0) / total_stats)
        stddev = max(stddev, 1e-5)

        # Update the parameters
        self.mean = (1.0 - step_size) * self.mean + step_size * mean
        self.stddev = (1.0 - step_size) * self.stddev + step_size * stddev

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        ls = np.ones([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        ls[~mask] = ss.norm.pdf(x[~mask], self.mean, self.stddev)
        return ls

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        lls = np.zeros([len(x), 1], dtype=np.float32)
        mask = np.isnan(x)
        lls[~mask] = ss.norm.logpdf(x[~mask], self.mean, self.stddev)
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = self.mean
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mask = np.isnan(x)
        x[mask] = ss.norm.rvs(self.mean, self.stddev, size=np.count_nonzero(mask))
        return x

    def moment(self, k: int = 1) -> float:
        return ss.norm.moment(k, self.mean, self.stddev)

    def params_count(self) -> int:
        return 2

    def params_dict(self) -> dict:
        return {
            'mean': self.mean,
            'stddev': self.stddev
        }
