import pytest
import numpy as np

from deeprob.spn.structure.leaf import Categorical, Isotonic, Uniform


@pytest.fixture
def domain():
    return list(range(6))


@pytest.fixture
def categorical_data(domain):
    return np.random.binomial(len(domain) - 2, p=0.3, size=[1000, 1])


@pytest.fixture
def continuous_data(domain):
    return 2.0 * np.random.randn(100, 1)


def test_categorical(domain, categorical_data):
    leaf = Categorical(0)
    leaf.fit(categorical_data, domain=domain, alpha=1e-5)
    p1 = len(categorical_data[categorical_data == 1]) / len(categorical_data)
    assert np.isclose(leaf.likelihood(np.ones([1, 1])).item(), p1)
    assert leaf.mpe(np.full([1, 1], np.nan)).item() == leaf.probabilities.argmax()
    leaf = Categorical(0, **leaf.params_dict())
    assert leaf.mpe(np.full([1, 1], np.nan)).item() == leaf.probabilities.argmax()
    assert leaf.log_likelihood(np.full([1, 1], domain[-1])).item() > np.log(1e-16)
    assert leaf.params_count() == 2 * len(domain)


def test_isotonic(domain, continuous_data):
    leaf = Isotonic(0)
    leaf.fit(continuous_data, domain=(-8.0, 8.0), alpha=1e-5)
    densities, breaks = np.histogram(continuous_data, bins='fd', density=True)
    idx = np.searchsorted(breaks, 0.8, side='right')
    assert np.isclose(leaf.likelihood(np.full([1, 1], 0.8)).item(), densities[idx - 1])
    assert -1.0 < leaf.mpe(np.full([1, 1], np.nan)).item() < 1.0
    leaf = Isotonic(0, **leaf.params_dict())
    assert -1.0 < leaf.mpe(np.full([1, 1], np.nan)).item() < 1.0
    assert leaf.log_likelihood(np.full([1, 1], 10.0)).item() > np.log(1e-16)
    assert leaf.params_count() == 2 * len(densities) + 1


def test_uniform(domain, continuous_data):
    leaf = Uniform(0)
    leaf.fit(continuous_data, domain=(-9.0, 9.0))
    ls = leaf.likelihood(np.array([[-1.5], [0.0], [2.0]])).flatten()
    assert set(ls) == set(leaf.likelihood(np.full([1, 1], 0.3)).flatten())
    leaf = Uniform(0, **leaf.params_dict())
    assert leaf.mpe(np.full([1, 1], np.nan)).item() == np.min(continuous_data)
    assert leaf.params_count() == 2
