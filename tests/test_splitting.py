import pytest
import numpy as np

from deeprob.spn.learning.splitting.rdc import rdc_scores
from deeprob.spn.structure import Uniform


@pytest.fixture
def distributions():
    return [Uniform, Uniform]


@pytest.fixture
def domains():
    return [[-5.0, 5.0], [-5.0, 5.0]]


@pytest.fixture
def linear_data():
    x = 2.0 * np.random.rand(1000) - 1.0
    y = np.pi * x + 1.0
    return np.stack([x, y], axis=1)


@pytest.fixture
def nonlinear_data():
    x = 6.0 * np.random.rand(1000) - 3.0
    y = np.exp(np.cos(x) - 0.5) + 1.0
    return np.stack([x, y], axis=1)


@pytest.fixture
def uncorrelated_data():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    return np.stack([x, y], axis=1)


def test_rdc_scores(linear_data, nonlinear_data, uncorrelated_data, distributions, domains):
    random_state = np.random.RandomState(42)
    rdc_matrix = rdc_scores(linear_data, distributions, domains, random_state)
    assert np.allclose(np.diagonal(rdc_matrix), 1.0)
    assert np.allclose(np.diagonal(np.fliplr(rdc_matrix)), 1.0, atol=1e-3)

    rdc_matrix = rdc_scores(nonlinear_data, distributions, domains, random_state)
    assert np.allclose(np.diagonal(rdc_matrix), 1.0)
    assert np.allclose(np.diagonal(np.fliplr(rdc_matrix)), 1.0, atol=1e-3)

    rdc_matrix = rdc_scores(uncorrelated_data, distributions, domains, random_state)
    assert np.allclose(np.diagonal(rdc_matrix), 1.0)
    assert np.all(np.diagonal(np.fliplr(rdc_matrix)) < 0.3)
