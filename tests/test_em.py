import pytest
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.datasets import load_diabetes
from tests.utils import resample_data

from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.learning.wrappers import learn_estimator
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.learning.em import expectation_maximization


@pytest.fixture
def data():
    data, _, = load_diabetes(return_X_y=True)
    return (data < np.median(data, axis=0)).astype(np.float32)


@pytest.fixture
def evi_data(data):
    return resample_data(data, 1000, np.random.RandomState(42))


@pytest.fixture
def blobs_data():
    blobs_data, _ = make_blobs(
        n_samples=1000, n_features=2, random_state=1337,
        centers=[[-1.0, 1.0], [1.0, -1.0]], cluster_std=0.25
    )
    return blobs_data


@pytest.fixture
def gaussian_spn():
    g0a, g1a = Gaussian(0, -0.5, 0.5), Gaussian(1, 0.5, 0.5)
    g0b, g1b = Gaussian(0, 0.5, 0.5), Gaussian(1, -0.5, 0.5)
    p0 = Product(children=[g0a, g1a])
    p1 = Product(children=[g0b, g1b])
    p2 = Product(children=[g0a, g1b])
    s0 = Sum(children=[p0, p1, p2], weights=[0.3, 0.5, 0.2])
    s0.id, p0.id, p1.id, p2.id = 0, 1, 2, 3
    g0a.id, g1a.id, g0b.id, g1b.id = 4, 5, 6, 7
    return s0


@pytest.fixture
def spn_mle(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * 10, [[0, 1]] * 10,
        learn_leaf='mle', split_rows='gmm', split_cols='gvs', min_rows_slice=512,
        random_state=42, verbose=False
    )


@pytest.fixture
def spn_clt(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * 10, [[0, 1]] * 10,
        learn_leaf='binary-clt', split_rows='kmeans', split_cols='gvs', min_rows_slice=512,
        learn_leaf_kwargs={'to_pc': False},
        random_state=42, verbose=False
    )


def test_spn_binary(spn_mle, evi_data):
    expectation_maximization(
        spn_mle, evi_data, num_iter=100, batch_perc=0.1, step_size=0.5,
        random_init=False, random_state=42, verbose=False
    )
    ll = log_likelihood(spn_mle, evi_data).mean()
    assert np.isclose(ll, -5.3, atol=5e-2)


def test_clt_binary(spn_clt, evi_data):
    expectation_maximization(
        spn_clt, evi_data, num_iter=100, batch_perc=0.1, step_size=0.5,
        random_init=True, random_state=42, verbose=False
    )
    ll = log_likelihood(spn_clt, evi_data).mean()
    assert np.isclose(ll, -5.1, atol=5e-2)


def test_spn_gaussian(gaussian_spn, blobs_data):
    expectation_maximization(
        gaussian_spn, blobs_data, num_iter=25, batch_perc=0.1, step_size=0.5,
        random_init=True, random_state=42, verbose=False
    )
    ll = log_likelihood(gaussian_spn, blobs_data).mean()
    assert np.isclose(ll, -0.7, atol=5e-2)
