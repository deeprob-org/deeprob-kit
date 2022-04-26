import pytest
import numpy as np

from deeprob.utils.statistics import estimate_priors_joints, compute_mutual_information
from deeprob.utils.statistics import compute_mean_quantiles, compute_gini, compute_bpp, compute_fid


@pytest.fixture
def data():
    return np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 0],
        [1, 0]
    ], dtype=np.float32)


@pytest.fixture
def priors():
    return np.array([[0.4, 0.6], [0.8, 0.2]])


@pytest.fixture
def joints():
    return np.array([
        [
            [[0.4, 0.0], [0.0, 0.6]],  # (0, 0, 0:2, 0:2)
            [[0.3, 0.1], [0.5, 0.1]]   # (0, 1, 0:2, 0:2)
        ],
        [
            [[0.3, 0.5], [0.1, 0.1]],  # (1, 0, 0:2, 0:2)
            [[0.8, 0.0], [0.0, 0.2]]   # (1, 1, 0:2, 0:2)
        ]
    ])


@pytest.fixture
def mi():
    return 0.3 * np.log(0.3 / (0.4 * 0.8)) \
        + 0.1 * np.log(0.1 / (0.4 * 0.2)) \
        + 0.5 * np.log(0.5 / (0.6 * 0.8)) \
        + 0.1 * np.log(0.1 / (0.6 * 0.2))


def test_estimate_priors_joints(data, priors, joints):
    priors, joints = estimate_priors_joints(data, alpha=0.0)
    assert priors.dtype == data.dtype
    assert joints.dtype == data.dtype
    assert np.allclose(priors, priors)
    assert np.allclose(joints, joints)


def test_estimate_mutual_information(priors, joints, mi):
    computed_mi = compute_mutual_information(priors, joints)
    assert np.allclose(np.diag(computed_mi), 0.0)
    assert np.all(computed_mi == computed_mi.T)
    assert np.allclose(computed_mi[0, 1], mi)


def test_compute_mean_quantiles(data):
    mean_quantiles = compute_mean_quantiles(data, 2)
    assert np.allclose(mean_quantiles, [[0.2, 0.0], [1.0, 0.4]])
    with pytest.raises(ValueError):
        compute_mean_quantiles(data, 0)
        compute_mean_quantiles(data, len(data) + 1)


def test_compute_gini(priors):
    g = 1.0 - (priors[0, 0] ** 2.0 + priors[0, 1] ** 2.0)
    assert compute_gini(priors[0]) == g
    with pytest.raises(ValueError):
        compute_gini(priors[:, 0])


def test_compute_bpp():
    assert compute_bpp(100.0, 10) == (-100.0 / np.log(2.0) / 10)


def test_compute_fid():
    dim = 512
    m1, c1, m2, c2 = np.zeros(dim), np.eye(dim), np.ones(dim), np.eye(dim)
    assert compute_fid(m1, c1, m2, c2) == dim
    with pytest.raises(ValueError):
        compute_fid(m1[1:], m2, c1, c2)
        compute_fid(m1, c1[1:, 1:], m2, c2)
        compute_fid(np.stack([m1, m1]), c1, m2, c2)
        compute_fid(m1, c1, m2, np.stack([c2, c2]))
