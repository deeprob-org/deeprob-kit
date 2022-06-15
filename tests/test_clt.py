import tempfile
import pytest
import numpy as np

from tests.utils import resample_data, complete_binary_data, binary_data_ids, compute_mpe_ids
from tests.utils import random_marginalize_data, complete_posterior_binary_data, complete_marginalized_binary_data
from sklearn.datasets import load_diabetes

from deeprob.spn.utils.validity import is_structured_decomposable
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.io import save_binary_clt_json, load_binary_clt_json
from deeprob.spn.algorithms.inference import log_likelihood


@pytest.fixture
def data():
    data, _, = load_diabetes(return_X_y=True)
    return (data < np.median(data, axis=0)).astype(np.float32)


@pytest.fixture
def evi_data(data):
    return resample_data(data, 1000, np.random.RandomState(42))


@pytest.fixture
def mar_data(evi_data):
    return random_marginalize_data(evi_data, 0.2, np.random.RandomState(42))


@pytest.fixture
def complete_data():
    return complete_binary_data(10)


@pytest.fixture
def complete_mar_data():
    mar_features = [7, 1, 5, 9]
    return complete_marginalized_binary_data(10, mar_features)


@pytest.fixture
def complete_mpe_data():
    mar_features = [7, 1, 5, 9]
    return complete_posterior_binary_data(10, mar_features)


@pytest.fixture
def binary_clt(evi_data):
    scope = list(range(evi_data.shape[1]))
    clt = BinaryCLT(scope, root=1)
    clt.fit(evi_data, [[0, 1]] * evi_data.shape[1], alpha=0.1, random_state=42)
    return clt


def test_complete_inference(binary_clt, complete_data):
    ls = binary_clt.likelihood(complete_data)
    lls = binary_clt.log_likelihood(complete_data)
    assert np.isclose(np.sum(ls).item(), 1.0)
    assert np.isclose(np.sum(np.exp(lls)).item(), 1.0)


def test_mar_inference(binary_clt, evi_data, mar_data):
    evi_ll = binary_clt.log_likelihood(evi_data)
    mar_ll = binary_clt.log_likelihood(mar_data)
    assert np.all(mar_ll >= evi_ll)


def test_mpe_inference(binary_clt, evi_data, mar_data):
    evi_ll = binary_clt.log_likelihood(evi_data)
    mpe_data = binary_clt.mpe(mar_data)
    assert not np.any(np.isnan(mpe_data))
    mpe_ll = binary_clt.log_likelihood(mpe_data)
    assert np.all(mpe_ll >= evi_ll)


def test_mpe_complete_inference(binary_clt, complete_data, complete_mar_data, complete_mpe_data):
    complete_lls = binary_clt.log_likelihood(complete_data)
    mpe_data = binary_clt.mpe(complete_mar_data)
    mpe_ids = binary_data_ids(mpe_data).tolist()
    expected_mpe_ids = compute_mpe_ids(complete_mpe_data, complete_lls.squeeze())
    assert mpe_ids == expected_mpe_ids


def test_ancestral_sampling(binary_clt, evi_data):
    evi_ll = binary_clt.log_likelihood(evi_data).mean()
    samples = np.empty(shape=(1000, 10), dtype=np.float32)
    approx_lls = list()
    for _ in range(200):
        samples[:] = np.nan
        samples = binary_clt.sample(samples)
        approx_lls.extend(binary_clt.log_likelihood(samples).squeeze().tolist())
    approx_ll = np.mean(approx_lls).item()
    assert np.isclose(evi_ll, approx_ll, atol=1e-3)


def test_pc_conversion(binary_clt, evi_data, mar_data):
    spn = binary_clt.to_pc()
    assert is_structured_decomposable(spn) is None

    clt_evi_ll = binary_clt.log_likelihood(evi_data).squeeze(axis=1)
    clt_mar_ll = binary_clt.log_likelihood(mar_data).squeeze(axis=1)
    spn_evi_ll = log_likelihood(spn, evi_data)
    spn_mar_ll = log_likelihood(spn, mar_data)
    assert np.allclose(clt_evi_ll, spn_evi_ll, atol=1e-7)
    assert np.allclose(clt_mar_ll, spn_mar_ll, atol=1e-7)


def test_save_load_json(binary_clt, evi_data):
    ll = binary_clt.log_likelihood(evi_data)
    with tempfile.TemporaryFile('r+') as f:
        save_binary_clt_json(binary_clt, f)
        with pytest.raises(ValueError):
            save_binary_clt_json(BinaryCLT([0, 1, 2], root=0), f)
        f.seek(0)
        loaded_clt = load_binary_clt_json(f)
    loaded_ll = loaded_clt.log_likelihood(evi_data)
    assert np.all(ll == loaded_ll)
