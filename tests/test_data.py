import pytest
import numpy as np

from deeprob.utils.data import mixed_ohe_data, ecdf_data, check_data_dtype
from deeprob.utils.data import DataFlatten, DataNormalizer, DataStandardizer


@pytest.fixture
def data_domains():
    return [(0.0, 4.0), [0, 1, 2], (0.5, 2.0), (0.0, 5.0)]


@pytest.fixture
def data():
    return np.array([
        [3.2, 0, 1.6, 5.0],
        [1.0, 1, 0.5, 0.5],
        [3.0, 0, 0.5, 1.5],
        [3.0, 2, 1.0, 1.0],
        [3.5, 1, 1.0, 4.0],
        [4.0, 0, 0.5, 2.5],
        [0.5, 0, 0.5, 3.0]
    ])


@pytest.fixture
def ohe_data():
    return np.array([
        [3.2, 1, 0, 0, 1.6, 5.0],
        [1.0, 0, 1, 0, 0.5, 0.5],
        [3.0, 1, 0, 0, 0.5, 1.5],
        [3.0, 0, 0, 1, 1.0, 1.0],
        [3.5, 0, 1, 0, 1.0, 4.0],
        [4.0, 1, 0, 0, 0.5, 2.5],
        [0.5, 1, 0, 0, 0.5, 3.0]
    ])


@pytest.fixture
def tensor_data(data):
    return np.stack([data, data], axis=2)


def test_mixed_ohe_data(data_domains, data, ohe_data):
    data = mixed_ohe_data(data, data_domains)
    assert data.tolist() == ohe_data.tolist()


def test_ecdf_data(ohe_data):
    unnorm_ecdf0 = len(ohe_data) * ecdf_data(ohe_data[:, 0])
    unnorm_ecdf1 = len(ohe_data) * ecdf_data(ohe_data[:, 1])
    assert unnorm_ecdf0.tolist() == [5, 2, 4, 4, 6, 7, 1]
    assert unnorm_ecdf1.tolist() == [7, 3, 7, 3, 3, 7, 7]


def test_check_data_dtype():
    uint8_data = np.arange(5).astype(np.uint8)
    uint32_data = np.arange(5).astype(np.uint32)
    float32_data = np.arange(5).astype(np.float32)
    float64_data = np.arange(5).astype(np.float64)
    assert check_data_dtype(uint8_data, np.float32).dtype == np.float32
    assert check_data_dtype(uint32_data, np.uint64).dtype == np.uint64
    assert check_data_dtype(float32_data, np.float32).dtype == np.float32
    assert check_data_dtype(float64_data, np.float32).dtype == np.float64


def test_data_flatten(tensor_data):
    transform = DataFlatten()
    transform.fit(tensor_data)
    data = transform.forward(tensor_data)
    orig_data = transform.backward(data)
    assert data.shape == (7, 8)
    assert np.alltrue(orig_data == tensor_data)


def test_data_normalizer(data):
    transform = DataNormalizer((2.0, 4.0))
    transform.fit(data)
    transformed_data = transform.forward(data)
    orig_data = transform.backward(transformed_data)
    assert (np.min(transformed_data), np.max(transformed_data)) == (2.0, 4.0)
    assert np.allclose(orig_data, data)


def test_data_standardizer_sample_wise(data):
    transform = DataStandardizer(sample_wise=True)
    transform.fit(data)
    transformed_data = transform.forward(data)
    orig_data = transform.backward(transformed_data)
    assert np.allclose(transformed_data.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(transformed_data.std(axis=0), 1.0, atol=1e-7)
    assert np.allclose(orig_data, data, atol=1e-7)


def test_data_standardizer_feature_wise(data):
    transform = DataStandardizer(sample_wise=False)
    transform.fit(data)
    transformed_data = transform.forward(data)
    orig_data = transform.backward(transformed_data)
    assert np.allclose(transformed_data.mean(), 0.0, atol=1e-7)
    assert np.allclose(transformed_data.std(), 1.0, atol=1e-7)
    assert np.allclose(orig_data, data, atol=1e-7)
