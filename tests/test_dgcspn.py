import pytest
import numpy as np
import torch

from deeprob.spn.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer
from deeprob.spn.models import DgcSpn


@pytest.fixture
def data():
    return torch.randn([8, 3, 32, 32])


@pytest.fixture
def mar_data(data):
    mar_data = data.clone()
    mar_data[torch.rand_like(mar_data) < 0.5] = np.nan
    return mar_data


@pytest.fixture
def ones_data(data):
    return torch.ones_like(data)


def test_gaussian_layer():
    gaussian = SpatialGaussianLayer(
        (3, 32, 32), out_channels=16,
        optimize_scale=False, uniform_loc=(-1.0, 1.0)
    )
    assert gaussian.out_features == (16, 32, 32)
    assert torch.all(gaussian.loc >= -1.0) and torch.all(gaussian.loc <= 1.0)
    assert torch.all(gaussian.scale == 1.0)
    assert not gaussian.scale.requires_grad

    gaussian = SpatialGaussianLayer(
        (3, 32, 32), out_channels=16,
        optimize_scale=True
    )
    assert torch.all(gaussian.scale > 0.0)
    assert gaussian.scale.requires_grad
    with pytest.raises(ValueError):
        SpatialGaussianLayer((3, 32, 32), 16, quantiles_loc=np.zeros([16, 3, 32, 32]), uniform_loc=(-1.0, 1.0))


def test_depthwise_product_layer(ones_data):
    product = SpatialProductLayer(
        (3, 32, 32), kernel_size=2, padding='full',
        stride=1, dilation=4, depthwise=True
    )
    assert product.pad == [4, 4, 4, 4]
    assert torch.all(product.weight == torch.ones(3, 1, 2, 2))
    assert product.out_features == (3, 32 + 4, 32 + 4)
    assert torch.allclose(product(ones_data)[:, :, 4:-4, 4:-4], torch.tensor(4.0))

    product = SpatialProductLayer(
        (3, 32, 32), kernel_size=2, padding='valid',
        stride=2, dilation=1, depthwise=True
    )
    assert product.pad == [0, 0, 0, 0]
    assert torch.all(product.weight == torch.ones(3, 1, 2, 2))
    assert product.out_features == (3, 16, 16)
    assert torch.allclose(product(ones_data), torch.tensor(4.0))


def test_non_depthwise_product_layer(ones_data):
    product = SpatialProductLayer(
        (3, 32, 32), kernel_size=2, padding='full',
        stride=1, dilation=8, depthwise=False
    )
    assert product.pad == [8, 8, 8, 8]
    assert tuple(product.weight.shape) == (3 ** 4, 3, 2, 2)
    assert torch.allclose(torch.sum(product.weight, dim=1), torch.tensor(1.0))
    assert product.out_features, (3 ** 4, 32 + 8, 32 + 8)
    assert torch.allclose(product(ones_data)[:, :, 8:-8, 8:-8], torch.tensor(4.0))


def test_sum_layer():
    s = SpatialSumLayer((3, 32, 32), out_channels=8)
    assert s.out_features == (8, 32, 32)


def test_root_layer():
    root = SpatialRootLayer((3, 32, 32), out_channels=8)
    assert root.out_channels == 8
    assert tuple(root.weight.shape) == (8, np.prod((3, 32, 32)))


@pytest.mark.parametrize("n_pooling,depthwise", [(0, False), (2, False), (0, True), (2, True)])
def test_dgcspn_mpe_log_prob(data, mar_data, n_pooling, depthwise):
    dgcspn = DgcSpn(data.shape[1:], n_batch=4, sum_channels=4, n_pooling=n_pooling, depthwise=depthwise)
    lls = dgcspn.log_prob(data)
    with torch.enable_grad():
        mpe_data = dgcspn.mpe(mar_data)
    mpe_lls = dgcspn.log_prob(mpe_data)
    assert torch.all(mpe_lls.squeeze() > lls.squeeze())
