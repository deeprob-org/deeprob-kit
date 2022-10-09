import pytest
import numpy as np
import torch

from deeprob.flows.utils import squeeze_depth2d, unsqueeze_depth2d
from deeprob.flows.utils import DequantizeLayer, LogitLayer
from deeprob.flows.models.realnvp import RealNVP1d, RealNVP2d
from deeprob.flows.models.maf import MAF


@pytest.fixture
def data():
    return torch.rand([32, 3, 8, 8])


@pytest.fixture
def flattened_data(data):
    return torch.flatten(data, start_dim=1)


def assert_flow_inverse(flow, data):
    target_data, ildj = flow.apply_backward(data)
    orig_data, ldj = flow.apply_forward(target_data)
    assert torch.allclose(ildj, -ldj, atol=5e-7)
    assert torch.allclose(orig_data, data, atol=5e-7)


def assert_sampling_autograd(flow):
    with torch.enable_grad():
        samples = flow.rsample(64)
        assert samples.requires_grad
        samples.mean().backward()


def test_squeeze_depth2d(data):
    squeezed_data = squeeze_depth2d(data)
    unsqueezed_data = unsqueeze_depth2d(squeezed_data)
    assert squeezed_data.shape[1:] == torch.Size([12, 4, 4])
    assert unsqueezed_data.shape[1:] == torch.Size([3, 8, 8])


def test_dequantize(data):
    dequantize = DequantizeLayer(data.shape[1:], n_bits=5)
    quantized_data, ldj = dequantize.apply_forward(data)
    dequantized_data, ildj = dequantize.apply_backward(quantized_data)
    requantized_data, _ = dequantize.apply_forward(dequantized_data)
    assert torch.all(quantized_data >= 0.0) and torch.all(quantized_data <= 1.0)
    assert torch.all(dequantized_data >= 0.0) and torch.all(dequantized_data <= 1.0)
    assert torch.allclose(ildj, -ldj)
    assert torch.allclose(quantized_data, requantized_data)


def test_logit(data):
    logit = LogitLayer(data.shape[1:], alpha=0.01)
    delogit_data, ildj = logit.apply_backward(data)
    logit_data, ldj = logit.apply_forward(delogit_data)
    assert torch.all(logit_data >= 0.0) and torch.all(logit_data <= 1.0)
    assert torch.all(delogit_data >= np.log(logit.alpha / (1.0 - logit.alpha)))
    assert torch.all(delogit_data <= np.log((1.0 - logit.alpha) / logit.alpha))
    assert torch.allclose(ildj, -ldj)
    assert torch.allclose(logit_data, data)


def test_realnvp1d(flattened_data):
    realnvp = RealNVP1d(flattened_data.shape[1:], batch_norm=True, affine=True).eval()
    assert_flow_inverse(realnvp, flattened_data)
    realnvp = RealNVP1d(flattened_data.shape[1:], batch_norm=False, affine=True).eval()
    assert_flow_inverse(realnvp, flattened_data)
    realnvp = RealNVP1d(flattened_data.shape[1:], batch_norm=True, affine=False).eval()
    assert_flow_inverse(realnvp, flattened_data)
    realnvp = RealNVP1d(
        flattened_data.shape[1:], batch_norm=True, affine=True,
        dequantize=True, logit=0.01
    ).eval()
    assert_flow_inverse(realnvp, flattened_data)
    assert_sampling_autograd(realnvp)
    with pytest.raises(ValueError):
        RealNVP1d(10, n_flows=0)
    with pytest.raises(ValueError):
        RealNVP1d(10, depth=0)
    with pytest.raises(ValueError):
        RealNVP1d(10, units=0)


def test_realnvp2d(data):
    realnvp = RealNVP2d(data.shape[1:], n_flows=2, n_blocks=2, channels=8, network='resnet', affine=True).eval()
    assert_flow_inverse(realnvp, data)
    realnvp = RealNVP2d(data.shape[1:], n_flows=2, n_blocks=2, channels=8, network='resnet', affine=False).eval()
    assert_flow_inverse(realnvp, data)
    realnvp = RealNVP2d(data.shape[1:], n_flows=2, n_blocks=2, channels=8, network='densenet', affine=True).eval()
    assert_flow_inverse(realnvp, data)
    realnvp = RealNVP2d(data.shape[1:], n_flows=2, n_blocks=2, channels=8, network='densenet', affine=False).eval()
    assert_flow_inverse(realnvp, data)
    realnvp = RealNVP2d(
        data.shape[1:], n_flows=2, n_blocks=2, channels=8, network='resnet', affine=True,
        dequantize=True, logit=0.01
    ).eval()
    assert_flow_inverse(realnvp, data)
    assert_sampling_autograd(realnvp)
    with pytest.raises(ValueError):
        RealNVP2d((3, 8, 8), n_flows=0)
    with pytest.raises(ValueError):
        RealNVP2d((3, 8, 8), n_blocks=0)
    with pytest.raises(ValueError):
        RealNVP2d((3, 8, 8), channels=0)
    with pytest.raises(NotImplementedError):
        RealNVP2d((3, 8, 8), network='unknown')


def test_maf(flattened_data):
    maf = MAF(flattened_data.shape[1:], batch_norm=True).eval()
    assert_flow_inverse(maf, flattened_data)
    maf = MAF(flattened_data.shape[1:], batch_norm=False).eval()
    assert_flow_inverse(maf, flattened_data)
    maf = MAF(flattened_data.shape[1:], batch_norm=True, units=8, sequential=False, random_state=42).eval()
    assert_flow_inverse(maf, flattened_data)
    maf = MAF(flattened_data.shape[1:], batch_norm=False, units=8, sequential=False, random_state=42).eval()
    assert_flow_inverse(maf, flattened_data)
    maf = MAF(
        flattened_data.shape[1:], batch_norm=True,
        dequantize=True, logit=0.01
    ).eval()
    assert_flow_inverse(maf, flattened_data)
    assert_sampling_autograd(maf)
    with pytest.raises(ValueError):
        MAF(10, n_flows=0)
    with pytest.raises(ValueError):
        MAF(10, depth=0)
    with pytest.raises(ValueError):
        MAF(10, units=0)

