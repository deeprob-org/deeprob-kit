import pytest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deeprob.torch.utils import ScaledTanh, MaskedLinear
from deeprob.torch.utils import get_activation_class, get_optimizer_class
from deeprob.torch.constraints import ScaleClipper
from deeprob.torch.initializers import dirichlet_
from deeprob.torch.callbacks import EarlyStopping
from deeprob.torch.metrics import RunningAverageMetric
from deeprob.torch.transforms import Normalize, Quantize, Flatten, Reshape
from deeprob.torch.datasets import SupervisedDataset, UnsupervisedDataset, WrappedDataset


def test_get_torch_classes():
    cls = get_activation_class('relu')
    assert cls == nn.ReLU
    cls = get_optimizer_class('sgd')
    assert cls == optim.SGD
    with pytest.raises(ValueError):
        get_activation_class('unknown-activation')
    with pytest.raises(ValueError):
        get_optimizer_class('unknown-optimizer')


def test_initializers():
    t1 = torch.empty(8, 4, 3)
    t2 = torch.empty(8, 4, 3)
    dirichlet_(t1, log_space=False)
    dirichlet_(t2, log_space=False, dim=1)
    t1 = torch.sum(t1, dim=-1)
    t2 = torch.sum(t2, dim=1)
    assert torch.allclose(t1, torch.ones_like(t1))
    assert torch.allclose(t2, torch.ones_like(t2))
    with pytest.raises(ValueError):
        dirichlet_(torch.tensor(0))
    with pytest.raises(IndexError):
        dirichlet_(t1, dim=2)


def test_constraints():
    module = nn.Module()
    module.scale = torch.tensor([[0.5, -1.0], [-0.5, 1.0], [0.5, 0.0]])
    clipped_scale = torch.tensor([[0.5, 1e-5], [1e-5, 1.0], [0.5, 1e-5]])
    clipper = ScaleClipper(eps=1e-5)
    clipper(module)
    assert torch.allclose(module.scale, clipped_scale)


def test_callbacks():
    checkpoint_filepath = tempfile.NamedTemporaryFile().name
    module = nn.Linear(256, 64)
    early_stopping = EarlyStopping(module, patience=2, filepath=checkpoint_filepath)
    assert not early_stopping.should_stop
    early_stopping(1.0, epoch=0)
    module.weight *= 2.0
    early_stopping(0.9, epoch=1)
    module.weight *= 2.0
    early_stopping(1.1, epoch=2)
    module.weight *= 2.0
    early_stopping(1.2, epoch=3)
    assert isinstance("{}".format(early_stopping), str)
    assert early_stopping.should_stop
    assert not torch.equal(module.state_dict()['weight'], early_stopping.get_best_state()['weight'])
    with pytest.raises(ValueError):
        EarlyStopping(module, patience=0)
    with pytest.raises(ValueError):
        EarlyStopping(module, delta=0.0)


def test_metrics():
    ram = RunningAverageMetric()
    ram(1.0, num_samples=1)
    ram(2.5, num_samples=2)
    assert np.isclose(ram.average(), 2.0)
    ram.reset()
    with pytest.raises(ZeroDivisionError):
        ram.average()
    with pytest.raises(ValueError):
        ram(metric=1.0, num_samples=0)


def test_transforms_normalize():
    data = 2.0 * torch.randn(50, 10) + 1.0
    normalize = Normalize(data.mean(), data.std())
    data_normalized = normalize.forward(data)
    assert torch.allclose(data_normalized.mean(), torch.zeros(1), atol=1e-7)
    assert torch.allclose(data_normalized.std(), torch.ones(1), atol=1e-7)
    assert torch.allclose(normalize.backward(data_normalized), data)


def test_transforms_quantize():
    data = torch.rand(50, 10)
    quantize = Quantize(n_bits=5)
    data_quantized = quantize(data)
    data_dequantized = quantize.backward(data_quantized)
    assert torch.all(data_quantized >= 0.0) and torch.all(data_quantized <= 1.0)
    assert torch.all(data_dequantized >= 0.0) and torch.all(data_dequantized <= 1.0)
    assert torch.allclose(data_quantized, quantize(data_dequantized))


def test_transforms_flatten():
    data = 2.0 * torch.randn(50, 10) + 1.0
    flatten = Flatten(shape=data.shape)
    flattened_data = flatten(data)
    assert flattened_data.shape == (np.prod(data.shape),)
    assert torch.all(flatten.backward(flattened_data) == data)


def test_transforms_reshape():
    data = 2.0 * torch.randn(50, 10) + 1.0
    reshape = Reshape(target_shape=(10, 50), shape=data.shape)
    reshaped_data = reshape(data)
    assert reshaped_data.shape == (10, 50)
    assert torch.all(reshape.backward(reshaped_data) == data)


def test_scaled_tanh():
    data = torch.randn(100, 1)
    scaled_tanh = ScaledTanh()
    assert torch.allclose(scaled_tanh(data), torch.zeros(1))
    scaled_tanh.weight.data.fill_(2.0)
    assert torch.allclose(scaled_tanh(data), 2.0 * torch.tanh(data))


def test_masked_linear():
    data = torch.randn(100, 2)
    masked_linear = MaskedLinear(2, 1, np.asarray([[0.2, 0.5]]))
    masked_linear.weight.data.fill_(2.0)
    masked_linear.bias.data.fill_(3.14)
    assert torch.allclose(
        masked_linear(data),
        3.14 + torch.sum(2.0 * data * torch.tensor([[0.2, 0.5]]), dim=1, keepdim=True)
     )
    with pytest.raises(ValueError):
        MaskedLinear(2, 1, np.asarray([[0.2]]))


def test_datasets():
    data = 2.0 * torch.randn(50, 8, 8) + 1.0
    targets = (torch.rand(50) <= 0.5).long()
    unsupervised_dataset = UnsupervisedDataset(data)
    assert len(unsupervised_dataset) == len(data)
    assert unsupervised_dataset.features_shape == data.shape[1:]
    assert unsupervised_dataset[0].shape == data.shape[1:]

    supervised_dataset = SupervisedDataset(data, targets)
    assert len(supervised_dataset) == len(data)
    assert supervised_dataset.features_shape == data.shape[1:]
    assert supervised_dataset.num_classes == len(torch.unique(targets))
    assert supervised_dataset[0][0].shape == data.shape[1:]
    assert supervised_dataset[0][1].dtype == torch.long

    dataset = WrappedDataset(supervised_dataset, unsupervised=True)
    assert len(dataset) == len(data)
    assert dataset.features_shape == data.shape[1:]
    assert dataset[0].shape == data.shape[1:]
    dataset = WrappedDataset(supervised_dataset, classes=[0, 1], unsupervised=False)

    assert len(dataset) == len(data)
    assert dataset.features_shape == data.shape[1:]
    assert dataset.num_classes == len(torch.unique(targets))
    assert dataset[0][0].shape == data.shape[1:]
    assert dataset[0][1].dtype == torch.long
