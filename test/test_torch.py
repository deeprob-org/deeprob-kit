import os
import unittest
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


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTorch, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_grad_enabled(False)
        cls.data = torch.rand([100, 5, 5])
        cls.targets = (torch.rand(100) < 0.3).long()
        cls.flattened_shape = torch.Size([np.prod(cls.data.shape).item()])
        cls.reshaped_shape = torch.Size([cls.data.shape[0], np.prod(cls.data.shape[1:]).item()])

    def test_get_torch_classes(self):
        cls = get_activation_class('relu')
        self.assertEqual(cls, nn.ReLU)
        self.assertRaises(ValueError, get_activation_class, name='unknown-activation')
        cls = get_optimizer_class('sgd')
        self.assertEqual(cls, optim.SGD)
        self.assertRaises(ValueError, get_optimizer_class, name='unknown-optimizer')

    def test_initializers(self):
        t1 = torch.empty(8, 4, 3)
        t2 = torch.empty(8, 4, 3)
        dirichlet_(t1, log_space=False)
        dirichlet_(t2, log_space=False, dim=1)
        t1 = torch.sum(t1, dim=-1)
        t2 = torch.sum(t2, dim=1)
        self.assertTrue(torch.allclose(t1, torch.ones_like(t1)))
        self.assertTrue(torch.allclose(t2, torch.ones_like(t2)))
        self.assertRaises(ValueError, dirichlet_, torch.tensor(0))
        self.assertRaises(IndexError, dirichlet_, t1, dim=2)

    def test_constraints(self):
        module = nn.Module()
        module.scale = torch.tensor([[0.5, -1.0], [-0.5, 1.0], [0.5, 0.0]])
        clipped_scale = torch.tensor([[0.5, 1e-5], [1e-5, 1.0], [0.5, 1e-5]])
        clipper = ScaleClipper(eps=1e-5)
        clipper(module)
        self.assertTrue(torch.allclose(module.scale, clipped_scale))

    def test_callbacks(self):
        checkpoint_filepath = 'checkpoint.pt'
        module = nn.Linear(256, 64)
        early_stopping = EarlyStopping(module, patience=2, filepath=checkpoint_filepath)
        self.assertFalse(early_stopping.should_stop)
        early_stopping(1.0, epoch=0)
        module.weight *= 2.0
        early_stopping(0.9, epoch=1)
        module.weight *= 2.0
        early_stopping(1.1, epoch=2)
        module.weight *= 2.0
        early_stopping(1.2, epoch=3)
        self.assertTrue(early_stopping.should_stop)
        self.assertFalse(torch.equal(module.state_dict()['weight'], early_stopping.get_best_state()['weight']))
        self.assertRaises(ValueError, EarlyStopping, module, patience=0)
        self.assertRaises(ValueError, EarlyStopping, module, delta=0.0)
        os.remove(checkpoint_filepath)

    def test_metrics(self):
        ram = RunningAverageMetric()
        ram(1.0, num_samples=1)
        ram(2.5, num_samples=2)
        self.assertEqual(ram.average(), 2.0)
        ram.reset()
        self.assertRaises(ZeroDivisionError, ram.average)
        self.assertRaises(ValueError, ram.__call__, metric=1.0, num_samples=0)

    def test_transforms_normalize(self):
        normalize = Normalize(self.data.mean(), self.data.std())
        data_normalized = normalize.forward(self.data)
        self.assertTrue(torch.allclose(data_normalized.mean(), torch.zeros(1), atol=1e-7))
        self.assertTrue(torch.allclose(data_normalized.std(), torch.ones(1), atol=1e-7))
        self.assertTrue(torch.allclose(normalize.backward(data_normalized), self.data, atol=1e-7))

    def test_transforms_quantize(self):
        quantize = Quantize(n_bits=5)
        data_quantized = quantize(self.data)
        data_dequantized = quantize.backward(data_quantized)
        self.assertTrue(torch.all(data_quantized >= 0.0) and torch.all(data_quantized <= 1.0))
        self.assertTrue(torch.all(data_dequantized >= 0.0) and torch.all(data_dequantized <= 1.0))
        self.assertTrue(torch.allclose(data_quantized, quantize(data_dequantized), atol=1e-7))

    def test_transforms_flatten(self):
        flatten = Flatten(shape=self.data.shape)
        flattened_data = flatten(self.data)
        self.assertEqual(flattened_data.shape, self.flattened_shape)
        self.assertTrue(torch.all(flatten.backward(flattened_data) == self.data))

    def test_transforms_reshape(self):
        reshape = Reshape(target_shape=self.reshaped_shape, shape=self.data.shape)
        reshaped_data = reshape(self.data)
        self.assertEqual(reshaped_data.shape, self.reshaped_shape)
        self.assertTrue(torch.all(reshape.backward(reshaped_data) == self.data))

    def test_scaled_tanh(self):
        data = torch.randn(100, 1)
        scaled_tanh = ScaledTanh()
        self.assertTrue(torch.allclose(scaled_tanh(data), torch.zeros(1), atol=1e-07))
        scaled_tanh.weight.data.fill_(2.0)
        self.assertTrue(torch.allclose(scaled_tanh(data), 2.0 * torch.tanh(data), atol=1e-7))

    def test_masked_linear(self):
        data = torch.randn(100, 2)
        masked_linear = MaskedLinear(2, 1, np.asarray([[0.2, 0.5]]))
        masked_linear.weight.data.fill_(2.0)
        masked_linear.bias.data.fill_(3.14)
        self.assertTrue(torch.allclose(
            masked_linear(data),
            3.14 + torch.sum(2.0 * data * torch.tensor([[0.2, 0.5]]), dim=1, keepdim=True),
            atol=1e-7
        ))

    def test_datasets(self):
        unsupervised_dataset = UnsupervisedDataset(self.data)
        self.assertEqual(len(unsupervised_dataset), len(self.data))
        self.assertEqual(unsupervised_dataset.features_shape, self.data.shape[1:])
        self.assertEqual(unsupervised_dataset[0].shape, self.data.shape[1:])
        supervised_dataset = SupervisedDataset(self.data, self.targets)
        self.assertEqual(len(supervised_dataset), len(self.data))
        self.assertEqual(supervised_dataset.features_shape, self.data.shape[1:])
        self.assertEqual(supervised_dataset.num_classes, len(torch.unique(self.targets)))
        self.assertEqual(supervised_dataset[0][0].shape, self.data.shape[1:])
        self.assertEqual(supervised_dataset[0][1].dtype, torch.long)
        dataset = WrappedDataset(supervised_dataset, unsupervised=True)
        self.assertEqual(len(dataset), len(self.data))
        self.assertEqual(dataset.features_shape, self.data.shape[1:])
        self.assertEqual(dataset[0].shape, self.data.shape[1:])
        dataset = WrappedDataset(supervised_dataset, classes=[0, 1], unsupervised=False)
        self.assertEqual(len(dataset), len(self.data))
        self.assertEqual(dataset.features_shape, self.data.shape[1:])
        self.assertEqual(dataset.num_classes, len(torch.unique(self.targets)))
        self.assertEqual(dataset[0][0].shape, self.data.shape[1:])
        self.assertEqual(dataset[0][1].dtype, torch.long)


if __name__ == '__main__':
    unittest.main()
