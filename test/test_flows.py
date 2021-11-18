import unittest
import torch
import numpy as np

from deeprob.flows.utils import squeeze_depth2d, unsqueeze_depth2d
from deeprob.flows.utils import DequantizeLayer, LogitLayer
from deeprob.flows.models.realnvp import RealNVP1d, RealNVP2d
from deeprob.flows.models.maf import MAF


class TestFlows(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFlows, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_grad_enabled(False)
        cls.data = torch.rand([32, 3, 8, 8])
        cls.data_shape = cls.data.shape[1:]
        cls.flattened_data = torch.flatten(cls.data, start_dim=1)
        cls.flattened_data_shape = cls.flattened_data.shape[1]

    def __assert_flow1d_inverse(self, flow):
        target_data, ildj = flow.apply_backward(self.flattened_data)
        data, ldj = flow.apply_forward(target_data)
        self.assertTrue(torch.allclose(ildj, -ldj, atol=1e-6))
        self.assertTrue(torch.allclose(data, self.flattened_data, atol=1e-6))

    def __assert_flow2d_inverse(self, flow):
        target_data, ildj = flow.apply_backward(self.data)
        data, ldj = flow.apply_forward(target_data)
        self.assertTrue(torch.allclose(ildj, -ldj, atol=1e-6))
        self.assertTrue(torch.allclose(data, self.data, atol=1e-6))

    def test_squeeze_depth2d(self):
        squeezed_data = squeeze_depth2d(self.data)
        unsqueezed_data = unsqueeze_depth2d(squeezed_data)
        self.assertEqual(
            squeezed_data.shape[1:],
            torch.Size([self.data_shape[0] * 4, self.data_shape[1] // 2, self.data_shape[2] // 2])
        )
        self.assertEqual(unsqueezed_data.shape[1:], self.data_shape)

    def test_dequantize(self):
        dequantize = DequantizeLayer(self.data_shape, n_bits=5)
        quantized_data, ldj = dequantize.apply_forward(self.data)
        dequantized_data, ildj = dequantize.apply_backward(quantized_data)
        requantized_data, _ = dequantize.apply_forward(dequantized_data)
        self.assertTrue(torch.all(quantized_data >= 0.0) and torch.all(quantized_data <= 1.0))
        self.assertTrue(torch.all(dequantized_data >= 0.0) and torch.all(dequantized_data <= 1.0))
        self.assertTrue(torch.allclose(ildj, -ldj, atol=1e-6))
        self.assertTrue(torch.allclose(quantized_data, requantized_data, atol=1e-6))

    def test_logit(self):
        logit = LogitLayer(self.data_shape, alpha=0.01)
        delogit_data, ildj = logit.apply_backward(self.data)
        logit_data, ldj = logit.apply_forward(delogit_data)
        self.assertTrue(torch.all(logit_data >= 0.0) and torch.all(logit_data <= 1.0))
        self.assertTrue(
            torch.all(delogit_data >= np.log(logit.alpha / (1.0 - logit.alpha))) and
            torch.all(delogit_data <= np.log((1.0 - logit.alpha) / logit.alpha))
        )
        self.assertTrue(torch.allclose(ildj, -ldj, atol=1e-6))
        self.assertTrue(torch.allclose(logit_data, self.data, atol=1e-6))

    def test_realnvp1d(self):
        realnvp = RealNVP1d(self.flattened_data_shape, batch_norm=True, affine=True).eval()
        self.__assert_flow1d_inverse(realnvp)
        realnvp = RealNVP1d(self.flattened_data_shape, batch_norm=False, affine=True).eval()
        self.__assert_flow1d_inverse(realnvp)
        realnvp = RealNVP1d(self.flattened_data_shape, batch_norm=True, affine=False).eval()
        self.__assert_flow1d_inverse(realnvp)
        realnvp = RealNVP1d(
            self.flattened_data_shape, batch_norm=True, affine=True,
            dequantize=True, logit=0.01
        ).eval()
        self.__assert_flow1d_inverse(realnvp)

    def test_realnvp2d(self):
        realnvp = RealNVP2d(self.data_shape, n_flows=2, n_blocks=2, channels=8, network='resnet', affine=True).eval()
        self.__assert_flow2d_inverse(realnvp)
        realnvp = RealNVP2d(self.data_shape, n_flows=2, n_blocks=2, channels=8, network='resnet', affine=False).eval()
        self.__assert_flow2d_inverse(realnvp)
        realnvp = RealNVP2d(self.data_shape, n_flows=2, n_blocks=2, channels=8, network='densenet', affine=True).eval()
        self.__assert_flow2d_inverse(realnvp)
        realnvp = RealNVP2d(self.data_shape, n_flows=2, n_blocks=2, channels=8, network='densenet', affine=False).eval()
        self.__assert_flow2d_inverse(realnvp)
        realnvp = RealNVP2d(
            self.data_shape, n_flows=2, n_blocks=2, channels=8, network='resnet', affine=True,
            dequantize=True, logit=0.01
        ).eval()
        self.__assert_flow2d_inverse(realnvp)

    def test_maf(self):
        maf = MAF(self.flattened_data_shape, batch_norm=True).eval()
        self.__assert_flow1d_inverse(maf)
        maf = MAF(self.flattened_data_shape, batch_norm=False).eval()
        self.__assert_flow1d_inverse(maf)
        maf = MAF(self.flattened_data_shape, batch_norm=True, units=8, sequential=False, random_state=42).eval()
        self.__assert_flow1d_inverse(maf)
        maf = MAF(self.flattened_data_shape, batch_norm=False, units=8, sequential=False, random_state=42).eval()
        self.__assert_flow1d_inverse(maf)
        maf = MAF(
            self.flattened_data_shape, batch_norm=True,
            dequantize=True, logit=0.01
        ).eval()
        self.__assert_flow1d_inverse(maf)


if __name__ == '__main__':
    unittest.main()
