import unittest
import torch
import numpy as np

from deeprob.spn.layers.dgcspn import SpatialGaussianLayer, SpatialProductLayer, SpatialSumLayer, SpatialRootLayer


class TestDgcSpn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDgcSpn, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_grad_enabled(False)
        cls.in_features = (3, 32, 32)
        cls.in_channels, cls.in_height, cls.in_width = cls.in_features
        cls.data = torch.ones([8, *cls.in_features])

    def test_gaussian_layer(self):
        gaussian = SpatialGaussianLayer(
            self.in_features, out_channels=16,
            optimize_scale=False, uniform_loc=(-1.0, 1.0)
        )
        self.assertEqual(gaussian.out_features, (16, self.in_height, self.in_width))
        self.assertTrue(torch.all(gaussian.loc >= -1.0) and torch.all(gaussian.loc <= 1.0))
        self.assertTrue(torch.all(gaussian.scale == 1.0))
        self.assertFalse(gaussian.scale.requires_grad)
        gaussian = SpatialGaussianLayer(
            self.in_features, out_channels=16,
            optimize_scale=True
        )
        self.assertTrue(torch.all(gaussian.scale > 0.0))
        self.assertTrue(gaussian.scale.requires_grad)
        self.assertRaises(
            ValueError, SpatialGaussianLayer, self.in_features, 16,
            quantiles_loc=np.zeros([16, *self.in_features]), uniform_loc=(-1.0, 1.0)
        )

    def test_depthwise_product_layer(self):
        product = SpatialProductLayer(
            self.in_features, kernel_size=2, padding='full',
            stride=1, dilation=4, depthwise=True
        )
        self.assertEqual(product.pad, [4, 4, 4, 4])
        self.assertTrue(torch.all(product.weight == torch.ones(self.in_channels, 1, 2, 2)))
        self.assertEqual(product.out_features, (self.in_channels, self.in_height + 4, self.in_width + 4))
        self.assertTrue(torch.allclose(product(self.data)[:, :, 4:-4, 4:-4], torch.tensor(4.0)))
        product = SpatialProductLayer(
            self.in_features, kernel_size=2, padding='valid',
            stride=2, dilation=1, depthwise=True
        )
        self.assertEqual(product.pad, [0, 0, 0, 0])
        self.assertTrue(torch.all(product.weight == torch.ones(self.in_channels, 1, 2, 2)))
        self.assertEqual(product.out_features, (self.in_channels, self.in_height // 2, self.in_width // 2))
        self.assertTrue(torch.allclose(product(self.data), torch.tensor(4.0)))

    def test_non_depthwise_product_layer(self):
        product = SpatialProductLayer(
            self.in_features, kernel_size=2, padding='full',
            stride=1, dilation=8, depthwise=False
        )
        self.assertEqual(product.pad, [8, 8, 8, 8])
        self.assertEqual(tuple(product.weight.shape), (self.in_channels ** 4, self.in_channels, 2, 2))
        self.assertTrue(torch.allclose(torch.sum(product.weight, dim=1), torch.tensor(1.0)))
        self.assertEqual(product.out_features, (self.in_channels ** 4, self.in_height + 8, self.in_width + 8))
        self.assertTrue(torch.allclose(product(self.data)[:, :, 8:-8, 8:-8], torch.tensor(4.0)))

    def test_sum_layer(self):
        sum = SpatialSumLayer(self.in_features, out_channels=8)
        self.assertEqual(sum.out_features, (8, self.in_height, self.in_width))

    def test_root_layer(self):
        root = SpatialRootLayer(self.in_features, out_channels=8)
        self.assertEqual(root.out_channels, 8)
        self.assertEqual(tuple(root.weight.shape), (8, np.prod(self.in_features)))


if __name__ == '__main__':
    unittest.main()
