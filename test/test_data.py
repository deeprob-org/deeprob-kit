import unittest
import numpy as np

from deeprob.utils.data import mixed_ohe_data, ecdf_data, check_data_dtype
from deeprob.utils.data import DataFlatten, DataNormalizer, DataStandardizer


class TestData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestData, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.data_domains = [(0.0, 4.0), [0, 1, 2], (0.5, 2.0), (0.0, 5.0)]
        cls.data = np.array([
            [3.2, 0, 1.6, 5.0],
            [1.0, 1, 0.5, 0.5],
            [3.0, 0, 0.5, 1.5],
            [3.0, 2, 1.0, 1.0],
            [3.5, 1, 1.0, 4.0]
        ])
        cls.ohe_data = np.array([
            [3.2, 1, 0, 0, 1.6, 5.0],
            [1.0, 0, 1, 0, 0.5, 0.5],
            [3.0, 1, 0, 0, 0.5, 1.5],
            [3.0, 0, 0, 1, 1.0, 1.0],
            [3.5, 0, 1, 0, 1.0, 4.0]
        ])
        cls.tensor_data = np.stack([cls.data, cls.data], axis=2)

    def test_mixed_ohe_data(self):
        data = mixed_ohe_data(self.data, self.data_domains)
        self.assertEqual(data.tolist(), self.ohe_data.tolist())

    def test_ecdf_data(self):
        unnorm_ecdf0 = len(self.ohe_data) * ecdf_data(self.ohe_data[:, 0])
        unnorm_ecdf1 = len(self.ohe_data) * ecdf_data(self.ohe_data[:, 1])
        self.assertEqual(unnorm_ecdf0.tolist(), [4, 1, 3, 3, 5])
        self.assertEqual(unnorm_ecdf1.tolist(), [5, 3, 5, 3, 3])

    def test_check_data_dtype(self):
        uint8_data = np.arange(5).astype(np.uint8)
        uint32_data = np.arange(5).astype(np.uint32)
        float32_data = np.arange(5).astype(np.float32)
        float64_data = np.arange(5).astype(np.float64)
        self.assertEqual(check_data_dtype(uint8_data, np.float32).dtype, np.float32)
        self.assertEqual(check_data_dtype(uint32_data, np.uint64).dtype, np.uint64)
        self.assertEqual(check_data_dtype(float32_data, np.float32).dtype, np.float32)
        self.assertEqual(check_data_dtype(float64_data, np.float32).dtype, np.float64)

    def test_data_flatten(self):
        transform = DataFlatten()
        transform.fit(self.tensor_data)
        data = transform.forward(self.tensor_data)
        orig_data = transform.backward(data)
        self.assertEqual(data.shape, (5, 8))
        self.assertTrue(np.alltrue(orig_data == self.tensor_data))

    def test_data_normalizer(self):
        transform = DataNormalizer((2.0, 4.0))
        transform.fit(self.data)
        data = transform.forward(self.data)
        orig_data = transform.backward(data)
        self.assertEqual((np.min(data), np.max(data)), (2.0, 4.0))
        self.assertTrue(np.allclose(orig_data, self.data))

    def test_data_standardizer_sample_wise(self):
        transform = DataStandardizer(sample_wise=True)
        transform.fit(self.data)
        data = transform.forward(self.data)
        orig_data = transform.backward(data)
        self.assertTrue(np.allclose(data.mean(axis=0), 0.0, atol=1e-7))
        self.assertTrue(np.allclose(data.std(axis=0), 1.0, atol=1e-7))
        self.assertTrue(np.allclose(orig_data, self.data, atol=1e-6))

    def test_data_standardizer_feature_wise(self):
        transform = DataStandardizer(sample_wise=False)
        transform.fit(self.data)
        data = transform.forward(self.data)
        orig_data = transform.backward(data)
        self.assertTrue(np.allclose(data.mean(), 0.0, atol=1e-7))
        self.assertTrue(np.allclose(data.std(), 1.0, atol=1e-7))
        self.assertTrue(np.allclose(orig_data, self.data, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
