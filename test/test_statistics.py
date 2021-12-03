import unittest
import numpy as np
from scipy import linalg

from deeprob.utils.statistics import estimate_priors_joints, compute_mutual_information
from deeprob.utils.statistics import compute_mean_quantiles, compute_gini, compute_bpp, compute_fid


class TestStatistics(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStatistics, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.data = np.array([
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
        cls.priors = np.array([[0.4, 0.6], [0.8, 0.2]])
        cls.joints = np.array([
            [
                [[0.4, 0.0], [0.0, 0.6]],  # (0, 0, 0:2, 0:2)
                [[0.3, 0.1], [0.5, 0.1]]   # (0, 1, 0:2, 0:2)
            ],
            [
                [[0.3, 0.5], [0.1, 0.1]],  # (1, 0, 0:2, 0:2)
                [[0.8, 0.0], [0.0, 0.2]]   # (1, 1, 0:2, 0:2)
            ]
        ])
        cls.mi = 0.3 * np.log(0.3 / (0.4 * 0.8)) \
            + 0.1 * np.log(0.1 / (0.4 * 0.2)) \
            + 0.5 * np.log(0.5 / (0.6 * 0.8)) \
            + 0.1 * np.log(0.1 / (0.6 * 0.2))

    def test_estimate_priors_joints(self):
        priors, joints = estimate_priors_joints(self.data, alpha=0.0)
        self.assertEqual(priors.dtype, self.data.dtype)
        self.assertEqual(joints.dtype, self.data.dtype)
        self.assertTrue(np.allclose(priors, self.priors))
        self.assertTrue(np.allclose(joints, self.joints))

    def test_estimate_mutual_information(self):
        mi = compute_mutual_information(self.priors, self.joints)
        self.assertTrue(np.allclose(np.diag(mi), 0.0))
        self.assertTrue(np.all(mi == mi.T))
        self.assertAlmostEqual(mi[0, 1], self.mi, places=6)

    def test_compute_mean_quantiles(self):
        mean_quantiles = compute_mean_quantiles(self.data, 2)
        self.assertTrue(np.allclose(mean_quantiles, [[0.2, 0.0], [1.0, 0.4]]))
        self.assertRaises(ValueError, compute_mean_quantiles, self.data, 0)
        self.assertRaises(ValueError, compute_mean_quantiles, self.data, len(self.data) + 1)

    def test_compute_gini(self):
        g = 1.0 - (self.priors[0, 0] ** 2.0 + self.priors[0, 1] ** 2.0)
        self.assertEqual(compute_gini(self.priors[0]), g)
        self.assertRaises(ValueError, compute_gini, self.priors[:, 0])

    def test_compute_bpp(self):
        self.assertAlmostEqual(compute_bpp(100.0, 10), -100.0 / np.log(2.0) / 10)

    def test_compute_fid(self):
        dim = 2048
        m1, c1, m2, c2 = np.zeros(dim), np.eye(dim), np.ones(dim), np.eye(dim)
        self.assertAlmostEqual(compute_fid(m1, c1, m2, c2), dim)
        self.assertRaises(ValueError, compute_fid, m1[1:], m2, c1, c2)
        self.assertRaises(ValueError, compute_fid, m1, c1[1:, 1:], m2, c2)
        self.assertRaises(ValueError, compute_fid, np.stack([m1, m1]), c1, m2, c2)
        self.assertRaises(ValueError, compute_fid, m1, c1, m2, np.stack([c2, c2]))


if __name__ == '__main__':
    unittest.main()
