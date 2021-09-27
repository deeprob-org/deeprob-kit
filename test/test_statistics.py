import unittest
import numpy as np

from deeprob.utils.statistics import estimate_priors_joints
from deeprob.utils.statistics import compute_mutual_information
from deeprob.utils.statistics import compute_mean_quantiles


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
        self.assertAlmostEqual(mi[0, 1], self.mi)

    def test_compute_mean_quantiles(self):
        mean_quantiles = compute_mean_quantiles(self.data, 2)
        self.assertTrue(np.allclose(mean_quantiles, [[0.2, 0.0], [1.0, 0.4]]))
        self.assertRaises(ValueError, compute_mean_quantiles, self.data, 0)
        self.assertRaises(ValueError, compute_mean_quantiles, self.data, len(self.data) + 1)


if __name__ == '__main__':
    unittest.main()
