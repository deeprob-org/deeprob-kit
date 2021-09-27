import unittest
import numpy as np

from deeprob.spn.structure.leaf import Categorical, Isotonic, Uniform


class TestLeaf(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLeaf, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        random_state = np.random.RandomState(42)
        cls.domain = list(range(6))
        cls.categorical_data = random_state.binomial(len(cls.domain) - 2, p=0.3, size=[1000, 1])
        cls.continuous_data = random_state.randn(100, 1)
        cls.epsilon = np.finfo(np.float64).eps

    def test_categorical(self):
        leaf = Categorical(0)
        leaf.fit(self.categorical_data, domain=self.domain, alpha=1e-5)
        p1 = len(self.categorical_data[self.categorical_data == 1]) / len(self.categorical_data)
        self.assertAlmostEqual(leaf.likelihood(np.ones([1, 1])).item(), p1, places=6)
        self.assertEqual(leaf.mpe(np.full([1, 1], np.nan)).item(), leaf.probabilities.argmax())
        leaf = Categorical(0, **leaf.params_dict())
        self.assertEqual(leaf.mpe(np.full([1, 1], np.nan)).item(), leaf.probabilities.argmax())
        self.assertGreater(leaf.log_likelihood(np.full([1, 1], self.domain[-1])).item(), np.log(self.epsilon))
        self.assertEqual(leaf.params_count(), 2 * len(self.domain))

    def test_isotonic(self):
        leaf = Isotonic(0)
        leaf.fit(self.continuous_data, domain=(-9.0, 9.0), alpha=1e-5)
        densities, breaks = np.histogram(self.continuous_data, bins='fd', density=True)
        idx = np.searchsorted(breaks, 0.8, side='right')
        self.assertAlmostEqual(leaf.likelihood(np.full([1, 1], 0.8)).item(), densities[idx - 1], places=6)
        self.assertTrue(-1.0 < leaf.mpe(np.full([1, 1], np.nan)).item() < 1.0)
        leaf = Isotonic(0, **leaf.params_dict())
        self.assertTrue(-1.0 < leaf.mpe(np.full([1, 1], np.nan)).item() < 1.0)
        self.assertGreater(leaf.log_likelihood(np.full([1, 1], 10.0)).item(), np.log(self.epsilon))
        self.assertEqual(leaf.params_count(), 2 * len(densities) + 1)

    def test_uniform(self):
        leaf = Uniform(0)
        leaf.fit(self.continuous_data, domain=(-9.0, 9.0))
        ls = leaf.likelihood(np.array([[-1.5], [0.0], [2.0]])).flatten()
        self.assertTrue(set(ls), set(leaf.likelihood(np.full([1, 1], 0.3)).flatten()))
        leaf = Uniform(0, **leaf.params_dict())
        self.assertEqual(leaf.mpe(np.full([1, 1], np.nan)).item(), np.min(self.continuous_data))
        self.assertEqual(leaf.params_count(), 2)


if __name__ == '__main__':
    unittest.main()
