import unittest
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.datasets import load_diabetes
from test.utils import resample_data

from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.learning.wrappers import learn_estimator
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.spn.learning.em import expectation_maximization


class TestEM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEM, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        random_state = np.random.RandomState(42)
        data, _, = load_diabetes(return_X_y=True)
        data = (data < np.median(data, axis=0)).astype(np.float32)
        cls.n_samples, cls.n_features = data.shape
        cls.evi_data = resample_data(data, 1000, random_state)
        cls.blobs_data, _ = make_blobs(
            n_samples=1000, n_features=2, random_state=1337,
            centers=[[-1.0, 1.0], [1.0, -1.0]], cluster_std=0.25
        )

    @staticmethod
    def __build_normal_spn():
        g0a, g1a = Gaussian(0, -0.5, 0.5), Gaussian(1, 0.5, 0.5)
        g0b, g1b = Gaussian(0, 0.5, 0.5), Gaussian(1, -0.5, 0.5)
        p0 = Product(children=[g0a, g1a])
        p1 = Product(children=[g0b, g1b])
        p2 = Product(children=[g0a, g1b])
        s0 = Sum(children=[p0, p1, p2], weights=[0.3, 0.5, 0.2])
        s0.id, p0.id, p1.id, p2.id = 0, 1, 2, 3
        g0a.id, g1a.id, g0b.id, g1b.id = 4, 5, 6, 7
        return s0

    def __learn_spn_mle(self):
        return learn_estimator(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features,
            learn_leaf='mle', split_rows='gmm', split_cols='gvs', min_rows_slice=512,
            random_state=42, verbose=False
        )

    def __learn_spn_clt(self):
        return learn_estimator(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features,
            learn_leaf='binary-clt', split_rows='kmeans', split_cols='gvs', min_rows_slice=512,
            learn_leaf_kwargs={'to_pc': False},
            random_state=42, verbose=False
        )

    def test_spn_binary(self):
        spn = self.__learn_spn_mle()
        expectation_maximization(
            spn, self.evi_data, num_iter=100, batch_perc=0.1, step_size=0.5,
            random_init=False, random_state=42, verbose=False
        )
        ll = log_likelihood(spn, self.evi_data).mean()
        self.assertAlmostEqual(ll, -5.3, places=1)

    def test_clt_binary(self):
        spn = self.__learn_spn_clt()
        expectation_maximization(
            spn, self.evi_data, num_iter=100, batch_perc=0.1, step_size=0.5,
            random_init=True, random_state=42, verbose=False
        )
        ll = log_likelihood(spn, self.evi_data).mean()
        self.assertAlmostEqual(ll, -5.1, places=1)

    def test_spn_gaussian(self):
        spn = self.__build_normal_spn()
        expectation_maximization(
            spn, self.blobs_data, num_iter=25, batch_perc=0.1, step_size=0.5,
            random_init=True, random_state=42, verbose=False
        )
        ll = log_likelihood(spn, self.blobs_data).mean()
        self.assertAlmostEqual(ll, -0.7, places=1)


if __name__ == '__main__':
    unittest.main()
