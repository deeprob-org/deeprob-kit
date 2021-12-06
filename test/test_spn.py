import unittest
import tempfile

from sklearn.datasets import load_diabetes
from test.utils import *

from deeprob.spn.utils.statistics import compute_statistics
from deeprob.spn.utils.filter import filter_nodes_by_type
from deeprob.spn.utils.validity import check_spn
from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.structure.node import bfs, dfs_post_order, topological_order, topological_order_layered
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.structure.io import save_spn_json, load_spn_json
from deeprob.spn.learning.learnspn import learn_spn
from deeprob.spn.learning.wrappers import learn_estimator, learn_classifier
from deeprob.spn.algorithms.structure import prune, marginalize
from deeprob.spn.algorithms.inference import likelihood, log_likelihood, mpe
from deeprob.spn.algorithms.moments import expectation, variance, skewness, kurtosis, moment


class TestSPN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSPN, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        random_state = np.random.RandomState(42)
        data, _, = load_diabetes(return_X_y=True)
        data = (data < np.median(data, axis=0)).astype(np.float32)
        cls.n_samples, cls.n_features = data.shape
        cls.evi_data = resample_data(data, 1000, random_state)
        cls.mar_data = random_marginalize_data(cls.evi_data, 0.2, random_state)

        cls.clf_index = 2
        cls.clf_data = marginalize_data(cls.evi_data, [cls.clf_index])
        cls.scope = [5, 9, 8]
        mar_scope = [s for s in range(cls.n_features) if s not in cls.scope]
        cls.scope_mar_data = marginalize_data(cls.evi_data, mar_scope)

        cls.binary_square_data = np.stack([
            random_state.binomial(1, 0.3, size=1000),
            random_state.binomial(1, 0.9, size=1000)
        ], axis=1)

        cls.complete_data = complete_binary_data(cls.n_features)
        mar_features = [1, 2, 3, 5, 8]
        cls.complete_mar_data = complete_marginalized_binary_data(cls.n_features, mar_features)
        cls.complete_mpe_data = complete_posterior_binary_data(cls.n_features, mar_features)

    @staticmethod
    def __build_normal_spn():
        g0a, g1a = Gaussian(0, 0.0, 1.0), Gaussian(1, 0.0, 1.0)
        g0b, g1b = Gaussian(0, 0.0, 1.0), Gaussian(1, 2.0, 0.5)
        p0 = Product(children=[g0a, g1a])
        p1 = Product(children=[g0b, g1b])
        s0 = Sum(children=[p0, p1], weights=[0.8, 0.2])
        s0.id, p0.id, p1.id = 0, 1, 2
        g0a.id, g1a.id, g0b.id, g1b.id = 3, 4, 5, 6
        return s0

    @staticmethod
    def __build_dag_spn():
        b0a, b1a = Bernoulli(0), Bernoulli(1)
        b0b, b1b = Bernoulli(0), Bernoulli(1)
        p0 = Product(children=[b0a, b1a])
        p1 = Product(children=[b0b, b1b])
        p2 = Product(children=[b0a, b1b])
        s0 = Sum(children=[p0, p1, p2], weights=[0.4, 0.4, 0.2])
        s0.id, p0.id, p1.id, p2.id = 0, 1, 2, 3
        b0a.id, b1a.id, b0b.id, b1b.id = 4, 5, 6, 7
        return s0

    @staticmethod
    def __build_cyclical_spn():
        b0a, b1a = Bernoulli(0), Bernoulli(1)
        b0b, b1b = Bernoulli(0), Bernoulli(1)
        p0 = Product(children=[b0a, b1a])
        p1 = Product(children=[b0b, b1b])
        s0 = Sum(children=[p0, p1], weights=[0.5, 0.5])
        s0.children.append(s0)
        s0.weights = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        s0.id, p0.id, p1.id = 0, 1, 2
        b0a.id, b1a.id, b0b.id, b1b.id = 3, 4, 5, 6
        return s0

    def __learn_binary_clt(self):
        scope = list(range(self.n_features))
        clt = BinaryCLT(scope, root=0)
        clt.fit(self.evi_data, [[0, 1]] * self.n_features, alpha=0.1, random_state=42)
        return clt

    def __learn_spn_unpruned(self):
        return learn_spn(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features,
            learn_leaf='mle', split_cols='gvs', min_rows_slice=64,
            random_state=42, verbose=False
        )

    def __learn_spn_mle(self):
        return learn_estimator(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features,
            learn_leaf='mle', split_rows='gmm', split_cols='gvs', min_rows_slice=64,
            random_state=42, verbose=False
        )

    def __learn_spn_clt(self):
        return learn_estimator(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features,
            learn_leaf='binary-clt', split_rows='kmeans', split_cols='gvs', min_rows_slice=64,
            learn_leaf_kwargs={'to_pc': False},
            random_state=42, verbose=False
        )

    def __learn_spn_mle_classifier(self):
        return learn_classifier(
            self.evi_data, [Bernoulli] * self.n_features, [[0, 1]] * self.n_features, class_idx=self.clf_index,
            learn_leaf='binary-clt', split_cols='rdc', min_rows_slice=64, learn_leaf_kwargs={'to_pc': True},
            random_state=42, verbose=False
        )

    def test_nodes_exceptions(self):
        self.assertRaises(ValueError, Sum)
        self.assertRaises(ValueError, Sum, [])
        self.assertRaises(ValueError, Sum, [0, 1, 1, 3])
        self.assertRaises(ValueError, Sum, [0, 1], children=[Sum([1]), Sum([0])], weights=[0.5, 0.5])
        self.assertRaises(ValueError, Sum, [0, 1], children=[Sum([0, 1]), Sum([0, 1])], weights=[0.5, 0.3, 0.2])
        self.assertRaises(ValueError, Sum, [0, 1], children=[Sum([0, 1]), Sum([0, 1])], weights=[0.5, 0.1])
        self.assertRaises(ValueError, Product)
        self.assertRaises(ValueError, Product, [])
        self.assertRaises(ValueError, Product, [0, 1, 1, 3])
        self.assertRaises(ValueError, Product, [0, 1], children=[Bernoulli(1), Bernoulli(1)])

    def test_validity(self):
        spn = self.__build_dag_spn()
        spn.weights = 2.0 * spn.weights
        self.assertRaises(ValueError, check_spn, spn, smooth=True)
        spn = self.__build_dag_spn()
        spn.children[0].children[0] = Bernoulli(1)
        self.assertRaises(ValueError, check_spn, spn, decomposable=True)
        spn = self.__build_dag_spn()
        spn.id = 42
        self.assertRaises(ValueError, check_spn, spn)
        spn = self.__build_dag_spn()
        spn.children[0].id = 42
        self.assertRaises(ValueError, check_spn, spn)

    def test_complete_inference(self):
        spn = self.__learn_spn_mle()
        ls = likelihood(spn, self.complete_data)
        lls = log_likelihood(spn, self.complete_data)
        self.assertAlmostEqual(np.sum(ls).item(), 1.0, places=6)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=6)
        spn = self.__learn_spn_clt()
        ls = likelihood(spn, self.complete_data)
        lls = log_likelihood(spn, self.complete_data)
        self.assertAlmostEqual(np.sum(ls).item(), 1.0, places=6)
        self.assertAlmostEqual(np.sum(np.exp(lls)).item(), 1.0, places=6)

    def test_mar_inference(self):
        spn = self.__learn_spn_mle()
        evi_ll = log_likelihood(spn, self.evi_data).mean()
        mar_ll = log_likelihood(spn, self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)
        spn = self.__learn_spn_clt()
        evi_ll = log_likelihood(spn, self.evi_data).mean()
        mar_ll = log_likelihood(spn, self.mar_data).mean()
        self.assertGreater(mar_ll, evi_ll)

    def test_mpe_inference(self):
        spn = self.__learn_spn_mle()
        evi_ll = log_likelihood(spn, self.evi_data).mean()
        mpe_data = mpe(spn, self.mar_data)
        mpe_ll = log_likelihood(spn, mpe_data).mean()
        self.assertFalse(np.any(np.isnan(mpe_data)))
        self.assertGreater(mpe_ll, evi_ll)
        spn = self.__learn_spn_clt()
        evi_ll = log_likelihood(spn, self.evi_data).mean()
        mpe_data = mpe(spn, self.mar_data)
        mpe_ll = log_likelihood(spn, mpe_data).mean()
        self.assertFalse(np.any(np.isnan(mpe_data)))
        self.assertGreater(mpe_ll, evi_ll)

    def test_mpe_complete_inference(self):
        spn = self.__learn_binary_clt().to_pc()
        complete_lls = log_likelihood(spn, self.complete_data)
        mpe_data = mpe(spn, self.complete_mar_data)
        mpe_ids = binary_data_ids(mpe_data).tolist()
        expected_mpe_ids = compute_mpe_ids(self.complete_mpe_data, complete_lls.squeeze())
        self.assertEqual(mpe_ids, expected_mpe_ids)

    def test_classifier(self):
        spn = self.__learn_spn_mle_classifier()
        clf_data = mpe(spn, self.clf_data)
        error_rate = np.mean(np.abs(clf_data[:, self.clf_index] - self.evi_data[:, self.clf_index]))
        self.assertFalse(np.any(np.isnan(clf_data)))
        self.assertGreater(1.0 - error_rate, 0.7)

    def test_bfs(self):
        spn = self.__build_dag_spn()
        node_ids = list(map(lambda n: n.id, bfs(spn)))
        self.assertEqual(node_ids, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_dfs_post_order(self):
        spn = self.__build_dag_spn()
        node_ids = list(map(lambda n: n.id, dfs_post_order(spn)))
        self.assertEqual(node_ids, [7, 4, 3, 6, 2, 5, 1, 0])

    def test_topological_order(self):
        spn = self.__build_dag_spn()
        ordering = topological_order(spn)
        node_ids = list(map(lambda node: node.id, ordering))
        self.assertEqual(node_ids, [0, 1, 2, 3, 5, 6, 4, 7])
        spn = self.__build_cyclical_spn()
        ordering = topological_order(spn)
        self.assertIsNone(ordering)

    def test_topological_order_layered(self):
        spn = self.__build_dag_spn()
        layers = topological_order_layered(spn)
        node_layered_ids = list(map(lambda layer: list(map(lambda node: node.id, layer)), layers))
        self.assertEqual(node_layered_ids, [[0], [1, 2, 3], [5, 6, 4, 7]])
        spn = self.__build_cyclical_spn()
        layers = topological_order_layered(spn)
        self.assertIsNone(layers)

    def test_prune(self):
        spn = self.__learn_spn_unpruned()
        ll = log_likelihood(spn, self.evi_data).mean()
        pruned_spn = prune(spn)
        pruned_ll = log_likelihood(pruned_spn, self.evi_data).mean()
        repruned_spn = prune(pruned_spn)
        repruned_ll = log_likelihood(repruned_spn, self.evi_data).mean()
        self.assertAlmostEqual(ll, pruned_ll, places=6)
        self.assertEqual(pruned_ll, repruned_ll)

    def test_marginalize(self):
        spn = self.__learn_spn_clt()
        mar_ll = log_likelihood(spn, self.scope_mar_data).mean()
        mar_spn = marginalize(spn, self.scope)
        struct_mar_ll = log_likelihood(mar_spn, self.evi_data).mean()
        self.assertAlmostEqual(struct_mar_ll, mar_ll, places=6)
        self.assertRaises(ValueError, marginalize, spn, [])
        self.assertRaises(ValueError, marginalize, spn, [42])

    def test_moments(self):
        spn = self.__build_normal_spn()
        self.assertAlmostEqual(expectation(spn)[1], 0.4, places=6)
        self.assertAlmostEqual(variance(spn)[1], 1.49, places=6)
        self.assertAlmostEqual(skewness(spn)[0], 0.0, places=6)
        self.assertAlmostEqual(kurtosis(spn)[0], 0.0, places=6)
        self.assertTrue(np.all(moment(spn, order=0) == 1.0))
        self.assertRaises(ValueError, moment, spn, order=-1)

    def test_compute_statistics(self):
        spn = self.__build_dag_spn()
        stats = compute_statistics(spn)
        self.assertEqual(
            stats, {'n_nodes': 8, 'n_sum': 1, 'n_prod': 3, 'n_leaves': 4, 'n_edges': 9, 'n_params': 7, 'depth': 2}
        )

    def test_filter_nodes_by_type(self):
        spn = self.__build_dag_spn()
        sums_prods = filter_nodes_by_type(spn, (Sum, Product))
        self.assertEqual(list(map(lambda x: type(x), sums_prods)), [Sum, Product, Product, Product])

    def test_save_load_json(self):
        spn = self.__build_dag_spn()
        ll = log_likelihood(spn, self.binary_square_data).mean()
        with tempfile.TemporaryFile('r+') as f:
            save_spn_json(spn, f)
            f.seek(0)
            loaded_spn = load_spn_json(f)
        loaded_ll = log_likelihood(loaded_spn, self.binary_square_data).mean()
        self.assertEqual(ll, loaded_ll)


if __name__ == '__main__':
    unittest.main()
