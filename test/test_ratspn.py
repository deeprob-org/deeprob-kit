import unittest
import torch

from collections import Counter
from test.utils import complete_binary_data

from deeprob.utils.region import RegionGraph
from deeprob.spn.models.ratspn import BernoulliRatSpn


class TestRatSpn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRatSpn, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_grad_enabled(False)
        cls.n_features = 15
        cls.complete_data = torch.tensor(complete_binary_data(cls.n_features), dtype=torch.float32)

    def __build_bernoulli_ratspn(self):
        return BernoulliRatSpn(
            self.n_features, rg_depth=3, rg_repetitions=4, rg_batch=4, rg_sum=2, random_state=42
        )

    def test_region_graph(self):
        rg = RegionGraph(self.n_features, depth=2, random_state=42)
        layers = rg.make_layers(n_repetitions=2)
        root_region = layers[0][0]
        leaf_arities = set(map(lambda x: len(x), layers[-1]))
        inner_partition_vars = list(map(lambda x: list(sorted(x[0] + x[1])), layers[1]))
        inner_region_vars = Counter(sum(layers[2], tuple()))
        self.assertEqual(root_region, tuple(range(self.n_features)))
        self.assertEqual(leaf_arities, {3, 4})
        self.assertEqual(inner_partition_vars.count(list(range(self.n_features))), 2)
        self.assertEqual(len(inner_region_vars), self.n_features)
        self.assertEqual(set(inner_region_vars.values()), {2})
        self.assertRaises(ValueError, rg.make_layers, n_repetitions=-1)
        self.assertRaises(ValueError, RegionGraph, n_features=-1, depth=1)
        self.assertRaises(ValueError, RegionGraph, n_features=8, depth=0)
        self.assertRaises(ValueError, RegionGraph, n_features=8, depth=4)

    def test_bernoulli_ratspn(self):
        ratspn = self.__build_bernoulli_ratspn()
        lls = ratspn(self.complete_data)
        self.assertAlmostEqual(torch.sum(torch.exp(lls)).item(), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
