import unittest
import numpy as np

from deeprob.context import ContextState
from deeprob.utils.data import check_data_dtype
from deeprob.spn.structure.leaf import Bernoulli
from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.utils.validity import check_spn


class TestContext(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestContext, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        pass

    def test_context(self):
        self.assertRaises(ValueError, ContextState, unknown_flag=False)

    def test_context_check_dtype(self):
        with ContextState(check_dtype=False):
            self.assertEqual(check_data_dtype(np.zeros(1, dtype=np.uint8), np.float32).dtype, np.uint8)
            self.assertEqual(check_data_dtype(np.zeros(1, dtype=np.uint32), np.uint64).dtype, np.uint32)
            self.assertEqual(check_data_dtype(np.zeros(1, dtype=np.float32), np.float64).dtype, np.float32)

    def test_context_check_spn(self):
        p, s = Product([0]), Sum([0])
        p.children = [Bernoulli(0), Bernoulli(0)]
        s.children = [Bernoulli(0), Bernoulli(1)]
        s.weights = np.array([1.0, 2.0], dtype=np.float32)
        with ContextState(check_spn=False):
            check_spn(s, smooth=True)
            check_spn(p, decomposable=True)
            check_spn(s, labeled=True)


if __name__ == '__main__':
    unittest.main()
