import pytest
import numpy as np

from deeprob.context import ContextState
from deeprob.utils.data import check_data_dtype
from deeprob.spn.structure.leaf import Bernoulli
from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.utils.validity import check_spn


def test_context():
    with pytest.raises(ValueError):
        ContextState(unknown_flag=False)


def test_context_check_dtype():
    with ContextState(check_dtype=False):
        assert check_data_dtype(np.zeros(1, dtype=np.uint8), np.float32).dtype == np.uint8
        assert check_data_dtype(np.zeros(1, dtype=np.uint32), np.uint64).dtype == np.uint32
        assert check_data_dtype(np.zeros(1, dtype=np.float32), np.float64).dtype == np.float32


def test_context_check_spn():
    p, s = Product([0]), Sum([0])
    p.children = [Bernoulli(0), Bernoulli(0)]
    s.children = [Bernoulli(0), Bernoulli(1)]
    s.weights = np.array([1.0, 2.0], dtype=np.float32)
    with ContextState(check_spn=False):
        check_spn(s, smooth=True)
        check_spn(p, decomposable=True)
        check_spn(s, labeled=True)
