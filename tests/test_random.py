import pytest
import numpy as np

from deeprob.utils.random import check_random_state


def test_check_random_state():
    assert check_random_state().__class__ == np.random.RandomState
    assert check_random_state(42).__class__ == np.random.RandomState
    with pytest.raises(ValueError):
        check_random_state(np.arange(10))
