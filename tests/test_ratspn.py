import numpy as np
import pytest
import torch

from collections import Counter
from tests.utils import complete_binary_data

from deeprob.utils.region import RegionGraph
from deeprob.spn.models.ratspn import BernoulliRatSpn


@pytest.fixture
def complete_data():
    return torch.tensor(complete_binary_data(15), dtype=torch.float32)


@pytest.fixture
def bernoulli_ratspn():
    return BernoulliRatSpn(
        15, rg_depth=3, rg_repetitions=4, rg_batch=4, rg_sum=2, random_state=42
    )


def test_region_graph():
    rg = RegionGraph(15, depth=2, random_state=42)
    layers = rg.make_layers(n_repetitions=2)
    root_region = layers[0][0]
    leaf_arities = set(map(lambda x: len(x), layers[-1]))
    inner_partition_vars = list(map(lambda x: list(sorted(x[0] + x[1])), layers[1]))
    inner_region_vars = Counter(sum(layers[2], tuple()))
    assert root_region == tuple(range(15))
    assert leaf_arities == {3, 4}
    assert inner_partition_vars.count(list(range(15))) == 2
    assert len(inner_region_vars) == 15
    assert set(inner_region_vars.values()) == {2}
    with pytest.raises(ValueError):
        rg.make_layers(n_repetitions=-1)
        RegionGraph(n_features=-1, depth=1)
        RegionGraph(n_features=8, depth=0)
        RegionGraph(n_features=8, depth=4)


def test_bernoulli_ratspn(bernoulli_ratspn, complete_data):
    lls = bernoulli_ratspn(complete_data)
    assert np.isclose(torch.sum(torch.exp(lls)).item(), 1.0)
