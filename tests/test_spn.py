from copy import deepcopy

import pytest
import tempfile
import numpy as np

from tests.utils import resample_data, marginalize_data, complete_binary_data, binary_data_ids, compute_mpe_ids
from tests.utils import random_marginalize_data, complete_posterior_binary_data, complete_marginalized_binary_data
from sklearn.datasets import load_diabetes

from deeprob.spn.utils.statistics import compute_statistics
from deeprob.spn.utils.filter import filter_nodes_by_type
from deeprob.spn.utils.validity import check_spn
from deeprob.spn.structure.node import Sum, Product
from deeprob.spn.structure.node import bfs, dfs_post_order, topological_order, topological_order_layered
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.leaf import Bernoulli, Gaussian
from deeprob.spn.structure.io import save_spn_json, load_spn_json
from deeprob.spn.learning.xpc import SD_LEVEL_1, SD_LEVEL_2
from deeprob.spn.learning.learnspn import learn_spn
from deeprob.spn.learning.wrappers import learn_estimator, learn_classifier
from deeprob.spn.algorithms.structure import prune, marginalize
from deeprob.spn.algorithms.inference import likelihood, log_likelihood, mpe
from deeprob.spn.algorithms.moments import expectation, variance, skewness, kurtosis, moment


@pytest.fixture
def data():
    data, _, = load_diabetes(return_X_y=True)
    return (data < np.median(data, axis=0)).astype(np.float32)


@pytest.fixture
def evi_data(data):
    return resample_data(data, 1000)


@pytest.fixture
def mar_data(evi_data):
    return random_marginalize_data(evi_data, 0.2)


@pytest.fixture
def clf_data(evi_data):
    return marginalize_data(evi_data, [2])


@pytest.fixture
def scope_mar_data(evi_data):
    scope = [5, 9, 8]
    mar_scope = [s for s in range(10) if s not in scope]
    return marginalize_data(evi_data, mar_scope)


@pytest.fixture
def binary_square_data():
    return np.stack([
        np.random.binomial(1, 0.3, size=1000),
        np.random.binomial(1, 0.9, size=1000)
    ], axis=1)


@pytest.fixture
def complete_data():
    return complete_binary_data(10)


@pytest.fixture
def complete_mar_data():
    return complete_marginalized_binary_data(10, [1, 2, 3, 5, 8])


@pytest.fixture
def complete_mpe_data():
    return complete_posterior_binary_data(10, [1, 2, 3, 5, 8])


@pytest.fixture
def gaussian_spn():
    g0a, g1a = Gaussian(0, 0.0, 1.0), Gaussian(1, 0.0, 1.0)
    g0b, g1b = Gaussian(0, 0.0, 1.0), Gaussian(1, 2.0, 0.5)
    p0 = Product(children=[g0a, g1a])
    p1 = Product(children=[g0b, g1b])
    s0 = Sum(children=[p0, p1], weights=[0.8, 0.2])
    s0.id, p0.id, p1.id = 0, 1, 2
    g0a.id, g1a.id, g0b.id, g1b.id = 3, 4, 5, 6
    return s0


@pytest.fixture
def dag_spn():
    b0a, b1a = Bernoulli(0), Bernoulli(1)
    b0b, b1b = Bernoulli(0), Bernoulli(1)
    p0 = Product(children=[b0a, b1a])
    p1 = Product(children=[b0b, b1b])
    p2 = Product(children=[b0a, b1b])
    s0 = Sum(children=[p0, p1, p2], weights=[0.4, 0.4, 0.2])
    s0.id, p0.id, p1.id, p2.id = 0, 1, 2, 3
    b0a.id, b1a.id, b0b.id, b1b.id = 4, 5, 6, 7
    return s0


@pytest.fixture
def cyclical_spn():
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


@pytest.fixture
def binary_clt(evi_data):
    scope = list(range(evi_data.shape[1]))
    clt = BinaryCLT(scope, root=0)
    clt.fit(evi_data, [[0, 1]] * evi_data.shape[1], alpha=0.1, random_state=42)
    return clt


@pytest.fixture
def spn_unpruned(evi_data):
    return learn_spn(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1],
        learn_leaf='mle', split_cols='gvs', min_rows_slice=64,
        random_state=42, verbose=False
    )


@pytest.fixture
def spn_mle(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1],
        learn_leaf='mle', split_rows='gmm', split_cols='gvs', min_rows_slice=64,
        random_state=42, verbose=False
    )


@pytest.fixture
def spn_clt(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1],
        learn_leaf='binary-clt', split_rows='kmeans', split_cols='gvs', min_rows_slice=64,
        learn_leaf_kwargs={'to_pc': False},
        random_state=42, verbose=False
    )


@pytest.fixture
def spn_mle_classifier(evi_data):
    return learn_classifier(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1], class_idx=1,
        learn_leaf='binary-clt', split_cols='rdc', min_rows_slice=64, learn_leaf_kwargs={'to_pc': True},
        random_state=42, verbose=False
    )


@pytest.fixture
def sd_xpc(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1], method='xpc',
        det=False, sd=True, min_part_inst=64, conj_len=3, arity=4,
        use_greedy_ordering=False, random_seed=42
    )


@pytest.fixture
def det_sd_xpc(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1], method='xpc',
        det=True, sd=True, min_part_inst=64, conj_len=3, arity=4,
        use_greedy_ordering=True, random_seed=42
    )


@pytest.fixture
def partial_sd_expc(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1], method='ensemble-xpc',
        ensemble_dim=10, det=False, sd_level=SD_LEVEL_1, min_part_inst=64, conj_len=3, arity=4,
        random_seed=42
    )


@pytest.fixture
def full_sd_expc(evi_data):
    return learn_estimator(
        evi_data, [Bernoulli] * evi_data.shape[1], [[0, 1]] * evi_data.shape[1], method='ensemble-xpc',
        ensemble_dim=10, det=False, sd_level=SD_LEVEL_2, min_part_inst=64, conj_len=3, arity=4,
        random_seed=42
    )


def test_nodes_exceptions():
    with pytest.raises(ValueError):
        Sum()
        Sum([])
        Sum([0, 1, 1, 3])
        Sum([0, 1], children=[Sum([1]), Sum([0])], weights=[0.5, 0.5])
        Sum([0, 1], children=[Sum([0, 1]), Sum([0, 1])], weights=[0.5, 0.3, 0.2])
        Sum([0, 1], children=[Sum([0, 1]), Sum([0, 1])], weights=[0.5, 0.1])
        Product()
        Product([])
        Product([0, 1, 1, 3])
        Product([0, 1], children=[Bernoulli(1), Bernoulli(1)])


def test_validity(dag_spn):
    spn = deepcopy(dag_spn)
    spn.weights = 2.0 * spn.weights
    with pytest.raises(ValueError):
        check_spn(spn, smooth=True)
    spn = deepcopy(dag_spn)
    spn.children[0].children[0] = Bernoulli(1)
    with pytest.raises(ValueError):
        check_spn(spn, decomposable=True)
    spn = deepcopy(dag_spn)
    spn.id = 42
    with pytest.raises(ValueError):
        check_spn(spn)
    spn = deepcopy(dag_spn)
    spn.children[0].id = 42
    with pytest.raises(ValueError):
        check_spn(spn)


def test_complete_inference(spn_mle, spn_clt, complete_data):
    ls = likelihood(spn_mle, complete_data)
    lls = log_likelihood(spn_mle, complete_data)
    assert np.isclose(np.sum(ls).item(), 1.0)
    assert np.isclose(np.sum(np.exp(lls)).item(), 1.0)

    ls = likelihood(spn_clt, complete_data)
    lls = log_likelihood(spn_clt, complete_data)
    assert np.isclose(np.sum(ls).item(), 1.0)
    assert np.isclose(np.sum(np.exp(lls)).item(), 1.0)

    p_ls = likelihood(spn_clt, complete_data, n_jobs=-1)
    p_lls = log_likelihood(spn_clt, complete_data, n_jobs=-1)
    assert np.isclose(p_ls, ls).all()
    assert np.isclose(p_lls, lls).all()


def test_mar_inference(spn_mle, spn_clt, evi_data, mar_data):
    evi_ll = log_likelihood(spn_mle, evi_data)
    mar_ll = log_likelihood(spn_mle, mar_data)
    assert np.all(mar_ll >= evi_ll)

    evi_ll = log_likelihood(spn_clt, evi_data)
    mar_ll = log_likelihood(spn_clt, mar_data)
    assert np.all(mar_ll >= evi_ll)

    p_mar_ll = log_likelihood(spn_clt, mar_data, n_jobs=-1)
    assert np.isclose(p_mar_ll, mar_ll).all()


def test_mpe_inference(spn_mle, spn_clt, evi_data, mar_data):
    evi_ll = log_likelihood(spn_mle, evi_data)
    mpe_data = mpe(spn_mle, mar_data)
    mpe_ll = log_likelihood(spn_mle, mpe_data)
    assert not np.any(np.isnan(mpe_data))
    assert mpe_ll.mean() > evi_ll.mean()

    evi_ll = log_likelihood(spn_clt, evi_data)
    mpe_data = mpe(spn_clt, mar_data)
    mpe_ll = log_likelihood(spn_clt, mpe_data)
    assert not np.any(np.isnan(mpe_data))
    assert mpe_ll.mean() > evi_ll.mean()


def test_mpe_complete_inference(binary_clt, complete_data, complete_mar_data, complete_mpe_data):
    spn = binary_clt.to_pc()
    complete_lls = log_likelihood(spn, complete_data)
    mpe_data = mpe(spn, complete_mar_data)
    mpe_ids = binary_data_ids(mpe_data).tolist()
    expected_mpe_ids = compute_mpe_ids(complete_mpe_data, complete_lls.squeeze())
    assert mpe_ids == expected_mpe_ids

    complete_lls = log_likelihood(spn, complete_data, n_jobs=-1)
    mpe_data = mpe(spn, complete_mar_data, n_jobs=-1)
    mpe_ids = binary_data_ids(mpe_data).tolist()
    expected_mpe_ids = compute_mpe_ids(complete_mpe_data, complete_lls.squeeze())
    assert mpe_ids == expected_mpe_ids


def test_classifier(spn_mle, clf_data, evi_data):
    clf_data = mpe(spn_mle, clf_data)
    error_rate = np.mean(np.abs(clf_data[:, 1] - evi_data[:, 1]))
    assert not np.any(np.isnan(clf_data))
    assert 1.0 - error_rate > 0.7


def test_bfs(dag_spn):
    node_ids = list(map(lambda n: n.id, bfs(dag_spn)))
    assert node_ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_dfs_post_order(dag_spn):
    node_ids = list(map(lambda n: n.id, dfs_post_order(dag_spn)))
    assert node_ids == [7, 4, 3, 6, 2, 5, 1, 0]


def test_topological_order(dag_spn, cyclical_spn):
    ordering = topological_order(dag_spn)
    node_ids = list(map(lambda node: node.id, ordering))
    assert node_ids == [0, 1, 2, 3, 5, 6, 4, 7]

    ordering = topological_order(cyclical_spn)
    assert ordering is None


def test_topological_order_layered(dag_spn, cyclical_spn):
    layers = topological_order_layered(dag_spn)
    node_layered_ids = list(map(lambda layer: list(map(lambda node: node.id, layer)), layers))
    assert node_layered_ids == [[0], [1, 2, 3], [5, 6, 4, 7]]

    layers = topological_order_layered(cyclical_spn)
    assert layers is None


def test_prune(spn_unpruned, evi_data):
    ll = log_likelihood(spn_unpruned, evi_data)
    pruned_spn = prune(spn_unpruned)
    pruned_ll = log_likelihood(pruned_spn, evi_data)
    repruned_spn = prune(pruned_spn)
    repruned_ll = log_likelihood(repruned_spn, evi_data)
    assert np.allclose(ll, pruned_ll)
    assert np.all(pruned_ll == repruned_ll)


def test_marginalize(spn_clt, evi_data, scope_mar_data):
    mar_ll = log_likelihood(spn_clt, scope_mar_data)
    mar_spn = marginalize(spn_clt, [5, 9, 8])
    struct_mar_ll = log_likelihood(mar_spn, evi_data)
    assert np.allclose(struct_mar_ll, mar_ll)
    with pytest.raises(ValueError):
        marginalize(spn_clt, [])
        marginalize(spn_clt, [42])


def test_moments(gaussian_spn):
    assert np.isclose(expectation(gaussian_spn)[1], 0.4)
    assert np.isclose(variance(gaussian_spn)[1], 1.49)
    assert np.isclose(skewness(gaussian_spn)[0], 0.0)
    assert np.isclose(kurtosis(gaussian_spn)[0], 0.0)
    assert np.all(moment(gaussian_spn, order=0) == 1.0)
    with pytest.raises(ValueError):
        moment(gaussian_spn, order=-1)


def test_compute_statistics(dag_spn):
    stats = compute_statistics(dag_spn)
    assert stats == {'n_nodes': 8, 'n_sum': 1, 'n_prod': 3, 'n_leaves': 4, 'n_edges': 9, 'n_params': 7, 'depth': 2}


def test_filter_nodes_by_type(dag_spn):
    sums_prods = filter_nodes_by_type(dag_spn, (Sum, Product))
    assert (list(map(lambda x: type(x), sums_prods)) == [Sum, Product, Product, Product])


def test_save_load_json(dag_spn, binary_square_data, cyclical_spn):
    ll = log_likelihood(dag_spn, binary_square_data)
    with tempfile.TemporaryFile('r+') as f:
        save_spn_json(dag_spn, f)
        with pytest.raises(ValueError):
            save_spn_json(cyclical_spn, f)
        f.seek(0)
        loaded_spn = load_spn_json(f)
    loaded_ll = log_likelihood(loaded_spn, binary_square_data)
    assert np.all(ll == loaded_ll)


def test_xpc_properties(sd_xpc, det_sd_xpc, partial_sd_expc, full_sd_expc):
    try:
        check_spn(sd_xpc, smooth=True, decomposable=True, structured_decomposable=True)
    except ValueError as e:
        assert False, f"{e}"

    try:
        check_spn(det_sd_xpc, smooth=True, decomposable=True, structured_decomposable=True)
    except ValueError as e:
        assert False, f"{e}"

    with pytest.raises(ValueError):
        check_spn(partial_sd_expc, smooth=True, decomposable=True, structured_decomposable=True)
    try:
        check_spn(full_sd_expc, smooth=True, decomposable=True, structured_decomposable=True)
    except ValueError as e:
        assert False, f"{e}"
