import pytest
import numpy as np

from deeprob.utils.graph import build_tree_structure, compute_bfs_ordering, maximum_spanning_tree


@pytest.fixture
def tree():
    return [1, -1, 0, 0, 1, 4, 4, 5]


@pytest.fixture
def scope():
    return [3, 5, 2, 0, 7, 1, 4, 9]


@pytest.fixture
def adj_matrix():
    return np.array([
        [0, 8, 0, 3],
        [0, 0, 2, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 0]
    ])


def test_build_tree_structure(tree, scope):
    root = build_tree_structure(tree, scope)
    node_ids = [root.get_id()]
    node_ids += [c.get_id() for c in root.get_children()]
    node_ids += [d.get_id() for c in root.get_children() for d in c.get_children()]
    node_ids += [root.get_children()[1].get_children()[0].get_children()[0].get_id()]
    got_tree, got_scope = root.get_tree_scope()
    assert node_ids == [5, 3, 7, 2, 0, 1, 4, 9]
    assert root.get_n_nodes() == 8
    assert set(got_scope) == set(scope)
    assert set(map(lambda x: (-1, x[1]) if x[0] == -1 else (got_scope[x[0]], x[1]), zip(got_tree, got_scope))) == \
           set(map(lambda x: (-1, x[1]) if x[0] == -1 else (scope[x[0]], x[1]), zip(tree, scope)))
    with pytest.raises(ValueError):
        build_tree_structure([0, 0, 1, 2, 3])
        build_tree_structure([-1, 0, -1, 2, 3])


def test_compute_bfs_ordering(tree):
    ordering = compute_bfs_ordering(tree)
    assert ordering == [1, 0, 4, 2, 3, 5, 6, 7]


def test_maximum_spanning_tree(adj_matrix):
    bfs, tree = maximum_spanning_tree(0, adj_matrix)
    assert bfs.tolist() == [0, 1, 3, 2]
    assert tree.tolist() == [-1, 0, 3, 1]
