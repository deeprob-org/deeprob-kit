# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, bfs
from deeprob.spn.utils.filter import collect_nodes, filter_nodes_by_type


def compute_statistics(root: Node) -> dict:
    """
    Compute some statistics of a SPN given its root.
    The computed statistics are the following:

    - n_nodes, the number of nodes
    - n_sum, the number of sum nodes
    - n_prod, the number of product nodes
    - n_leaves, the number of leaves
    - n_edges, the number of edges
    - n_params, the number of parameters
    - depth, the depth of the network

    :param root: The root of the SPN.
    :return: A dictionary containing the statistics.
    """
    stats = {
        'n_nodes': len(collect_nodes(root)),
        'n_sum': len(filter_nodes_by_type(root, Sum)),
        'n_prod': len(filter_nodes_by_type(root, Product)),
        'n_leaves': len(filter_nodes_by_type(root, Leaf)),
        'n_edges': compute_edges_count(root),
        'n_params': compute_parameters_count(root),
        'depth': compute_depth(root)
    }
    return stats


def compute_edges_count(root: Node) -> int:
    """
    Get the number of edges of a SPN given its root.

    :param root: The root of the SPN.
    :return: The number of edges.
    """
    return sum(len(n.children) for n in filter_nodes_by_type(root, (Sum, Product)))


def compute_parameters_count(root: Node) -> int:
    """
    Get the number of parameters of a SPN given its root.

    :param root:  The root of the SPN.
    :return: The number of parameters.
    """
    n_weights = sum(len(n.weights) for n in filter_nodes_by_type(root, Sum))
    n_leaf_params = sum(n.params_count() for n in filter_nodes_by_type(root, Leaf))
    return n_weights + n_leaf_params


def compute_depth(root: Node) -> int:
    """
    Get the depth of the SPN given its root.

    :param root: The root of the SPN.
    :return: The depth of the network.
    """
    depths = dict()
    for node in bfs(root):
        d = depths.setdefault(node, 0)
        for c in node.children:
            depths[c] = d + 1
    return max(depths.values())
