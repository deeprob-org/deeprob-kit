# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, topological_order
from deeprob.spn.utils.validity import check_spn


def eval_backward(root: Node, lls: np.ndarray) -> np.ndarray:
    """
    Compute the log-gradients at each SPN node.

    :param root: The root of the SPN.
    :param lls: The log-likelihoods at each node.
    :return: The log-gradients w.r.t. the nodes.
    :raises ValueError: If a parameter is out of domain.
    """
    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    n_nodes, n_samples = lls.shape
    if n_nodes != len(nodes):
        raise ValueError("Incompatible log-likelihoods broadcasting at each node")

    # Initialize the log-gradients array and the cached log-gradients dictionary of lists
    grads = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)
    cached_grads = defaultdict(list)

    # Initialize the identity log-gradient at root node
    grads[root.id] = 0.0

    for node in nodes:
        # Compute log-gradient at the underlying node by logsumexp
        # Note that at this point of topological ordering, the node have no incoming arcs
        # Hence, we can finally compute the log-gradients w.r.t. this node
        if node.id != root.id:
            grads[node.id] = logsumexp(cached_grads[node.id], axis=0)
            del cached_grads[node.id]  # Cached log-gradients no longer necessary

        if isinstance(node, Sum):
            for c, w in zip(node.children, node.weights):
                g = grads[node.id] + np.log(w)
                cached_grads[c.id].append(g)
        elif isinstance(node, Product):
            for c in node.children:
                g = grads[node.id] + lls[node.id] - lls[c.id]
                cached_grads[c.id].append(g)
        elif isinstance(node, Leaf):
            pass  # Leaves have no children
        else:
            raise NotImplementedError(
                "Gradient evaluation not implemented for node of type {}".format(node.__class__.__name__)
            )

    return grads
