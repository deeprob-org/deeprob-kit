from typing import List, Union
import numpy as np

from spn.structure.Base import Node, Sum, Product, rebuild_scopes_bottom_up, assign_ids
from spn.structure.leaves.cltree.CLTree import CLTree
from spn.structure.leaves.cltree.MLE import update_cltree_parameters_mle
from spn.structure.leaves.parametric.Parametric import Bernoulli

from deeprob.spn import structure as spn
from deeprob.spn.learning import learn_estimator
from deeprob.spn.structure import Gaussian
from deeprob.utils import build_tree_structure


def deeprob_learn_binary_clt(data: np.ndarray) -> spn.BinaryCLT:
    n_features = data.shape[1]
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope, root=0)
    clt.fit(data, domain, alpha=0.1, random_state=42)
    return clt


def deeprob_learn_binary_spn(data: np.ndarray) -> spn.Node:
    return deeprob_learn_binary_clt(data).to_pc()


def deeprob_learn_continuous_spn(data: np.ndarray) -> spn.Node:
    n_features = data.shape[1]
    distributions = [Gaussian] * n_features
    root = learn_estimator(data, distributions, method='learnspn')
    return root


def spflow_learn_binary_clt(data: np.ndarray) -> CLTree:
    n_features = data.shape[1]
    scope = list(range(n_features))
    clt = CLTree(scope, data)
    update_cltree_parameters_mle(clt, data, alpha=0.1)
    return clt


# This function has been readapted from deeprob-kit's BinaryCLT compilation to SPN
def spflow_learn_binary_spn(data: np.ndarray) -> Node:
    clt = deeprob_learn_binary_clt(data)
    root = build_tree_structure(clt.tree, scope=clt.scope)
    factors = {clt.scope[i]: np.exp(clt.params[i]) for i in range(len(clt.tree))}
    neg_buffer, pos_buffer = [], []
    nodes_stack = [root]
    last_node_visited = None
    while nodes_stack:
        node = nodes_stack[-1]
        if node.is_leaf() or (last_node_visited in node.get_children()):
            leaves: List[Union[Bernoulli, Sum]] = [
                Bernoulli(p=0.0, scope=node.get_id()),
                Bernoulli(p=1.0, scope=node.get_id()),
            ]
            if not node.is_leaf():
                neg_prod = Product(children=[leaves[0]] + neg_buffer[-len(node.get_children()):])
                pos_prod = Product(children=[leaves[1]] + pos_buffer[-len(node.get_children()):])
                del neg_buffer[-len(node.get_children()):]
                del pos_buffer[-len(node.get_children()):]
                sum_children = [neg_prod, pos_prod]
            else:
                sum_children = leaves
            weights = factors[node.get_id()]
            neg_buffer.append(Sum(children=sum_children, weights=weights[0]))
            pos_buffer.append(Sum(children=sum_children, weights=weights[1]))
            last_node_visited = nodes_stack.pop()
        else:
            nodes_stack.extend(node.get_children())
    return rebuild_scopes_bottom_up(assign_ids(pos_buffer[0]))


def spflow_learn_continuous_spn(data: np.ndarray) -> spn.Node:
    n_features = data.shape[1]
    distributions = [Gaussian] * n_features
    root = learn_estimator(data, distributions, method='learnspn')
    return root
