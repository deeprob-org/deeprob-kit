# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import List
from copy import deepcopy
from collections import defaultdict

import numpy as np

from deeprob.context import ContextState
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, topological_order, Sum, Product, assign_ids
from deeprob.spn.utils.validity import check_spn


def prune(root: Node, copy: bool = True) -> Node:
    """
    Prune (or simplify) the given SPN to a minimal and equivalent SPN.

    :param root: The root of the SPN.
    :param copy: Whether to copy the SPN before pruning it.
    :return: A minimal and equivalent SPN.
    :raises ValueError: If the SPN structure is not a directed acyclic graph (DAG).
    :raises ValueError: If an unknown node type is found.
    """
    # Copy the SPN before proceeding, if specified
    if copy:
        root = deepcopy(root)

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    # Build a dictionary that maps each id of a node to the corresponding node object
    nodes_map = dict(map(lambda n: (n.id, n), nodes))

    # Proceed by reversed topological order
    for node in reversed(nodes):
        # Skip leaves
        if isinstance(node, Leaf):
            continue

        # Retrieve the children nodes from the mapping
        children_nodes = list(map(lambda n: nodes_map[n.id], node.children))
        if len(children_nodes) == 1:
            nodes_map[node.id] = children_nodes[0]
        elif isinstance(node, Product):
            # Subsequent product nodes, concatenate the children of them
            children = list()
            for child in children_nodes:
                if not isinstance(child, Product):
                    children.append(child)
                    continue
                product_children = map(lambda n: nodes_map[n.id], child.children)
                children.extend(product_children)
            nodes_map[node.id].children = children
        elif isinstance(node, Sum):
            # Subsequent sum nodes, concatenate the children of them and adjust the weights accordingly
            # Important! This implementation take care also of directed acyclic graphs (DAGs)
            children_weights = defaultdict(float)
            for i, child in enumerate(children_nodes):
                if not isinstance(child, Sum):
                    children_weights[child] += node.weights[i]
                    continue
                sum_children = map(lambda n: nodes_map[n.id], child.children)
                for j, sum_child in enumerate(sum_children):
                    children_weights[sum_child] += node.weights[i] * child.weights[j]
            children, weights = zip(*children_weights.items())
            nodes_map[node.id].weights = np.array(weights, dtype=node.weights.dtype)
            nodes_map[node.id].children = children
        else:
            raise ValueError("Unknown node type called {}".format(node.__class__.__name__))

    return assign_ids(nodes_map[root.id])


def marginalize(root: Node, keep_scope: List[int], copy: bool = True) -> Node:
    """
    Marginalize some random variables of a SPN, obtaining the compilation of a marginal query.

    :param root: The root of the SPN to marginalize.
    :param keep_scope: The scope of the random variables to keep.
                       All the other random variables will be marginalized.
    :param copy: Whether to copy the SPN before marginalizing it.
    :return: A SPN in which an EVI query is equivalent to a MAR query under the given scope.
    :raises ValueError: If the scope of the random variables to keep is not valid.
    :raises ValueError: If the SPN structure is not a directed acyclic graph (DAG).
    :raises ValueError: If an unknown node type is found.
    :raises NotImplementedError: If non-BinaryCLT multivariate leaves are found.
    """
    if not keep_scope:
        raise ValueError("The scope of the random variables to keep must not be empty")
    keep_scope_s = set(keep_scope)
    if len(keep_scope) != len(keep_scope_s):
        raise ValueError("The scope of the random variables to keep must not contain duplicates")
    if not keep_scope_s.issubset(set(root.scope)):
        raise ValueError("The scope of the random variables to keep must be a subset of the scope of the SPN")

    # Copy the SPN before proceeding, if specified
    if copy:
        root = deepcopy(root)

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    # Build a dictionary that maps each id of a node to the corresponding node object
    nodes_map = dict(map(lambda n: (n.id, n), nodes))

    # Proceed by reversed topological order
    for node in reversed(nodes):
        if isinstance(node, Leaf):
            # Marginalize leaves, set to None if the leaf is fully marginalized
            if isinstance(node, BinaryCLT):
                # Convert the binary Chow-Liu Tree to a SPN and marginalize that instead
                clt_scope = list(keep_scope_s.intersection(node.scope))
                if clt_scope:
                    with ContextState(check_spn=False):  # Disable checking the SPN obtained by CLT to PC conversion
                        nodes_map[node.id] = marginalize(node.to_pc(), clt_scope, copy=False)
                else:
                    nodes_map[node.id] = None
            elif len(node.scope) == 1:
                nodes_map[node.id] = node if node.scope[0] in keep_scope else None
            else:
                raise NotImplementedError(
                    "Structural marginalization for arbitrarily multivariate leaves not yet implemented"
                )
            continue

        # Retrieve the children nodes from the mapping
        children_nodes = list(filter(
            lambda n: n is not None, map(lambda n: nodes_map[n.id], node.children)
        ))

        if not children_nodes:
            nodes_map[node.id] = None
        elif len(children_nodes) == 1:
            nodes_map[node.id] = children_nodes[0]
        else:
            if isinstance(node, Product):
                nodes_map[node.id].scope = list(sum(map(lambda n: n.scope, children_nodes), []))
                nodes_map[node.id].children = children_nodes
            elif isinstance(node, Sum):
                nodes_map[node.id].scope = children_nodes[0].scope
                nodes_map[node.id].children = children_nodes
            else:
                raise ValueError("Unknown node type called {}".format(node.__class__.__name__))

    root = assign_ids(nodes_map[root.id])
    return prune(root, copy=False)
