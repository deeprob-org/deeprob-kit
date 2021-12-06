# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, List

import numpy as np

from deeprob.context import is_check_spn_enabled
from deeprob.spn.structure.node import Node, Sum, Product
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.utils.filter import collect_nodes, filter_nodes_by_type


def check_spn(
    root: Node,
    labeled: bool = True,
    smooth: bool = False,
    decomposable: bool = False,
    structured_decomposable: bool = False
):
    """
    Check a SPN have certain properties. Defaults to checking only 'labeled'.
    This function combines several checks over a SPN, hence reducing the computational effort
    used to retrieve the nodes from the SPN.

    :param root: The root node of the SPN.
    :param labeled: Whether to check if the SPN is correctly labeled.
    :param smooth: Whether to check if the SPN is smooth.
    :param decomposable: Whether to check if the SPN is decomposable.
    :param structured_decomposable: Whether to check if the SPN is structured decomposable.
    :raises ValueError: If the SPN doesn't have a certain property.
    """
    if not is_check_spn_enabled():  # Skip the checks entirely, if specified
        return

    # Collect the nodes starting from the root node
    nodes = collect_nodes(root)

    # Check the SPN nodes are correctly labeled
    if labeled:
        result = is_labeled(root, nodes=nodes)
        if result is not None:
            raise ValueError("SPN is not correctly labeled: {}".format(result))

    # Check the SPN is smooth
    if smooth:
        sum_nodes: List[Sum] = list(filter(lambda n: isinstance(n, Sum), nodes))
        result = is_smooth(root, sum_nodes=sum_nodes)
        if result is not None:
            raise ValueError("SPN is not smooth: {}".format(result))

    # Check the SPN is decomposable
    if decomposable:
        product_nodes: List[Product] = list(filter(lambda n: isinstance(n, Product), nodes))
        result = is_decomposable(root, product_nodes=product_nodes)
        if result is not None:
            raise ValueError("SPN is not decomposable: {}".format(result))

    # Check the SPN is structured decomposable
    if structured_decomposable:
        result = is_structured_decomposable(root, nodes=nodes)
        if result is not None:
            raise ValueError("SPN is not structured decomposable: {}".format(result))


def is_smooth(root: Node, sum_nodes: Optional[List[Sum]] = None) -> Optional[str]:
    """
    Check if the SPN is smooth (or complete).
    It checks that each child of a sum node has the same scope.
    Furthermore, it checks that the sum of the weights of a sum node is close to 1.

    :param root: The root of the SPN.
    :param sum_nodes: The list of sum nodes. If None, it will be retrieved starting from the root node.
    :return: None if the SPN is smooth, a reason otherwise.
    """
    if sum_nodes is None:
        sum_nodes = filter_nodes_by_type(root, Sum)

    for node in sum_nodes:
        if not np.isclose(np.sum(node.weights), 1.0):
            return "Weights of node #{} don't sum up to 1".format(node.id)
        if len(node.children) == 0:
            return "Sum node #{} has no children".format(node.id)
        if len(node.children) != len(node.weights):
            return "Weights and children length mismatch in node #{}".format(node.id)
        if any(map(lambda c: set(c.scope) != set(node.scope), node.children)):
            return "Children of Sum node #{} have different scopes".format(node.id)
    return None


def is_decomposable(root: Node, product_nodes: Optional[List[Product]] = None) -> Optional[str]:
    """
    Check if the SPN is decomposable (or consistent).
    It checks that each child of a product node has disjointed scopes.

    :param root: The root of the SPN.
    :param product_nodes: The list of product nodes. If None, it will be retrieved starting from the root node.
    :return: None if the SPN is decomposable, a reason otherwise.
    """
    if product_nodes is None:
        product_nodes = filter_nodes_by_type(root, Product)

    for node in product_nodes:
        if len(node.children) == 0:
            return "Product node #{} has no children".format(node.id)
        s_scope = set(sum([c.scope for c in node.children], []))
        if set(node.scope) != s_scope:
            return "Children of Product node #{} don't have disjointed scopes".format(node.id)
    return None


def is_structured_decomposable(root: Node, nodes: Optional[List[Node]] = None, verbose: bool = False) -> Optional[str]:
    """
    Check if the PC is structured decomposable.
    It checks that product nodes follow a vtree.
    Note that if a PC is structured decomposable then it's also decomposable / consistent.

    :param root: The root of the PC.
    :param nodes: The list of nodes. If None, it will be retrieved starting from the root node.
    :param verbose: if True, it prints the product nodes scopes in a relevant order.
    :return: None if the PC is structured decomposable, a reason otherwise.
    """
    if nodes is None:
        nodes = collect_nodes(root)

    s_scope = set()
    for n in nodes:
        if isinstance(n, Product):
            s_scope.add(tuple(sorted(n.scope)))
        elif isinstance(n, BinaryCLT):
            s_scope.update([tuple(sorted(scope)) for scope in n.get_scopes()])
        elif not isinstance(n, Sum) and not isinstance(n, Leaf):
            raise Exception("Case not yet considered for {} nodes".format(type(n)))
    scopes = [set(t) for t in list(s_scope)]

    # Ordering scopes is not needed, but useful for printing when verbose = True
    if verbose:
        scopes.sort(key=len)
        for scope in scopes:
            print(scope)

    # Quadratic in the number of product nodes, but at least does not require a vtree structure
    for s1 in scopes:
        for s2 in scopes:
            int_len = len(s1.intersection(s2))
            if int_len != 0 and int_len != min(len(s1), len(s2)):
                return "Intersection between scope {} and scope {}".format(s1, s2)
    return None


def is_labeled(root: Node, nodes: Optional[List[Node]] = None) -> Optional[str]:
    """
    Check if the SPN is labeled correctly.
    It checks that the initial id is zero and each id is consecutive.

    :param root: The root of the SPN.
    :param nodes: The list of nodes. If None, it will be retrieved starting from the root node.
    :return: None if the SPN is labeled correctly, a reason otherwise.
    """
    if nodes is None:
        nodes = collect_nodes(root)

    ids = set(map(lambda n: n.id, nodes))
    if None in ids:
        return "Some nodes have missing ids"
    if len(ids) != len(nodes):
        return "Some nodes have repeated ids"
    if min(ids) != 0:
        return "Node ids are not starting at 0"
    if max(ids) != len(ids) - 1:
        return "Node ids are not consecutive"
    return None
