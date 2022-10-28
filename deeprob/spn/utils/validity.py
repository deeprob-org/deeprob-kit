# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, List, Tuple

from deeprob.context import is_check_spn_enabled
from deeprob.spn.structure.node import Node, Sum, Product
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.utils.filter import collect_nodes


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

    # Collect the nodes starting from the root node (cache)
    nodes = collect_nodes(root)

    # Check the SPN nodes are correctly labeled
    if labeled:
        result = is_labeled(root, nodes=nodes)
        if result is not None:
            raise ValueError(f"SPN is not correctly labeled: {result}")

    # Check the SPN is smooth
    if smooth:
        result = is_smooth(root, nodes=nodes)
        if result is not None:
            raise ValueError(f"SPN is not smooth: {result}")

    # Check the SPN is decomposable
    if decomposable:
        result = is_decomposable(root, nodes=nodes)
        if result is not None:
            raise ValueError(f"SPN is not decomposable: {result}")

    # Check the SPN is structured decomposable
    if structured_decomposable:
        result = is_structured_decomposable(root, nodes=nodes)
        if result is not None:
            raise ValueError(f"SPN is not structured decomposable: {result}")


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


def is_smooth(root: Node, nodes: Optional[List[Node]] = None) -> Optional[str]:
    """
    Check if the SPN is smooth (or complete).
    It checks that each child of a sum node has the same scope.

    :param root: The root of the SPN.
    :param nodes: The list of nodes. If None, it will be retrieved starting from the root node.
    :return: None if the SPN is smooth, a reason otherwise.
    """
    if nodes is None:
        nodes = collect_nodes(root)
    sum_nodes: List[Sum] = list(filter(lambda n: isinstance(n, Sum), nodes))

    for node in sum_nodes:
        if len(node.children) == 0:
            return f"Sum node #{node.id} has no children"
        if len(node.children) != len(node.weights):
            return f"Weights and children length mismatch in node #{node.id}"
        if any(map(lambda c: set(c.scope) != set(node.scope), node.children)):
            return f"Children of Sum node #{node.id} have different scopes"
    return None


def is_decomposable(root: Node, nodes: Optional[List[Node]] = None) -> Optional[str]:
    """
    Check if the SPN is decomposable (or consistent).
    It checks that each child of a product node has disjointed scopes.

    :param root: The root of the SPN.
    :param nodes: The list of nodes. If None, it will be retrieved starting from the root node.
    :return: None if the SPN is decomposable, a reason otherwise.
    """
    if nodes is None:
        nodes = collect_nodes(root)
    product_nodes: List[Product] = list(filter(lambda n: isinstance(n, Product), nodes))

    for node in product_nodes:
        if len(node.children) == 0:
            return f"Product node #{node.id} has no children"
        s_scope = set(sum([c.scope for c in node.children], []))
        if set(node.scope) != s_scope:
            return f"Children of Product node #{node.id} don't have disjointed scopes"
    return None


def is_structured_decomposable(root: Node, nodes: Optional[List[Node]] = None) -> Optional[str]:
    """
    Check if the PC is structured decomposable.
    It checks that product nodes follow a vtree.
    Note that if a PC is structured decomposable then it's also decomposable.

    :param root: The root of the PC.
    :param nodes: The list of nodes. If None, it will be retrieved starting from the root node.
    :return: None if the PC is structured decomposable, a reason otherwise.
    """
    # Shortcut: a PC is structured decomposable if it is compatible with itself
    if nodes is None:
        nodes = collect_nodes(root)
    return are_compatible(root, root, nodes_a=nodes, nodes_b=nodes)


def are_compatible(
    root_a: Node,
    root_b: Node,
    nodes_a: Optional[List[Node]] = None,
    nodes_b: Optional[List[Node]] = None
) -> Optional[str]:
    """
    Check if two PCs are compatible.

    :param root_a: The root of the first PC.
    :param root_b: The root of the second PC.
    :param nodes_a: The list of nodes of the first PC. If None, it will be retrieved starting from the root node.
    :param nodes_b: The list of nodes of the second PC. If None, it will be retrieved starting from the root node.
    :return: None if the two PCs are compatible, a reason otherwise.
    """
    if nodes_a is None:
        nodes_a = collect_nodes(root_a)
    if nodes_b is None:
        nodes_b = collect_nodes(root_b)

    # Check smoothness and decomposability first
    res = is_smooth(root_a, nodes_a)
    if res is not None:
        return f'First PC: {res}'
    res = is_decomposable(root_a, nodes_a)
    if res is not None:
        return f'First PC: {res}'
    res = is_smooth(root_b, nodes_b)
    if res is not None:
        return f'Second PC: {res}'
    res = is_decomposable(root_b, nodes_b)
    if res is not None:
        return f'Second PC: {res}'

    # Get scopes as sets
    scopes_a = collect_scopes(nodes_a)
    scopes_b = collect_scopes(nodes_b)
    scopes_a = list(map(lambda s: set(s), scopes_a))
    scopes_b = list(map(lambda s: set(s), scopes_b))

    # Quadratic in the number of product nodes
    for s1 in scopes_a:
        for s2 in scopes_b:
            int_len = len(s1.intersection(s2))
            if int_len != 0 and int_len != min(len(s1), len(s2)):
                return f"Incompatibility found between scope {s1} and scope {s2}"
    return None


def collect_scopes(nodes: List[Node]) -> List[Tuple[int]]:
    """
    Collect the scopes of each node.

    :param nodes: The list of nodes.
    :return: A list of scopes.
    """
    scopes = list()
    for n in nodes:
        if isinstance(n, Product):
            scopes.append(tuple(sorted(n.scope)))
        elif isinstance(n, BinaryCLT):
            scopes.extend([tuple(sorted(scope)) for scope in n.get_scopes()])
        elif not isinstance(n, Sum) and not isinstance(n, Leaf):
            raise NotImplementedError(f"Case not considered for {type(n)} nodes")
    return scopes
