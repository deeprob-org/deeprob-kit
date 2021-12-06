# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Union, List, Tuple, Type

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, bfs


def collect_nodes(root: Node) -> List[Node]:
    """
    Get all the nodes in a SPN.

    :param root: The root of the SPN.
    :return: A list of nodes.
    """
    return filter_nodes_by_type(root)


def filter_nodes_by_type(
    root: Node,
    ntype: Union[Type[Node], Tuple[Type[Node], ...]] = Node
) -> List[Union[Node, Leaf, Sum, Product]]:
    """
    Get the nodes of some specified types in a SPN.

    :param root: The root of the SPN.
    :param ntype: The node type. Multiple node types can be specified as a tuple.
    :return: A list of nodes of some specific types.
    """
    return list(filter(lambda n: isinstance(n, ntype), bfs(root)))
