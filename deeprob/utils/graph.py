# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from __future__ import annotations
from typing import Optional, Union, List, Tuple
from collections import deque

import numpy as np
from scipy import sparse as sp


class TreeNode:
    """A simple class to model a node of a tree."""
    def __init__(self, node_id: int, parent: TreeNode = None):
        """
        Initialize a binary CLT.

        :param node_id: The ID of the node.
        :param parent: The parent node.
        """
        self.id = node_id
        self.__parent = None
        self.__children = []
        self.set_parent(parent)

    def get_id(self) -> int:
        """
        Get the ID of the node.

        :return: The ID of the node.
        """
        return self.id

    def get_parent(self) -> TreeNode:
        """
        Get the parent node.

        :return: The parent node, None if the node has no parent.
        """
        return self.__parent

    def get_children(self) -> List[TreeNode]:
        """
        Get the children list of the node.

        :return: The children list of the node.
        """
        return self.__children

    def set_parent(self, parent: TreeNode):
        """
        Set the parent node and update its children list.

        :param parent: The parent node.
        """
        if self.__parent is None and parent is not None:
            self.__parent = parent
            self.__parent.get_children().append(self)

    def is_leaf(self) -> bool:
        """
        Check whether the node is leaf.

        :return: True if the node is leaf, False otherwise.
        """
        return len(self.__children) == 0

    def get_n_nodes(self) -> int:
        """
        Get the number of the nodes of the tree rooted at self.

        :return: The number of nodes of the tree rooted at self.
        """
        n_nodes = 0
        queue = [self]
        while queue:
            current_node = queue.pop(0)
            queue.extend(current_node.get_children())
            n_nodes += 1
        return n_nodes

    def get_tree_scope(self) -> Tuple[list, list]:
        """
        Return the list of predecessors and the related scope of the tree rooted at self.
        Note that tree[root] must be -1, as it doesn't have a predecessor.

        :return tree: List of predecessors.
        :return scope: The related scope list.
        """
        tree = []
        scope = []
        queue = [self]
        while queue:
            current_node = queue.pop(0)
            queue.extend(current_node.get_children())
            scope.append(current_node.id)
            tree.append(current_node.get_parent().id if current_node.get_parent() is not None else -1)
        tree[scope.index(self.id)] = -1
        tree = [scope.index(t) if t != -1 else -1 for t in tree]
        return tree, scope


def build_tree_structure(tree: Union[List[int], np.ndarray], scope: Optional[List[int]] = None) -> TreeNode:
    """
    Build a Tree node recursive data structure given a tree structure encoded as a list of predecessors.
    Note that tree[root] must be -1, as it doesn't have a predecessor.
    Optionally, a scope can be used to specify the tree node ids.

    :param tree: The tree structure, as a sequence of predecessors.
    :param scope: An optional scope, as a list of ids.
    :return: The Tree node structure's root.
    :raises ValueError: If the tree structure is not compatible with the root node.
    :raises ValueError: If the scope contains duplicates.
    :raises ValueError: If the scope is incompatible with the tree structure.
    """
    if isinstance(tree, np.ndarray):
        tree = tree.tolist()
    if tree.count(-1) != 1:
        raise ValueError("Invalid tree structure")
    root_idx = tree.index(-1)

    if scope is None:
        root_id = root_idx
        nodes = [TreeNode(node_id) for node_id in range(len(tree))]
        for node_id, parent_id in enumerate(tree):
            if parent_id != -1:
                nodes[node_id].set_parent(nodes[parent_id])
    else:
        if len(set(scope)) != len(scope):
            raise ValueError("The scope must not contain duplicates")
        if len(scope) != len(tree):
            raise ValueError("Invalid scope's number of variables")

        root_id = scope[root_idx]
        nodes = {node_id: TreeNode(node_id) for node_id in scope}
        for node_idx, parent_idx in enumerate(tree):
            if parent_idx != -1:
                node_id = scope[node_idx]
                parent_id = scope[parent_idx]
                nodes[node_id].set_parent(nodes[parent_id])

    return nodes[root_id]


def compute_bfs_ordering(tree: Union[List[int], np.ndarray]) -> Union[List[int], np.ndarray]:
    """
    Compute the breadth-first-search variable ordering given a tree structure.
    Note that tree[root] must be -1, as it doesn't have a predecessor.

    :param tree: The tree structure, as a sequence of predecessors.
    :return: The BFS variable ordering as a Numpy array.
    """
    # Build the tree structure first
    root = build_tree_structure(tree)

    # Pre-Order exploration
    ordering = list()
    nodes_queue = deque([root])
    while nodes_queue:
        node = nodes_queue.popleft()
        ordering.append(node.get_id())
        if not node.is_leaf():
            nodes_queue.extend(node.get_children())

    if isinstance(tree, list):
        return ordering
    return np.array(ordering, dtype=tree.dtype)


def maximum_spanning_tree(root: int, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the maximum spanning tree of a graph starting from a given root node.

    :param root: The root node index.
    :param adj_matrix: The graph's adjacency matrix.
    :return: The breadth first traversal ordering and the maximum spanning tree.
             The maximum spanning tree is given as a list of predecessors.
    """
    # Compute the maximum spanning tree of an adjacency matrix
    # Note adding one to the adjacency matrix, because the graph must be fully connected
    mst = sp.csgraph.minimum_spanning_tree(-(adj_matrix + 1.0), overwrite=True)
    bfs, tree = sp.csgraph.breadth_first_order(
        mst, directed=False, i_start=root, return_predecessors=True
    )
    tree[root] = -1
    return bfs, tree
