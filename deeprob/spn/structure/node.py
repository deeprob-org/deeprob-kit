# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from __future__ import annotations
import abc
from typing import Optional, Union, List, Iterator
from collections import deque, defaultdict

import numpy as np
from scipy.special import logsumexp


class Node(abc.ABC):
    def __init__(self, scope: List[int], children: Optional[List[Node]] = None):
        """
        Initialize a SPN node given the children list and its scope.

        :param scope: The scope.
        :param children: A list of nodes. If None, children are initialized as an empty list.
        :raises ValueError: If the scope is empty.
        :raises ValueError: If the scope contains duplicates.
        """
        if not scope:
            raise ValueError("The scope must not be empty")
        if len(scope) != len(set(scope)):
            raise ValueError("The scope must not contain duplicates")
        if children is None:
            children = list()

        self.id = 0
        self.scope = scope
        self.children = children

    @abc.abstractmethod
    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting likelihoods.
        """

    @abc.abstractmethod
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the logarithmic likelihood of the node given some input.

        :param x: The inputs.
        :return: The resulting log-likelihoods.
        """


class Sum(Node):
    def __init__(
        self,
        scope: Optional[List[int]] = None,
        children: Optional[List[Node]] = None,
        weights: Optional[Union[List[float], np.ndarray]] = None,
    ):
        """
        Initialize a SPN sum node given a list of children and their weights and a scope.

        :param scope: The scope. If None, the scope is initialized based on children scopes.
        :param children: A list of nodes. If None, children are initialized as an empty list.
        :param weights: The weights associated to each children node. It can be None.
        :raises ValueError: If both scope and children are None.
        :raises ValueError: If children nodes have different scopes.
        :raises ValueError: If the length of weights and children are different.
        :raises ValueError: If weights don't sum up to 1.
        """
        if children is None:
            if scope is None:
                raise ValueError("Cannot infer Sum node's scope without children")
        else:
            if scope is None:
                scope = children[0].scope
            s_scope = set(scope)
            if any(map(lambda c: set(c.scope) != s_scope, children[1:])):
                raise ValueError("Children of Sum node have different scopes")
            if weights is not None and len(weights) != len(children):
                raise ValueError("Weights and children length mismatch")

        if weights is not None:
            if isinstance(weights, list):
                weights = np.array(weights, dtype=np.float32)
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("Weights don't sum up to 1")
        self.weights = weights

        super().__init__(scope, children)

    def em_init(self, random_state: np.random.RandomState):
        """
        Random initialize the node's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """
        weights = random_state.dirichlet(np.ones(len(self.children)))
        self.weights = weights.astype(np.float32)

    def em_step(self, stats: np.ndarray, step_size: float):
        """
        Compute a batch Expectation-Maximization (EM) step.

        :param stats: The sufficient statistics of each sample.
        :param step_size: The step size of update.
        """
        unnorm_weights = self.weights * np.sum(stats, axis=1) + np.finfo(np.float32).eps
        weights = unnorm_weights / np.sum(unnorm_weights)

        # Update the parameters
        self.weights = (1.0 - step_size) * self.weights + step_size * weights

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.dot(x, self.weights), axis=1)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        return logsumexp(x, b=self.weights, axis=1, keepdims=True)


class Product(Node):
    def __init__(
        self,
        scope: Optional[List[int]] = None,
        children: Optional[List[Node]] = None
    ):
        """
        Initialize a product node given a list of children and its scope.

        :param scope: The scope. If None, the scope is initialized based on children scopes.
        :param children: A list of nodes. If None, children are initialized as an empty list.
        :raises ValueError: If both scope and children are None.
        :raises ValueError: If children nodes don't have disjointed scopes.
        """
        if children is None:
            if scope is None:
                raise ValueError("Cannot infer Product node's scope without children")
        else:
            c_scope = list(sum([c.scope for c in children], []))
            s_scope = set(c_scope)
            if scope is None:
                if len(c_scope) != len(s_scope):
                    raise ValueError("Children of Product node don't have disjointed scopes")
                scope = c_scope
            elif set(scope) != s_scope:
                raise ValueError("Children of Product node don't have disjointed scopes")

        super().__init__(scope, children)

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.prod(x, axis=1, keepdims=True)

    def log_likelihood(self, x: np.append) -> np.ndarray:
        return np.sum(x, axis=1, keepdims=True)


def assign_ids(root: Node) -> Node:
    """
    Assign the ids to the nodes of a SPN.

    :param root: The root of the SPN.
    :return: The same SPN with each node having modified ids.
    :raises ValueError: If the SPN structure is not a DAG.
    """
    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    next_id = 0
    for node in nodes:
        node.id = next_id
        next_id += 1
    return root


def bfs(root: Node) -> Iterator[Node]:
    """
    Compute the Breadth First Search (BFS) ordering for a SPN.

    :param root: The root of the SPN.
    :return: The BFS nodes iterator.
    """
    seen, queue = {root}, deque([root])
    while queue:
        node = queue.popleft()
        yield node
        for c in node.children:
            if c not in seen:
                seen.add(c)
                queue.append(c)


def dfs_post_order(root: Node) -> Iterator[Node]:
    """
    Compute Depth First Search (DFS) Post-Order ordering for a SPN.

    :param root: The root of the SPN.
    :return: The DFS Post-Order nodes iterator.
    """
    seen, stack = {root}, [root]
    while stack:
        node = stack[-1]
        if set(node.children).issubset(seen):
            stack.pop()
            yield node
            continue
        for c in node.children:
            if c not in seen:
                seen.add(c)
                stack.append(c)


def topological_order(root: Node) -> Optional[List[Node]]:
    """
    Compute the Topological Ordering for a SPN, using the Kahn's Algorithm.

    :param root: The root of the SPN.
    :return: A list of nodes that form a topological ordering.
             If the SPN graph is not acyclic, it returns None.
    """
    ordering = list()
    num_outgoings = defaultdict(int)
    num_outgoings[root] = 0

    # Initialize the number of outgoings edges for each node
    for node in bfs(root):
        for c in node.children:
            num_outgoings[c] += 1

    # Check the unusual case where the root node have outgoings edges, i.e. a trivial cycle has been found
    if num_outgoings[root] != 0:
        return None

    # Non-layered topological ordering implementation
    queue = deque([root])
    while queue:
        node = queue.popleft()
        ordering.append(node)
        for c in node.children:
            num_outgoings[c] -= 1
            if num_outgoings[c] == 0:
                queue.append(c)

    # Check if a cycle has been found
    if sum(num_outgoings.values()) != 0:
        return None
    return ordering


def topological_order_layered(root: Node) -> Optional[List[List[Node]]]:
    """
    Compute the Topological Ordering Layered for a SPN, using the Kahn's Algorithm.

    :param root: The root of the SPN.
    :return: A list of layers that form a topological ordering.
             If the SPN graph is not acyclic, it returns None.
    """
    ordering = list()
    num_outgoings = defaultdict(int)
    num_outgoings[root] = 0

    # Initialize the number of outgoings edges for each node
    for node in bfs(root):
        for c in node.children:
            num_outgoings[c] += 1

    # Check the unusual case where the root node have outgoings edges, i.e. a trivial cycle has been found
    if num_outgoings[root] != 0:
        return None

    # Layered topological ordering implementation
    ordering.append([root])
    while True:
        layer = list()
        for node in ordering[-1]:
            for c in node.children:
                num_outgoings[c] -= 1
                if num_outgoings[c] == 0:
                    layer.append(c)
        if not layer:
            break
        ordering.append(layer)

    # Check if a cycle has been found
    if sum(num_outgoings.values()) != 0:
        return None
    return ordering
