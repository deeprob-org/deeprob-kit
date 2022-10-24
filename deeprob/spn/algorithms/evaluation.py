# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Tuple, List, Any, Callable

import joblib
import numpy as np

from deeprob.spn.utils.validity import check_spn
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, topological_order, topological_order_layered


def parallel_layerwise_eval(
    layers: List[List[Node]],
    eval_func: Callable[[Node], None],
    reverse: bool = False,
    n_jobs: int = -1
):
    """
    Execute a function per node layerwise in parallel.

    :param layers: The layers, i.e., the layered topological ordering.
    :param eval_func: The evaluation function for a given node.
    :param reverse: Whether to reverse the layered topological ordering.
    :param n_jobs: The number of parallel jobs. It follows the joblib's convention.
    """
    if reverse:
        layers = reversed(layers)

    # Run parallel threads using joblib
    with joblib.parallel_backend('threading', n_jobs=n_jobs):
        with joblib.Parallel() as parallel:
            for layer in layers:
                parallel(joblib.delayed(eval_func)(node) for node in layer)


def eval_bottom_up(
    root: Node,
    x: np.ndarray,
    leaf_func: Callable[[Leaf, np.ndarray, Any], np.ndarray],
    node_func: Callable[[Node, np.ndarray, Any], np.ndarray],
    leaf_func_kwargs: Optional[dict] = None,
    node_func_kwargs: Optional[dict] = None,
    return_results: bool = False,
    n_jobs: int = 0
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Evaluate the SPN bottom up given some inputs and leaves and nodes evaluation functions.

    :param root: The root of the SPN.
    :param x: The inputs.
    :param leaf_func: The function to compute at each leaf node.
    :param node_func: The function to compute at each inner node.
    :param leaf_func_kwargs: The optional parameters of the leaf evaluation function.
    :param node_func_kwargs: The optional parameters of the inner nodes evaluation function.
    :param return_results: A flag indicating if this function must return the log likelihoods of each node of the SPN.
    :param n_jobs: The number of parallel jobs. It follows the joblib's convention. Set to 0 to disable.
    :return: The outputs. Additionally, it returns the output of each node.
    :raises ValueError: If a parameter is out of domain.
    """
    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if node_func_kwargs is None:
        node_func_kwargs = dict()

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    def eval_forward(n):
        if isinstance(n, Leaf):
            ls[n.id] = leaf_func(n, x[:, n.scope], **leaf_func_kwargs)
        else:
            children_ls = np.stack([ls[c.id] for c in n.children], axis=1)
            ls[n.id] = node_func(n, children_ls, **node_func_kwargs)

    if n_jobs == 0:
        # Compute the topological ordering
        ordering = topological_order(root)
        if ordering is None:
            raise ValueError("SPN structure is not a directed acyclic graph (DAG)")
        n_nodes, n_samples = len(ordering), len(x)
        ls = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)
        for node in reversed(ordering):
            eval_forward(node)
    else:
        # Compute the layered topological ordering
        layers = topological_order_layered(root)
        if layers is None:
            raise ValueError("SPN structure is not a directed acyclic graph (DAG)")
        n_nodes, n_samples = sum(map(len, layers)), len(x)
        ls = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)
        parallel_layerwise_eval(layers, eval_forward, reverse=True, n_jobs=n_jobs)

    if return_results:
        return ls[root.id], ls
    return ls[root.id]


def eval_top_down(
    root: Node,
    x: np.ndarray,
    lls: np.ndarray,
    leaf_func: Callable[[Leaf, np.ndarray, Any], np.ndarray],
    sum_func: Callable[[Sum, np.ndarray, Any], np.ndarray],
    leaf_func_kwargs: Optional[dict] = None,
    sum_func_kwargs: Optional[dict] = None,
    inplace: bool = False,
    n_jobs: int = 0
) -> np.ndarray:
    """
    Evaluate the SPN top down given some inputs, the likelihoods of each node and a leaves evaluation function.
    The leaves to evaluate are chosen by following the nodes given by the sum nodes evaluation function.

    :param root: The root of the SPN.
    :param x: The inputs with some NaN values.
    :param lls: The log-likelihoods of each node.
    :param leaf_func: The leaves evaluation function.
    :param sum_func: The sum nodes evaluation function.
    :param leaf_func_kwargs: The optional parameters of the leaf evaluation function.
    :param sum_func_kwargs: The optional parameters of the sum nodes evaluation function.
    :param inplace: Whether to make inplace assignments.
    :param n_jobs: The number of parallel jobs. It follows the joblib's convention. Set to 0 to disable.
    :return: The NaN-filled inputs.
    :raises ValueError: If a parameter is out of domain.
    """
    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if sum_func_kwargs is None:
        sum_func_kwargs = dict()

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    # Copy the input array, if not inplace mode
    if not inplace:
        x = np.copy(x)

    def eval_backward(n):
        if isinstance(n, Leaf):
            mask = np.ix_(masks[n.id], n.scope)
            x[mask] = leaf_func(n, x[mask], **leaf_func_kwargs)
        elif isinstance(n, Product):
            for c in n.children:
                masks[c.id] |= masks[n.id]
        elif isinstance(n, Sum):
            children_lls = np.stack([lls[c.id] for c in n.children], axis=1)
            branch = sum_func(n, children_lls, **sum_func_kwargs)
            for i, c in enumerate(n.children):
                masks[c.id] |= masks[n.id] & (branch == i)
        else:
            raise NotImplementedError(f"Top down evaluation not implemented for node of type {n.__class__.__name__}")

    if n_jobs == 0:
        # Compute the topological ordering
        ordering = topological_order(root)
        if ordering is None:
            raise ValueError("SPN structure is not a directed acyclic graph (DAG)")
        n_nodes, n_samples = len(ordering), len(x)

        # Build the array consisting of top-down path masks
        masks = np.zeros(shape=(n_nodes, n_samples), dtype=np.bool_)
        masks[root.id] = True
        for node in ordering:
            eval_backward(node)
    else:
        # Compute the layered topological ordering
        layers = topological_order_layered(root)
        if layers is None:
            raise ValueError("SPN structure is not a directed acyclic graph (DAG)")
        n_nodes, n_samples = sum(map(len, layers)), len(x)

        # Build the array consisting of top-down path masks
        masks = np.zeros(shape=(n_nodes, n_samples), dtype=np.bool_)
        masks[root.id] = True
        parallel_layerwise_eval(layers, eval_backward, reverse=False, n_jobs=n_jobs)

    return x
