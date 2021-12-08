# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Tuple, Any, Callable

import joblib
import numpy as np

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, topological_order_layered
from deeprob.spn.utils.validity import check_spn


def eval_bottom_up(
    root: Node,
    x: np.ndarray,
    leaf_func: Callable[[Leaf, np.ndarray, Any], np.ndarray],
    node_func: Callable[[Node, np.ndarray, Any], np.ndarray],
    leaf_func_kwargs: Optional[dict] = None,
    node_func_kwargs: Optional[dict] = None,
    return_results: bool = False,
    n_jobs: int = -1
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
    :param n_jobs: The number of parallel jobs. It follows the joblib's convention.
    :return: The outputs. Additionally it returns the output of each node.
    :raises ValueError: If a parameter is out of domain.
    """
    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if node_func_kwargs is None:
        node_func_kwargs = dict()

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    # Compute the layered topological ordering
    layers = topological_order_layered(root)
    if layers is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    n_nodes, n_samples = sum(map(len, layers)), len(x)
    ls = np.empty(shape=(n_nodes, n_samples), dtype=np.float32)

    def eval_forward(node):
        if isinstance(node, Leaf):
            ls[node.id] = leaf_func(node, x[:, node.scope], **leaf_func_kwargs)
        else:
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            ls[node.id] = node_func(node, children_ls, **node_func_kwargs)

    with joblib.parallel_backend('threading', n_jobs=n_jobs):
        with joblib.Parallel() as parallel:
            for layer in reversed(layers):
                parallel(joblib.delayed(eval_forward)(node) for node in layer)

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
    n_jobs: int = -1
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
    :param n_jobs: The number of parallel jobs. It follows the joblib's convention.
    :return: The NaN-filled inputs.
    :raises ValueError: If a parameter is out of domain.
    """
    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if sum_func_kwargs is None:
        sum_func_kwargs = dict()

    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    # Compute the layered topological ordering
    layers = topological_order_layered(root)
    if layers is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    # Copy the input array, if not inplace mode
    if not inplace:
        x = np.copy(x)

    # Build the array consisting of top-down path masks
    n_nodes, n_samples = sum(map(len, layers)), len(x)
    masks = np.zeros(shape=(n_nodes, n_samples), dtype=np.bool_)
    masks[root.id] = True

    def eval_backward(node):
        if isinstance(node, Leaf):
            mask = np.ix_(masks[node.id], node.scope)
            x[mask] = leaf_func(node, x[mask], **leaf_func_kwargs)
        elif isinstance(node, Product):
            for c in node.children:
                masks[c.id] |= masks[node.id]
        elif isinstance(node, Sum):
            children_lls = np.stack([lls[c.id] for c in node.children], axis=1)
            branch = sum_func(node, children_lls, **sum_func_kwargs)
            for i, c in enumerate(node.children):
                masks[c.id] |= masks[node.id] & (branch == i)
        else:
            raise NotImplementedError(
                "Top down evaluation not implemented for node of type {}".format(node.__class__.__name__)
            )

    with joblib.parallel_backend('threading', n_jobs=n_jobs):
        with joblib.Parallel() as parallel:
            for layer in layers:
                parallel(joblib.delayed(eval_backward)(node) for node in layer)

    return x
