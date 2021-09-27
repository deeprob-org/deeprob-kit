import numpy as np

from typing import Optional, Union, Tuple, Any, Callable

from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, topological_order
from deeprob.spn.utils.validity import check_spn


def eval_bottom_up(
    root: Node,
    x: np.ndarray,
    leaf_func: Callable[[Leaf, np.ndarray, Any], np.ndarray],
    node_func: Callable[[Node, np.ndarray, Any], np.ndarray],
    leaf_func_kwargs: Optional[dict] = None,
    node_func_kwargs: Optional[dict] = None,
    return_results: bool = False,
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
    :return: The outputs. Additionally it returns the output of each node.
    :raises ValueError: If a parameter is out of domain.
    """
    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if node_func_kwargs is None:
        node_func_kwargs = dict()

    n_samples, n_features = x.shape
    ls = np.empty(shape=(len(nodes), n_samples), dtype=np.float32)

    for node in reversed(nodes):
        if isinstance(node, Leaf):
            ls[node.id] = leaf_func(node, x[:, node.scope], **leaf_func_kwargs)
        else:
            children_ls = np.stack([ls[c.id] for c in node.children], axis=1)
            ls[node.id] = node_func(node, children_ls, **node_func_kwargs)

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
    inplace: bool = False
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
    :return: The NaN-filled inputs.
    :raises ValueError: If a parameter is out of domain.
    """
    # Check the SPN
    check_spn(root, labeled=True, smooth=True, decomposable=True)

    nodes = topological_order(root)
    if nodes is None:
        raise ValueError("SPN structure is not a directed acyclic graph (DAG)")

    if leaf_func_kwargs is None:
        leaf_func_kwargs = dict()
    if sum_func_kwargs is None:
        sum_func_kwargs = dict()

    if not inplace:
        x = np.copy(x)
    n_samples, n_features = x.shape

    # Build the array consisting of top-down path masks
    masks = np.zeros(shape=(len(nodes), n_samples), dtype=np.bool_)
    masks[root.id] = True

    for node in nodes:
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

    return x
