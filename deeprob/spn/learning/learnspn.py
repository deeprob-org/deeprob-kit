# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from enum import Enum
from collections import deque
from typing import Optional, Union, Type, List, NamedTuple

import numpy as np
from tqdm import tqdm

from deeprob.utils.random import RandomState, check_random_state
from deeprob.spn.structure.leaf import Leaf
from deeprob.spn.structure.node import Node, Sum, Product, assign_ids
from deeprob.spn.learning.leaf import LearnLeafFunc, get_learn_leaf_method, learn_naive_factorization
from deeprob.spn.learning.splitting.rows import SplitRowsFunc, get_split_rows_method, split_rows_clusters
from deeprob.spn.learning.splitting.cols import SplitColsFunc, get_split_cols_method, split_cols_clusters


class OperationKind(Enum):
    """
    Operation kind used by LearnSPN algorithm.
    """
    REM_FEATURES = 1
    CREATE_LEAF = 2
    SPLIT_NAIVE = 3
    SPLIT_ROWS = 4
    SPLIT_COLS = 5


class Task(NamedTuple):
    """
    Recursive task information used by LearnSPN algorithm.
    """
    parent: Node
    data: np.ndarray
    scope: List[int]
    no_cols_split: bool = False
    no_rows_split: bool = False
    is_first: bool = False


def learn_spn(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    learn_leaf: Union[str, LearnLeafFunc] = 'mle',
    split_rows: Union[str, SplitRowsFunc] = 'kmeans',
    split_cols: Union[str, SplitColsFunc] = 'rdc',
    learn_leaf_kwargs: dict = None,
    split_rows_kwargs: dict = None,
    split_cols_kwargs: dict = None,
    min_rows_slice: int = 256,
    min_cols_slice: int = 2,
    random_state: Optional[RandomState] = None,
    verbose: bool = True
) -> Node:
    """
    Learn the structure and parameters of a SPN given some training data and several hyperparameters.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature). Each domain is either a list of values, for discrete
                    distributions, or a tuple (consisting of min value and max value), for continuous distributions.
    :param learn_leaf: The method to use to learn a distribution leaf node,
                       It can be either 'mle', 'isotonic', 'binary-clt' or a custom LearnLeafFunc.
    :param split_rows: The rows splitting method.
                       It can be either 'kmeans', 'gmm', 'rdc', 'random' or a custom SplitRowsFunc function.
    :param split_cols: The columns splitting method.
                       It can be either 'gvs', 'rgvs', 'wrgvs', 'ebvs', 'ebvs_ae', 'gbvs', 'gbvs_ag', 'rdc', 'random'
                       or a custom SplitColsFunc function.
    :param learn_leaf_kwargs: The parameters of the learn leaf method.
    :param split_rows_kwargs: The parameters of the rows splitting method.
    :param split_cols_kwargs: The parameters of the cols splitting method.
    :param min_rows_slice: The minimum number of samples required to split horizontally.
    :param min_cols_slice: The minimum number of features required to split vertically.
    :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
    :param verbose: Whether to enable verbose mode.
    :return: A learned valid SPN.
    :raises ValueError: If a parameter is out of scope.
    """
    if len(distributions) == 0:
        raise ValueError("The list of distribution classes must be non-empty")
    if len(domains) == 0:
        raise ValueError("The list of domains must be non-empty")
    if min_rows_slice <= 0:
        raise ValueError("The minimum number of samples required to split horizontally must be positive")
    if min_cols_slice <= 0:
        raise ValueError("The minimum number of samples required to split vertically must be positive")

    n_samples, n_features = data.shape
    if len(distributions) != n_features or len(domains) != n_features:
        raise ValueError("Each data column should correspond to a random variable having a distribution and a domain")

    # Setup the learn leaf, split rows and split cols functions
    learn_leaf_func = get_learn_leaf_method(learn_leaf) if isinstance(learn_leaf, str) else learn_leaf
    split_rows_func = get_split_rows_method(split_rows) if isinstance(split_rows, str) else split_rows
    split_cols_func = get_split_cols_method(split_cols) if isinstance(split_cols, str) else split_cols

    if learn_leaf_kwargs is None:
        learn_leaf_kwargs = dict()
    if split_rows_kwargs is None:
        split_rows_kwargs = dict()
    if split_cols_kwargs is None:
        split_cols_kwargs = dict()

    # Setup the initial scope as [0, # of features - 1]
    initial_scope = list(range(n_features))

    # Check the random state
    random_state = check_random_state(random_state)

    # Add the random state to learning leaf parameters
    learn_leaf_kwargs['random_state'] = random_state

    # Initialize the progress bar (with unspecified total), if verbose is enabled
    if verbose:
        tk = tqdm(
            total=np.inf, leave=None, unit='node',
            bar_format='{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]'
        )

    tasks = deque()
    tmp_node = Product(initial_scope)
    tasks.append(Task(tmp_node, data, initial_scope, is_first=True))

    while tasks:
        # Get the next task
        task = tasks.popleft()

        # Select the operation to apply
        n_samples, n_features = task.data.shape
        # Get the indices of uninformative features
        zero_var_idx = np.isclose(np.var(task.data, axis=0), 0.0)
        # If all the features are uninformative, then split using Naive Bayes model
        if np.all(zero_var_idx):
            op = OperationKind.SPLIT_NAIVE
        # If only some of the features are uninformative, then remove them
        elif np.any(zero_var_idx):
            op = OperationKind.REM_FEATURES
        # Create a leaf node if the data split dimension is small or last rows splitting failed
        elif task.no_rows_split or n_features < min_cols_slice or n_samples < min_rows_slice:
            op = OperationKind.CREATE_LEAF
        # Use rows splitting if previous columns splitting failed or it is the first task
        elif task.no_cols_split or task.is_first:
            op = OperationKind.SPLIT_ROWS
        # Defaults to columns splitting
        else:
            op = OperationKind.SPLIT_COLS

        if op == OperationKind.REM_FEATURES:
            node = Product(task.scope)

            # Model the removed features using Naive Bayes
            rem_scope = [task.scope[i] for i, in np.argwhere(zero_var_idx)]
            dists, doms = [distributions[s] for s in rem_scope], [domains[s] for s in rem_scope]
            naive = learn_naive_factorization(
                task.data[:, zero_var_idx], dists, doms, rem_scope,
                learn_leaf_func=learn_leaf_func, **learn_leaf_kwargs
            )
            node.children.append(naive)

            # Add the tasks regarding non-removed features
            is_first = task.is_first and len(tasks) == 0
            oth_scope = [task.scope[i] for i, in np.argwhere(~zero_var_idx)]
            tasks.append(Task(node, task.data[:, ~zero_var_idx], oth_scope, is_first=is_first))
            task.parent.children.append(node)
        elif op == OperationKind.CREATE_LEAF:
            # Create a leaf node
            dists, doms = [distributions[s] for s in task.scope], [domains[s] for s in task.scope]
            leaf = learn_leaf_func(task.data, dists, doms, task.scope, **learn_leaf_kwargs)
            task.parent.children.append(leaf)
        elif op == OperationKind.SPLIT_NAIVE:
            # Split the data using a naive factorization
            dists, doms = [distributions[s] for s in task.scope], [domains[s] for s in task.scope]
            node = learn_naive_factorization(
                task.data, dists, doms, task.scope,
                learn_leaf_func=learn_leaf_func, **learn_leaf_kwargs
            )
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_ROWS:
            # Split the data by rows (sum node)
            dists, doms = [distributions[s] for s in task.scope], [domains[s] for s in task.scope]
            clusters = split_rows_func(task.data, dists, doms, random_state, **split_rows_kwargs)
            slices, weights = split_rows_clusters(task.data, clusters)

            # Check whether only one partitioning is returned
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, no_cols_split=False, no_rows_split=True))
                continue

            # Add sub-tasks and append Sum node
            node = Sum(task.scope, weights=weights)
            for local_data in slices:
                tasks.append(Task(node, local_data, task.scope))
            task.parent.children.append(node)
        elif op == OperationKind.SPLIT_COLS:
            # Split the data by columns (product node)
            dists, doms = [distributions[s] for s in task.scope], [domains[s] for s in task.scope]
            clusters = split_cols_func(task.data, dists, doms, random_state, **split_cols_kwargs)
            slices, scopes = split_cols_clusters(task.data, clusters, task.scope)

            # Check whether only one partitioning is returned
            if len(slices) == 1:
                tasks.append(Task(task.parent, task.data, task.scope, no_cols_split=True, no_rows_split=False))
                continue

            # Add sub-tasks and append Product node
            node = Product(task.scope)
            for i, local_data in enumerate(slices):
                tasks.append(Task(node, local_data, scopes[i]))
            task.parent.children.append(node)
        else:
            raise NotImplementedError("Operation of kind {} not implemented".format(op))

        if verbose:
            tk.update()
            tk.refresh()

    if verbose:
        tk.close()

    root = tmp_node.children[0]
    return assign_ids(root)
