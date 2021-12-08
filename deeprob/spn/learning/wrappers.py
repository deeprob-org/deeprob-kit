# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Type, List

import numpy as np
from tqdm import tqdm

from deeprob.spn.structure.leaf import LeafType, Leaf
from deeprob.spn.structure.node import Node, Sum, assign_ids
from deeprob.spn.learning.learnspn import learn_spn
from deeprob.spn.learning.xpc import learn_xpc, learn_expc
from deeprob.spn.algorithms.structure import prune


def learn_estimator(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: Optional[List[Union[list, tuple]]] = None,
    method: str = 'learnspn',
    **kwargs
) -> Node:
    """
    Learn a SPN density estimator given some training data, the features distributions and domains.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature). Each domain is either a list of values, for discrete
                    distributions, or a tuple (consisting of min value and max value), for continuous distributions.
                    If None, domains are determined automatically.
    :param method: The method used for structure learning. It can be either 'learnspn', 'xpc' or 'ensemble-xpc'.
    :param kwargs: Additional parameters for structure learning.
    :return: A learned valid and optimized SPN.
    :raises ValueError: If the method used for structure learning is not known.
    :raises ValueError: If the method is 'xpc' or 'ensemble-xpc' but the variable domains are not binary.
    """
    if domains is None:
        domains = compute_data_domains(data, distributions)

    if method == 'learnspn':
        root = learn_spn(data, distributions, domains, **kwargs)
        return prune(root, copy=False)
    if method == 'xpc':
        if not all(d == [0, 1] for d in domains):
            raise ValueError("The domains must be binary for learning a XPC")
        root, _ = learn_xpc(data, **kwargs)
        return root
    if method == 'ensemble-xpc':
        if not all(d == [0, 1] for d in domains):
            raise ValueError("The domains must be binary for learning an Ensemble-XPC")
        root, _ = learn_expc(data, **kwargs)
        return root
    raise ValueError("Unknown SPN learning method called {}".format(method))


def learn_classifier(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: Optional[List[Union[list, tuple]]] = None,
    class_idx: int = -1,
    verbose: bool = True,
    **kwargs
) -> Node:
    """
    Learn a SPN classifier given some training data, the features distributions and domains and
    the class index in the training data.

    :param data: The training data.
    :param distributions: A list of distribution classes (one for each feature).
    :param domains: A list of domains (one for each feature). Each domain is either a list of values, for discrete
                    distributions, or a tuple (consisting of min value and max value), for continuous distributions.
                    If None, domains are determined automatically.
    :param class_idx: The index of the class feature in the training data.
    :param verbose: Whether to enable verbose mode.
    :param kwargs: Other parameters for structure learning.
    :return: A learned valid and optimized SPN.
    """
    if domains is None:
        domains = compute_data_domains(data, distributions)

    n_samples, _ = data.shape
    classes = data[:, class_idx]

    # Initialize the tqdm wrapped unique classes array, if verbose is enabled
    unique_classes = np.unique(classes)
    if verbose:
        unique_classes = tqdm(unique_classes, bar_format='{l_bar}{bar:24}{r_bar}', unit='class')

    # Learn each sub-spn's structure individually
    weights = []
    children = []
    for c in unique_classes:
        local_data = data[classes == c]
        weight = len(local_data) / n_samples
        branch = learn_spn(local_data, distributions, domains, verbose=verbose, **kwargs)
        weights.append(weight)
        children.append(prune(branch, copy=False))

    root = Sum(children=children, weights=weights)
    return assign_ids(root)


def compute_data_domains(data: np.ndarray, distributions: List[Type[Leaf]]) -> List[Union[list, tuple]]:
    """
    Compute the domains based on the training data and the features distributions.

    :param data: The training data.
    :param distributions: A list of distribution classes.
    :return: A list of domains. Each domain is either a list of values, for discrete distributions, or
             a tuple (consisting of min value and max value), for continuous distributions.
    :raises ValueError: If an unknown distribution type is found.
    """
    domains = []
    for i, d in enumerate(distributions):
        col = data[:, i]
        if d.LEAF_TYPE == LeafType.DISCRETE:
            vals = np.unique(col).tolist()
            domains.append(vals)
        elif d.LEAF_TYPE == LeafType.CONTINUOUS:
            vmin = np.min(col).item()
            vmax = np.max(col).item()
            domains.append((vmin, vmax))
        else:
            raise ValueError("Unknown distribution type {}".format(d.LEAF_TYPE))
    return domains
