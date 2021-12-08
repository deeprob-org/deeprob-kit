# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Type, List, Callable, Any

import numpy as np

from deeprob.utils.random import RandomState
from deeprob.spn.structure.node import Node, Product
from deeprob.spn.structure.leaf import LeafType, Leaf, Bernoulli, Isotonic
from deeprob.spn.structure.cltree import BinaryCLT

#: A signature for a learn SPN leaf function.
LearnLeafFunc = Callable[
    [np.ndarray,                # The data
     List[Type[Leaf]],          # The distributions
     List[Union[list, tuple]],  # The domains
     List[int],                 # The scope
     Any],                      # Other arguments
    Node                        # A SPN node
]


def get_learn_leaf_method(learn_leaf: str) -> LearnLeafFunc:
    """
    Get the learn leaf method.

    :param learn_leaf: The learn leaf method string to use.
    :return: A learn leaf function.
    :raises ValueError: If the leaf learning method is unknown.
    """
    if learn_leaf == 'mle':
        return learn_mle
    if learn_leaf == 'isotonic':
        return learn_isotonic
    if learn_leaf == 'binary-clt':
        return learn_binary_clt
    raise ValueError("Unknown learn leaf method called {}".format(learn_leaf))


def learn_mle(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    scope: List[int],
    alpha: float = 0.1,
    random_state: Optional[RandomState] = None
) -> Node:
    """
    Learn a leaf using Maximum Likelihood Estimate (MLE).
    If the data is multivariate, a naive factorized model is learned.

    :param data: The data, where each column correspond to a random variable.
    :param distributions: The distributions of the random variables.
    :param domains: The domains of the random variables.
    :param scope: The scope of the leaf.
    :param alpha: Laplace smoothing factor.
    :param random_state: The random state. It can be None.
    :return: A leaf distribution.
    :raises ValueError: If there are inconsistencies between the data, distributions and domains.
    """
    if len(scope) != len(distributions) or len(domains) != len(distributions):
        raise ValueError("Each data column should correspond to a random variable having a distribution and a domain")

    if len(scope) == 1:
        sc, dist, dom = scope[0], distributions[0], domains[0]
        leaf = dist(sc)
        leaf.fit(data, dom, alpha=alpha)
        return leaf

    return learn_naive_factorization(
        data, distributions, domains, scope, learn_mle,
        alpha=alpha, random_state=random_state
    )


def learn_isotonic(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    scope: List[int],
    alpha: float = 0.1,
    random_state: Optional[RandomState] = None
) -> Node:
    """
    Learn a leaf using Isotonic method.
    If the data is multivariate, a naive factorized model is learned.

    :param data: The data.
    :param distributions: The distribution of the random variables.
    :param domains: The domain of the random variables.
    :param scope: The scope of the leaf.
    :param alpha: Laplace smoothing factor.
    :param random_state: The random sate. It can be None.
    :return: A leaf distribution.
    :raises ValueError: If there are inconsistencies between the data, distributions and domains.
    """
    if len(scope) != len(distributions) or len(domains) != len(distributions):
        raise ValueError("Each data column should correspond to a random variable having a distribution and a domain")

    if len(scope) == 1:
        sc, dist, dom = scope[0], distributions[0], domains[0]
        leaf = Isotonic(sc) if dist.LEAF_TYPE == LeafType.CONTINUOUS else dist(sc)
        leaf.fit(data, dom, alpha=alpha)
        return leaf

    return learn_naive_factorization(
        data, distributions, domains, scope, learn_isotonic,
        alpha=alpha, random_state=random_state
    )


def learn_binary_clt(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    scope: List[int],
    to_pc: bool = False,
    alpha: float = 0.1,
    random_state: Optional[RandomState] = None
) -> Node:
    """
    Learn a leaf using a Binary Chow-Liu Tree (CLT).
    If the data is univariate, a Maximum Likelihood Estimate (MLE) leaf is returned.

    :param data: The data.
    :param distributions: The distributions of the random variables.
    :param domains: The domains of the random variables.
    :param scope: The scope of the leaf.
    :param to_pc: Whether to convert the CLT into an equivalent PC.
    :param alpha: Laplace smoothing factor.
    :param random_state: The random state. It can be None.
    :return: A leaf distribution.
    :raises ValueError: If there are inconsistencies between the data, distributions and domains.
    :raises ValueError: If the data doesn't follow a Bernoulli distribution.
    """
    if len(scope) != len(distributions) or len(domains) != len(distributions):
        raise ValueError("Each data column should correspond to a random variable having a distribution and a domain")
    if any(d != Bernoulli for d in distributions):
        raise ValueError("Binary Chow-Liu trees are only available for Bernoulli data")

    # If univariate, learn using MLE instead
    if len(scope) == 1:
        return learn_mle(
            data, distributions, domains, scope,
            alpha=alpha, random_state=random_state
        )

    # If multivariate, learn a binary CLTree
    leaf = BinaryCLT(scope)
    leaf.fit(data, domains, alpha=alpha, random_state=random_state)

    # Make the conversion to a probabilistic circuit, if specified
    if to_pc:
        return leaf.to_pc()
    return leaf


def learn_naive_factorization(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    scope: List[int],
    learn_leaf_func: LearnLeafFunc,
    **learn_leaf_kwargs
) -> Node:
    """
    Learn a leaf as a naive factorized model.

    :param data: The data.
    :param distributions: The distribution of the random variables.
    :param domains: The domain of the random variables.
    :param scope: The scope of the leaf.
    :param learn_leaf_func: The function to use to learn the sub-distributions parameters.
    :param learn_leaf_kwargs: Additional parameters for learn_leaf_func.
    :return: A naive factorized model.
    :raises ValueError: If there are inconsistencies between the data, distributions and domains.
    """
    if len(scope) != len(distributions) or len(domains) != len(distributions):
        raise ValueError("Each data column should correspond to a random variable having a distribution and a domain")

    node = Product(scope)
    for i, s in enumerate(scope):
        leaf = learn_leaf_func(data[:, [i]], [distributions[i]], [domains[i]], [s], **learn_leaf_kwargs)
        leaf.id = i + 1  # Set the leaves ids sequentially
        node.children.append(leaf)
    return node
