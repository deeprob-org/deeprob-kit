# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Tuple

import numpy as np

from deeprob.utils.graph import maximum_spanning_tree
from deeprob.utils.statistics import estimate_priors_joints, compute_mutual_information
from deeprob.spn.utils.partitioning import Partition, generate_random_partitioning
from deeprob.spn.structure.node import Node, Sum, Product, assign_ids
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.leaf import Bernoulli
from deeprob.spn.learning.leaf import learn_mle


# SD stands for Structured Decomposable
SD_LEVEL_0 = 0  # non-SD ensemble of non-SD PCs
SD_LEVEL_1 = 1  # non-SD ensemble OF SD PCs
SD_LEVEL_2 = 2  # SD ensemble
SD_LEVELS = [SD_LEVEL_0, SD_LEVEL_1, SD_LEVEL_2]

ROOT = -1


def build_disjunction(
    data: np.ndarray,
    scope: list,
    assignments: Optional[np.ndarray] = None,
    alpha: float = 0.01
) -> Node:
    """
    Build a disjunction (sum node) of conjunctions (product nodes).
    If assignments are given, every conjunction is associated to a specific assignment (the number of conjunctions
    is the same as the given assignments); otherwise, every conjunction will be associated to a specific
    assignment occurring in the input data (the number of conjunctions is the same as the unique assignments
    occurring in the data).

    :param data: The input data matrix.
    :param scope: The scope.
    :param assignments: The optional assignments.
    :param alpha: Laplace smoothing factor.
    """
    unq_data, counts = np.unique(data, axis=0, return_counts=True)
    assignments = unq_data if assignments is None else assignments
    assert unq_data.shape[0] <= assignments.shape[0]

    weights = np.zeros(assignments.shape[0])
    for i in range(assignments.shape[0]):
        index = np.where(np.all(assignments[i] == unq_data, axis=1))[0]
        if len(index):
            weights[i] = counts[index[0]]
    weights = (weights + alpha) / (weights + alpha).sum()

    prod_nodes = []
    for i in range(assignments.shape[0]):
        children = []
        for j in range(assignments.shape[1]):
            children.append(Bernoulli(scope=[scope[j]], p=assignments[i, j]))
        prod_nodes.append(Product(children=children))

    disjunction = Sum(children=prod_nodes, weights=weights) if len(prod_nodes) > 1 else prod_nodes[0]
    return assign_ids(disjunction)


def build_leaf(
    data: np.ndarray,
    part: Partition,
    use_clt: bool,
    trees_dict: dict,
    det: bool,
    alpha: float
) -> Node:
    """
    Build a multivariate leaf distribution for an XPC.

    :param data: The input data matrix.
    :param part: The partition associated to the leaf to build.
    :param use_clt: True if it is possible to use CLTrees as leaf nodes, False otherwise.
    :param trees_dict: A dictionary of trees (see the function build_trees_dict).
    :param det: True to force determinism, False otherwise.
    :param alpha: Laplace smoothing factor.
    """
    data_slice = part.get_slice(data)
    scope = part.col_ids.tolist()

    if part.is_conj:
        leaf = Product(children=[Bernoulli(scope=[scope[k]], p=float(data_slice[0][k])) for k in range(len(scope))])

    elif part.is_naive or not use_clt:
        if not det or part.disc_assignments.shape[0] == 2 ** part.disc_assignments.shape[1]:
            leaf = learn_mle(data_slice, [Bernoulli] * len(scope), [[0, 1]] * len(scope), scope, alpha)
        else:
            leaf = build_disjunction(data=data_slice, scope=scope, assignments=part.disc_assignments, alpha=alpha)

    else:
        if trees_dict is not None:
            tree, scope = trees_dict[len(scope)]
            data_slice = data[part.row_ids][:, scope]
        else:
            tree, scope = None, part.col_ids.tolist()
        leaf = BinaryCLT(tree=tree, scope=scope)
        leaf.fit(data_slice, domain=[[0, 1]] * len(scope), alpha=0.01)

    return leaf


def greedy_vars_ordering(
    data: np.ndarray,
    conj_len: int,
    alpha: float = 0.01
) -> list:
    """
    Return the ordering of the random variables according to the implemented heuristic.

    :param data: The input data matrix.
    :param conj_len: The conjunction length.
    :param alpha: Laplace smoothing factor.

    :return ordering: The ordering.
    """
    priors, joints = estimate_priors_joints(data, alpha)
    mut_info = compute_mutual_information(priors, joints)
    sums = np.sum(mut_info, axis=0)

    ordering = []
    free_vars = np.arange(data.shape[1]).tolist()
    while free_vars:
        peek_var = free_vars[np.argmax(sums[free_vars])]
        free_vars.remove(peek_var)
        ordering.append(peek_var)
        if len(free_vars) > conj_len - 1:
            idx = np.argpartition(-mut_info[peek_var][free_vars], conj_len - 1)[:conj_len - 1]
            vars_ = np.array(free_vars)[idx].tolist()
        else:
            vars_ = free_vars.copy()
        free_vars = list(set(free_vars) - set(vars_))
        ordering.extend(vars_)
    return ordering


def build_trees_dict(
    data: np.ndarray,
    cl_parts_l: list,
    conj_vars_l: list,
    alpha: float,
    random_state: np.random.RandomState
) -> dict:
    """
    Return a dictionary where:
     - a key refers to a scope length
     - a value is a list of two lists: the first is a list of predecessors, the second its scope.

    :param data: The input data matrix.
    :param cl_parts_l: List of lists. Every sublist is associated to a specific XPC and contains
     the leaf partitions over which a CLTree will be learnt.
    :param conj_vars_l: List of lists. Every sublist contains the variables of a conjunction (e.g. [[3, 5]]).
     If a sublist occurs before another, then the former has been used first. There are no duplicates.
    :param alpha: Laplace smoothing factor.
    :param random_state: The random state.

    :return tree_dict: The dictionary.
    """
    # Compute the mutual information for each slice associated to every partition in cl_parts_l
    # and add it to a cumulative matrix (cumulative_info).
    n_vars = data.shape[1]
    cumulative_info = np.zeros((n_vars, n_vars))
    for cl_parts in cl_parts_l:
        for part in cl_parts:
            priors, joints = estimate_priors_joints(part.get_slice(data), alpha)
            mi = compute_mutual_information(priors, joints)
            cumulative_info[part.col_ids[:, None], part.col_ids] += mi

    # Free_vars are the variables not involved in any conjunction and will appear at the bottom of the circuit
    free_vars = list(set(np.arange(n_vars)) - set([var for conj_vars in conj_vars_l for var in conj_vars]))

    # Create a tree for each scope in scopes
    scopes = conj_vars_l + [free_vars] if free_vars else conj_vars_l
    trees = []
    for scope in scopes:
        _, tree = maximum_spanning_tree(
            adj_matrix=cumulative_info[scope][:, scope],
            root=scope.index(random_state.choice(scope))
        )
        trees.append(list(tree))

    # Concatenate trees and create the dictionary
    # The root of every tree is added as child to the root node of the tree with the minimum higher length scope.
    tree_dict = dict()
    tree = trees[-1].copy()
    scope = scopes[-1].copy()
    for k in reversed(range(0, len(trees) - 1)):
        tree_dict[len(scope)] = [tree.copy(), scope.copy()]
        tree += [t + len(scope) if t != ROOT else t for t in trees[k]]
        tree[tree.index(ROOT)] = tree.index(ROOT, len(scope))
        scope += scopes[k]

    return tree_dict


def build_xpc(
    data: np.ndarray,
    part_root: Partition,
    trees_dict: dict,
    det: bool,
    use_clt: bool,
    alpha: float
) -> Node:
    """
    Build the XPC induced by the partitions tree in a bottom up way.
    The building process is based on the post-order traversal exploration of the partitions tree.

    :param data: The input data matrix.
    :param part_root: The root partition of the tree.
    :param trees_dict: None if no dependency tree has to be respected, a dictionary of trees otherwise.
    :param det: True to force determinism, False otherwise.
    :param use_clt: True to use CLTrees as leaf nodes, False otherwise.
    :param alpha: Laplace smoothing factor.

    :return: the XPC induced by the partition tree
    """
    partitions_stack = [part_root]
    pc_nodes_stack = []
    last_part_visited = None

    while partitions_stack:
        part = partitions_stack[-1]
        if not part.is_partitioned() or (last_part_visited in part.sub_partitions):
            if part.is_partitioned():
                pc_child_nodes = pc_nodes_stack[-len(part.sub_partitions):]
                pc_nodes_stack = pc_nodes_stack[:len(pc_nodes_stack) - len(part.sub_partitions)]
                if part.is_horizontally_partitioned():
                    # Create sum node
                    weights = [len(sub_part.row_ids) / len(part.row_ids) for sub_part in part.sub_partitions]
                    pc_nodes_stack.append(Sum(weights=weights, children=pc_child_nodes))
                else:
                    # Create product node
                    pc_child_nodes_ = []
                    for c in pc_child_nodes:
                        if isinstance(c, Product) or (isinstance(c, Sum) and len(c.children) == 1):
                            pc_child_nodes_.extend(c.children)
                        else:
                            pc_child_nodes_.append(c)
                    pc_prod_node = Product(children=pc_child_nodes_)
                    pc_nodes_stack.append(pc_prod_node)
            else:
                # Create leaf (it could be either a PC or a multivariate leaf)
                leaf = build_leaf(data, part, use_clt, trees_dict, det, alpha)
                pc_nodes_stack.append(leaf)
            last_part_visited = partitions_stack.pop()
        else:
            partitions_stack.extend(part.sub_partitions[::-1])

    xpc = pc_nodes_stack[0]
    assign_ids(xpc)
    return xpc


def learn_xpc(
    data: np.ndarray,
    det: bool,
    sd: bool,
    min_part_inst: int,
    conj_len: int,
    arity: int,
    n_max_parts: int = 200,
    use_clt: bool = True,
    use_greedy_ordering: Optional[bool] = False,
    alpha: int = 0.01,
    random_seed: int = 42
) -> Tuple[Node, dict]:
    """
    Learn an eXtremely randomized Probabilistic Circuit (XPC).

    :param data: The input data matrix.
    :param det: True to force determinism, False otherwise.
    :param sd: True to force structured decomposability, False otherwise.
    :param min_part_inst: The minimum number of instances allowed per partition.
    :param conj_len: The conjunction length.
    :param arity: The maximum number of children for a sum node.
    :param n_max_parts: The maximum number of partitions for the partitions tree.
    :param use_clt: True to use CLTrees as multivariate leaves, False otherwise.
    :param use_greedy_ordering: True to use a greedy ordering, False otherwise.
    :param alpha: Laplace smoothing factor.
    :param random_seed: Random State.
    """
    assert arity > 1 or arity <= 2 ** conj_len, 'Arity must be in the interval [2, 2 ** conj_len]'
    assert sd or not use_greedy_ordering, 'Using the greedy ordering makes sense only if sd = True.'

    random_state = np.random.RandomState(random_seed)
    if use_greedy_ordering:
        ordering = greedy_vars_ordering(data, conj_len)
    else:
        ordering = np.arange(data.shape[1]).tolist()
        random_state.shuffle(ordering)

    part_root, cl_parts_l, conj_vars_l, n_parts = \
        generate_random_partitioning(
            data=data,
            sd=sd,
            min_part_inst=min_part_inst,
            conj_len=conj_len,
            arity=arity,
            n_max_parts=n_max_parts,
            uncond_vars=ordering,
            random_state=random_state)
    assert n_parts > 1, 'No partitioning found.'

    trees_dict = None
    if sd and use_clt:
        trees_dict = build_trees_dict(data, [cl_parts_l], conj_vars_l, alpha, random_state)

    # creating useful dictionary
    utils = {'part_root': part_root, 'cl_parts_l': cl_parts_l, 'conj_vars_l': conj_vars_l,
             'n_parts': n_parts, 'trees_dict': trees_dict}
    xpc = build_xpc(data, part_root, trees_dict, det, use_clt, alpha)
    return xpc, utils


def learn_expc(
    data: np.ndarray,
    ensemble_dim: int,
    det: bool,
    sd_level: int,
    min_part_inst: int,
    conj_len: int,
    arity: int,
    n_max_parts: int = 200,
    use_clt: bool = True,
    alpha: int = 0.01,
    random_seed: int = 42
) -> Tuple[Node, list]:
    """
    Learn an Ensemble (i.e. a mixture) of eXtremely randomized Probabilistic Circuit (EXPC).

    :param data: The input data matrix.
    :param ensemble_dim: The number of circuits in the ensemble/mixture.
    :param det: True to force determinism, False otherwise.
    :param sd_level: 0 a non-SD ensemble of non-SD PCs, 1 for a non-SD ensemble of SD PCs and 2 for a SD ensemble.
    :param min_part_inst: The minimum number of instances allowed per partition.
    :param conj_len: The conjunction length.
    :param arity: The maximum number of children for a Sum node.
    :param n_max_parts: The maximum number of partitions for the partitions tree.
    :param use_clt: True to use CLTrees as multivariate leaves, False otherwise.
    :param alpha: Laplace smoothing factor.
    :param random_seed: A random seed.
    """
    assert sd_level in SD_LEVELS, 'Choose a value in {0, 1, 2}.'
    assert arity > 1 or arity <= 2 ** conj_len, 'Arity must be in the interval [2, 2 ** conj_len].'
    assert not (sd_level == SD_LEVEL_2 and conj_len == 1), 'No randomness in this setting. Change hyper parameters.'

    random_state = np.random.RandomState(random_seed)
    conj_vars_l_l = [None] * ensemble_dim
    cl_parts_l_l = [None] * ensemble_dim
    trees_dict_l = [None] * ensemble_dim
    part_root_l = [None] * ensemble_dim
    n_parts_l = [None] * ensemble_dim
    xpc_l = [None] * ensemble_dim

    sd = (sd_level in [SD_LEVEL_1, SD_LEVEL_2])
    if sd_level == SD_LEVEL_2:
        ordering = greedy_vars_ordering(data, conj_len)
    else:
        ordering = np.arange(data.shape[1]).tolist()

    for i in range(ensemble_dim):
        if sd_level != SD_LEVEL_2:
            np.random.shuffle(ordering)
        part_root_l[i], cl_parts_l_l[i], conj_vars_l_l[i], n_parts_l[i] = \
            generate_random_partitioning(
                data=data,
                sd=sd,
                min_part_inst=min_part_inst,
                conj_len=conj_len,
                arity=arity,
                n_max_parts=n_max_parts,
                uncond_vars=ordering,
                random_state=random_state)
    assert not all(n_parts == 1 for n_parts in n_parts_l), 'No Partitioning Found'

    if sd_level == SD_LEVEL_0 or not use_clt:
        # no tree structure to respect
        trees_dict = None
        for i in range(ensemble_dim):
            print('Learning XPC %s/%s' % (i + 1, ensemble_dim))
            xpc_l[i] = build_xpc(data, part_root_l[i], trees_dict, det, use_clt, alpha)
    elif sd_level == SD_LEVEL_1:
        for i in range(ensemble_dim):
            print('Learning XPC %s/%s' % (i + 1, ensemble_dim))
            # learn a tree for each XPC
            trees_dict_l[i] = build_trees_dict(data, [cl_parts_l_l[i]], conj_vars_l_l[i], alpha, random_state)
            xpc_l[i] = build_xpc(data, part_root_l[i], trees_dict_l[i], det, use_clt, alpha)
    elif sd_level == SD_LEVEL_2:
        # learn a tree structure for the whole ensemble
        print('Learning a dependency tree for the ensemble..')
        trees_dict = build_trees_dict(data, cl_parts_l_l, max(conj_vars_l_l, key=len), alpha, random_state)
        trees_dict_l = [trees_dict] * ensemble_dim
        for i in range(ensemble_dim):
            print('Building XPC %s/%s' % (i + 1, ensemble_dim))
            xpc_l[i] = build_xpc(data, part_root_l[i], trees_dict, det, use_clt, alpha)

    # creating useful list of dictionaries
    utils = [{'part_root': part_root_l[i], 'cl_parts_l': cl_parts_l_l[i], 'conj_vars_l': conj_vars_l_l[i],
              'n_parts': n_parts_l[i], 'trees_dict': trees_dict_l[i]} for i in range(ensemble_dim)]
    expc = Sum(weights=np.full(ensemble_dim, 1 / ensemble_dim), children=xpc_l)
    assign_ids(expc)
    return expc, utils
