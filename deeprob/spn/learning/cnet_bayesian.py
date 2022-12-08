from __future__ import annotations
from typing import Optional, List

import numpy as np
from scipy.special import gammaln

from deeprob.utils.statistics import compute_prior_counts, compute_joint_counts, estimate_priors_joints
from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.spn.structure.cnet import BinaryCNet


def compute_or_bd_scores(
    data: np.ndarray,
    ess: float = 0.1
):
    """
    Compute the BDeu scores for the candidate OR nodes given the data.

    :param data: The binary data matrix.
    :param ess: The equivalent sample size (ESS).
    :return: The score array.
    """
    n_samples, n_features = data.shape
    prior_counts = compute_prior_counts(data=data)
    alpha_i = ess
    alpha_ik = ess / 2
    log_gamma_nodes = gammaln(alpha_i) - gammaln(n_samples + alpha_i) \
                      + np.sum(gammaln(prior_counts + alpha_ik) - gammaln(alpha_ik), axis=-1)
    return log_gamma_nodes


def compute_clt_bd_scores(
    data: np.ndarray,
    ess: float = 0.1
):
    """
    Compute the pairwise BDeu scores for constructing a CLT given the data.

    :param data: The binary data matrix.
    :param ess: The equivalent sample size (ESS).
    :return: The pairwise BDeu score matrix.
    """
    joint_counts = compute_joint_counts(data=data)
    alpha_ij = ess / 2
    alpha_ijk = ess / (2 * 2)
    parent_counts = np.sum(joint_counts, axis=-2)
    log_gamma_pairs = gammaln(alpha_ij) - gammaln(parent_counts + alpha_ij) \
                      + np.sum(gammaln(joint_counts + alpha_ijk) - gammaln(alpha_ijk), axis=-2)
    return np.sum(log_gamma_pairs, axis=-1)


def estimate_clt_params_bayesian(
    clt: BinaryCLT,
    data: np.ndarray,
    ess: float = 0.1
):
    """
    Compute the Bayesian posterior parameters for a CLT.

    :param clt: The CLT.
    :param data: The binary data matrix.
    :param ess: The equivalent sample size (ESS).
    :return: The CLT parameters in the log space.
    """
    n_samples, n_features = data.shape
    priors, joints = estimate_priors_joints(data, alpha=ess / 4)

    vs = np.arange(n_features)
    params = np.einsum('ikl,il->ilk', joints[vs, clt.tree], np.reciprocal(priors[clt.tree]))
    params[clt.root] = priors[clt.root]

    # Re-normalize the factors, because there can be FP32 approximation errors
    params /= np.sum(params, axis=2, keepdims=True)
    return np.log(params)


def eval_tree_score(
    tree: Optional[List[int], np.ndarray],
    clt_scores: np.ndarray,
    or_scores: np.ndarray
):
    """
    Evaluate the BDeu score for a tree structure.

    :param tree: The tree structure.
    :param clt_scores: The pairwise score matrix.
    :param or_scores: The OR score array.
    :return: The BDeu score of the tree structure.
    """
    root_idx = tree.argmin()
    parent_indices_no_root = np.delete(tree, obj=root_idx)
    child_indices_no_root = np.delete(np.arange(len(tree)), obj=root_idx)
    return np.sum(clt_scores[child_indices_no_root, parent_indices_no_root]) + or_scores[root_idx]


def select_cand_cuts(
    data: np.ndarray,
    ess: float = 0.1,
    n_cand_cuts: int = 10
):
    """
    Select the candidate cutting nodes.

    :param data: The binary data.
    :param ess: The equivalent sample size (ESS).
    :param n_cand_cuts: The number of candidate cutting nodes.
    :return: The indices of the selected nodes.
    """
    # Compute the counts
    n_samples, n_features = data.shape
    counts_features = data.sum(axis=0)

    prior_counts = compute_prior_counts(data)
    joint_counts = compute_joint_counts(data)
    smoothing_joint, smoothing_prior = ess / 2, ess
    if ess < 0.01:
        prior_counts = prior_counts.astype(np.float64)
        joint_counts = joint_counts.astype(np.float64)
    log_priors = np.log(prior_counts + smoothing_joint) - np.log(n_samples + smoothing_prior)
    mean_entropy = -(log_priors * np.exp(log_priors)).sum() / n_features

    conditionals = np.empty((n_features, n_features, 2, 2), dtype=prior_counts.dtype)
    conditionals[:, :, 0, 0] = ((joint_counts[:, :, 0, 0] + smoothing_joint).T /
                                (prior_counts[:, 0] + smoothing_prior)).T
    conditionals[:, :, 0, 1] = ((joint_counts[:, :, 0, 1] + smoothing_joint).T /
                                (prior_counts[:, 0] + smoothing_prior)).T
    conditionals[:, :, 1, 0] = ((joint_counts[:, :, 1, 0] + smoothing_joint).T /
                                (prior_counts[:, 1] + smoothing_prior)).T
    conditionals[:, :, 1, 1] = ((joint_counts[:, :, 1, 1] + smoothing_joint).T /
                                (prior_counts[:, 1] + smoothing_prior)).T

    vs = np.repeat(np.arange(n_features)[None, :], n_features, axis=0)
    vs = vs[~np.eye(vs.shape[0], dtype=bool)].reshape(vs.shape[0], -1)
    parents = np.repeat(np.arange(n_features)[:, None], n_features - 1, axis=1)

    ratio_features = counts_features / n_samples
    entropies = ratio_features * \
                np.mean(-np.sum(conditionals[parents, vs, 1, :] * np.log(conditionals[parents, vs, 1, :]), axis=-1),
                        axis=1) + \
                (1 - ratio_features) * \
                np.mean(-np.sum(conditionals[parents, vs, 0, :] * np.log(conditionals[parents, vs, 0, :]), axis=-1),
                        axis=1)

    info_gains = mean_entropy - entropies
    selected_idx = np.argmax(info_gains) if n_cand_cuts == 1 else np.argpartition(info_gains,
                                                                                  -n_cand_cuts)[-n_cand_cuts:]
    return selected_idx


def learn_cnet_bd(
    data: np.ndarray,
    ess: float = 0.1,
    n_cand_cuts: int = 10,
):
    """
    Learn a binary CNet using the Bayesian-Dirichlet equivalent uniform (BDeu) score.

    :param cnet: The binary CNet.
    :param data: The training data.
    :param ess: The equivalent sample size (ESS).
    :param n_cand_cuts: The number of candidate cutting nodes.
    :return: A binary CNet.
    """
    n_samples, n_features = data.shape
    root = BinaryCNet(scope=list(range(n_features)))
    root.assign_indices(row_indices=np.arange(n_samples), col_indices=np.arange(n_features))
    root.fit_clt(data=data)
    # use Bayesian posterior parameters.
    root.clt.params = estimate_clt_params_bayesian(clt=root.clt, data=data, ess=ess)
    or_score_matrix = compute_or_bd_scores(data=data, ess=ess)
    clt_score_matrix = compute_clt_bd_scores(data=data, ess=ess)
    clt_score = eval_tree_score(tree=root.clt.tree, clt_scores=clt_score_matrix, or_scores=or_score_matrix)

    node_stack = [[root, ess, clt_score]]
    while node_stack:
        node, node_ess, node_clt_score = node_stack.pop(0)
        if len(node.scope) == 1:
            continue

        partition = data[node.row_indices][:, node.col_indices]
        or_score_matrix = compute_or_bd_scores(data=partition, ess=node_ess)

        k = min(n_cand_cuts, len(node.scope))
        search_indices = select_cand_cuts(data=partition, ess=node_ess, n_cand_cuts=k)

        best_or_idx = -1
        best_cnet_score = -np.inf
        best_left_clt = None
        best_right_clt = None
        best_left_clt_score = -np.inf
        best_right_clt_score = -np.inf

        for i in search_indices:
            left_row_indices = node.row_indices[partition[:, i] == 0]
            right_row_indices = node.row_indices[partition[:, i] == 1]

            if len(left_row_indices) == 0 or len(right_row_indices) == 0:
                continue

            child_col_indices = np.delete(node.col_indices, obj=i)
            left_partition = data[left_row_indices][:, child_col_indices]
            right_partition = data[right_row_indices][:, child_col_indices]
            new_scope = node.scope.copy()
            del new_scope[i]

            left_clt = BinaryCLT(scope=new_scope)
            left_or_score_matrix = compute_or_bd_scores(data=left_partition, ess=node_ess / 2)
            left_clt_score_matrix = compute_clt_bd_scores(data=left_partition, ess=node_ess / 2)

            right_clt = BinaryCLT(scope=new_scope)
            right_or_score_matrix = compute_or_bd_scores(data=right_partition, ess=node_ess / 2)
            right_clt_score_matrix = compute_clt_bd_scores(data=right_partition, ess=node_ess / 2)

            left_clt.fit(data=left_partition, domain=[[0, 1]] * len(new_scope), alpha=0.01)
            right_clt.fit(data=right_partition, domain=[[0, 1]] * len(new_scope), alpha=0.01)

            left_clt.params = estimate_clt_params_bayesian(left_clt, data=left_partition, ess=node_ess / 2)
            right_clt.params = estimate_clt_params_bayesian(right_clt, data=right_partition, ess=node_ess / 2)

            left_clt_score = eval_tree_score(tree=left_clt.tree,
                                             clt_scores=left_clt_score_matrix,
                                             or_scores=left_or_score_matrix)
            right_clt_score = eval_tree_score(tree=right_clt.tree,
                                              clt_scores=right_clt_score_matrix,
                                              or_scores=right_or_score_matrix)
            cnet_score = left_clt_score + right_clt_score + or_score_matrix[i]

            if cnet_score > best_cnet_score:
                best_cnet_score = cnet_score
                best_or_idx = i
                best_left_clt = left_clt
                best_right_clt = right_clt
                best_left_clt_score = left_clt_score
                best_right_clt_score = right_clt_score

        if best_cnet_score > node_clt_score:
            node.or_id = node.scope[best_or_idx]
            node.clt = None
            left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
            right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
            child_col_indices = np.delete(node.col_indices, obj=best_or_idx)
            left_weight = (len(left_row_indices) + node_ess / 2) / (len(node.row_indices) + node_ess)
            right_weight = 1 - left_weight
            new_scope = node.scope.copy()
            del new_scope[best_or_idx]

            left_child = BinaryCNet(scope=new_scope)
            left_child.clt = best_left_clt
            left_child.assign_indices(row_indices=left_row_indices, col_indices=child_col_indices)
            right_child = BinaryCNet(scope=new_scope)
            right_child.clt = best_right_clt
            right_child.assign_indices(row_indices=right_row_indices, col_indices=child_col_indices)

            node_stack.append([left_child, node_ess / 2, best_left_clt_score])
            node_stack.append([right_child, node_ess / 2, best_right_clt_score])
            node.weights = [left_weight, right_weight]
            node.children = [left_child, right_child]
    return root


def learn_cnet_bic(
    data: np.ndarray,
    alpha: float = 0.01,
    n_cand_cuts: int = 10,
):
    """
    Learn a binary CNet using the Bayesian Information Criterion (BIC) score.

    :param data: The binary data.
    :param alpha: The Laplace smoothing factor.
    :param n_cand_cuts: The number of candidate cutting nodes.
    :return: A binary CNet.
    """
    n_samples, n_features = data.shape
    root = BinaryCNet(scope=list(range(n_features)))
    root.assign_indices(row_indices=np.arange(n_samples), col_indices=np.arange(n_features))
    root.fit_clt(data=data)
    clt_score = np.sum(root.clt.log_likelihood(data)) - 0.5 * np.log(n_samples) * (2 * n_features - 1)

    node_stack = [[root, clt_score]]
    while node_stack:
        node, node_clt_score = node_stack.pop(0)
        if len(node.scope) == 1:
            continue

        partition = data[node.row_indices][:, node.col_indices]

        k = min(n_cand_cuts, len(node.scope))
        search_indices = select_cand_cuts(partition, ess=4 * alpha, n_cand_cuts=k)

        best_or_idx = -1
        best_cnet_score = -np.inf
        best_left_clt = None
        best_right_clt = None
        best_left_clt_score = 0.0
        best_right_clt_score = 0.0
        for i in search_indices:
            left_row_indices = node.row_indices[partition[:, i] == 0]
            right_row_indices = node.row_indices[partition[:, i] == 1]

            if len(left_row_indices) == 0 or len(right_row_indices) == 0:
                continue

            child_col_indices = np.delete(node.col_indices, obj=i)
            left_partition = data[left_row_indices][:, child_col_indices]
            right_partition = data[right_row_indices][:, child_col_indices]
            new_scope = node.scope.copy()
            del new_scope[i]

            left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
            right_weight = 1 - left_weight

            left_clt = BinaryCLT(scope=new_scope)
            right_clt = BinaryCLT(scope=new_scope)

            left_clt.fit(data=left_partition, domain=[[0, 1]] * len(new_scope), alpha=alpha)
            right_clt.fit(data=right_partition, domain=[[0, 1]] * len(new_scope), alpha=alpha)

            left_clt_score = np.sum(left_clt.log_likelihood(left_partition)) \
                             - 0.5 * np.log(len(data)) * (2 * len(new_scope) - 1)
            right_clt_score = np.sum(right_clt.log_likelihood(right_partition)) \
                              - 0.5 * np.log(len(data)) * (2 * len(new_scope) - 1)
            or_score = len(left_partition) * np.log(left_weight) + len(right_partition) * np.log(right_weight) \
                       - 0.5 * np.log(len(data))
            cnet_score = left_clt_score + right_clt_score + or_score

            if cnet_score > best_cnet_score:
                best_cnet_score = cnet_score
                best_or_idx = i
                best_left_clt = left_clt
                best_right_clt = right_clt
                best_left_clt_score = left_clt_score
                best_right_clt_score = right_clt_score

        if best_cnet_score > node_clt_score:
            node.or_id = node.scope[best_or_idx]
            node.clt = None
            left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
            right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
            child_col_indices = np.delete(node.col_indices, obj=best_or_idx)

            left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
            right_weight = 1 - left_weight
            new_scope = node.scope.copy()
            del new_scope[best_or_idx]

            left_child = BinaryCNet(scope=new_scope)
            left_child.clt = best_left_clt
            left_child.assign_indices(row_indices=left_row_indices, col_indices=child_col_indices)
            right_child = BinaryCNet(scope=new_scope)
            right_child.clt = best_right_clt
            right_child.assign_indices(row_indices=right_row_indices, col_indices=child_col_indices)

            node_stack.append([left_child, best_left_clt_score])
            node_stack.append([right_child, best_right_clt_score])
            node.weights = [left_weight, right_weight]
            node.children = [left_child, right_child]
    return root
