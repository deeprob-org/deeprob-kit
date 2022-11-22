from __future__ import annotations
from typing import Optional, Union, List

import copy
import numpy as np

from deeprob.utils.statistics import compute_prior_counts, compute_joint_counts
from deeprob.spn.structure.node import Node
from deeprob.spn.structure.cltree import BinaryCLT


class ORNode(Node):
    def __init__(
        self,
        scope: Optional[List[int]],
        children: Optional[List[Node]] = None,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        or_id: Optional[int] = None
    ):
        """
        Initialize an OR node given weights, child instances and child nodes.

        :param scope: The scope of the OR node.
        :param children: The child nodes of the OR node.
        :param weights: The weights of the OR node.
        :param or_id: The id of the OR node.
        """
        if weights is not None:
            if isinstance(weights, list):
                weights = np.array(weights, dtype=np.float32)
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("Weights don't sum up to 1")
        self.weights = weights
        self.or_id = or_id
        self.row_indices = None
        self.col_indices = None
        self.clt = None

        super().__init__(scope, children)

    def assign_indices(
        self,
        row_indices: Optional[List[int], np.ndarray],
        col_indices: Optional[List[int], np.ndarray]
    ):
        """
        Assign the corresponding indices of the OR node's partition in the original data set.

        :param row_indices: Row indices of the partition.
        :param col_indices: Column indices of the partition.
        :return:
        """
        self.row_indices = row_indices
        self.col_indices = col_indices

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        pass

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        pass


class BinaryCNet(ORNode):
    def __init__(
        self,
        scope: Optional[List[int]],
        children: Optional[List[Node]] = None,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        or_id: Optional[int] = None
    ):
        """
        Initialize a binary cutset network (CNet).

        :param scope: The scope of the binary CNet.
        :param children: The child OR nodes of the binary CNet.
        :param weights: The weights of the current OR node.
        :param or_id: The id of the current OR node.
        """
        super().__init__(scope, children, weights, or_id)

    def fit(
        self,
        data: np.ndarray,
        alpha: float = 0.01,
        min_n_samples: int = 10,
        min_n_features: int = 1,
        min_mean_entropy: float = 0.01
    ):
        """
        Fit the structure and the MLE parameters given some training data and hyper-parameters.

        :param data: The training data.
        :param alpha: The Laplace smoothing factor.
        :param min_n_samples: The minimum number of samples to split.
        :param min_n_features: The minimum number of features to split.
        :param min_mean_entropy: The minimum mean entropy of RVs given the data to split.
        :return:
        """
        n_samples, n_features = data.shape
        self.scope = list(range(n_features))
        self.assign_indices(row_indices=np.arange(n_samples), col_indices=np.arange(n_features))
        root = BinaryCNet(scope=list(range(n_features)))
        root.assign_indices(row_indices=np.arange(n_samples), col_indices=np.arange(n_features))
        node_stack = [root]
        while node_stack:
            node = node_stack.pop(0)
            partition = data[node.row_indices][:, node.col_indices]
            n_samples, n_features = partition.shape
            if n_samples <= min_n_samples or n_features <= min_n_features:
                # stopped due to few samples or features
                node.fit_clt(data=partition, alpha=alpha)
                continue
            best_or_idx, mean_entropy, max_info_gain = self.__select_variable_entropy(partition, alpha=alpha)
            if mean_entropy < min_mean_entropy or max_info_gain <= 0:
                # stopped due to small entropy or negative information gain
                node.fit_clt(data=partition, alpha=alpha)
                continue
            left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
            right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
            child_col_indices = np.delete(node.col_indices, obj=best_or_idx)
            left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
            right_weight = 1 - left_weight
            new_scope = node.scope.copy()
            del new_scope[best_or_idx]
            left_child = BinaryCNet(scope=new_scope)
            left_child.assign_indices(row_indices=left_row_indices, col_indices=child_col_indices)
            right_child = BinaryCNet(scope=new_scope)
            right_child.assign_indices(row_indices=right_row_indices, col_indices=child_col_indices)
            node_stack.append(left_child)
            node_stack.append(right_child)
            node.children = [left_child, right_child]
            node.weights = [left_weight, right_weight]
            node.or_id = node.scope[best_or_idx]
        self.or_id = root.or_id
        self.children = root.children
        self.weights = root.weights

    def fit_clt(
        self,
        data: np.ndarray,
        alpha: float = 0.01
    ):
        """
        Fit a Binary CLT for the RVs in the scope of the current OR node.

        :param data: The data partition.
        :param alpha: The laplace smoothing factor.
        :return:
        """
        clt = BinaryCLT(scope=self.scope)
        clt.fit(data=data, domain=[[0, 1]] * len(self.scope), alpha=alpha)
        self.clt = clt

    @staticmethod
    def __select_variable_entropy(
        data: np.ndarray,
        alpha: float = 0.01
    ):
        """
        Select the best cut node based on the reduced entropy (information gain).

        :param data: The training data partition.
        :param alpha: The Laplace smoothing factor.
        :return: The index of the selected RV,
                 the mean entropy of the RVs in the partition,
                 the information gain of the selected RV.
        """
        n_samples, n_features = data.shape
        counts_features = np.sum(data, axis=0)

        prior_counts = compute_prior_counts(data)
        joint_counts = compute_joint_counts(data)
        priors = (prior_counts + 2 * alpha) / (n_samples + 4 * alpha)
        priors[:, 0] = 1.0 - priors[:, 1]
        mean_entropy = -(priors * np.log(priors)).sum() / n_features

        conditionals = np.empty((n_features, n_features, 2, 2), dtype=np.float32)
        # as we are computing the probabilities for all nodes after cutting on a node,
        # the laplace smoothing factor is essentially the same as computing general prior probabilities
        conditionals[:, :, 0, 0] = ((joint_counts[:, :, 0, 0] + 2 * alpha).T / (prior_counts[:, 0] + 4 * alpha)).T
        conditionals[:, :, 0, 1] = ((joint_counts[:, :, 0, 1] + 2 * alpha).T / (prior_counts[:, 0] + 4 * alpha)).T
        conditionals[:, :, 1, 0] = ((joint_counts[:, :, 1, 0] + 2 * alpha).T / (prior_counts[:, 1] + 4 * alpha)).T
        conditionals[:, :, 1, 1] = ((joint_counts[:, :, 1, 1] + 2 * alpha).T / (prior_counts[:, 1] + 4 * alpha)).T

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
        selected_idx = np.argmax(info_gains)
        return selected_idx, mean_entropy, info_gains[selected_idx]

    def __is_leaf(self):
        """
        Check if the current OR node is a leaf.

        :return: True if the OR node has fitted a binary CLT, otherwise False
        """
        return True if self.clt else False

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        n_samples, n_features = x.shape
        root = copy.copy(self)
        root.row_indices, root.col_indices = np.arange(n_samples), np.arange(n_features)
        node_stack = [root]
        log_likes = np.zeros(n_samples)
        while node_stack:
            node = node_stack.pop(0)
            partition = x[node.row_indices][:, node.col_indices]
            if node.__is_leaf():
                log_likes[node.row_indices] += node.clt.log_likelihood(partition).squeeze()
                continue
            node_idx = node.scope.index(node.or_id)
            left_child = copy.copy(node.children[0])
            right_child = copy.copy(node.children[1])
            left_child.row_indices = node.row_indices[partition[:, node_idx] == 0]
            right_child.row_indices = node.row_indices[partition[:, node_idx] == 1]
            log_likes[left_child.row_indices] += np.log(node.weights[0])
            log_likes[right_child.row_indices] += np.log(node.weights[1])
            left_child.col_indices = np.delete(node.col_indices, obj=node_idx)
            right_child.col_indices = np.delete(node.col_indices, obj=node_idx)
            node_stack.append(left_child)
            node_stack.append(right_child)
        return log_likes
