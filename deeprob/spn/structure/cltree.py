# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from __future__ import annotations
from typing import Optional, Union, List

import numpy as np
import scipy.stats as ss
from scipy.special import logsumexp

from deeprob.utils.random import RandomState, check_random_state
from deeprob.utils.graph import build_tree_structure, compute_bfs_ordering, maximum_spanning_tree
from deeprob.utils.statistics import compute_mutual_information, estimate_priors_joints
from deeprob.spn.structure.leaf import Leaf, LeafType, Bernoulli
from deeprob.spn.structure.node import Node, Sum, Product, assign_ids


class BinaryCLT(Leaf):
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(
        self,
        scope: List[int],
        root: Optional[int] = None,
        tree: Optional[Union[List[int], np.ndarray]] = None,
        params: Optional[Union[List[List[List[float]]], np.ndarray]] = None
    ):
        """
        Initialize Binary Chow-Liu Tree (CLT) multi-variate leaf node.

        :param scope: The scope of the leaf.
        :param root: The root node of the CLT. If None it will be chosen randomly.
        :param tree: A sequence of variable ids predecessors (encoding the tree structure).
        :param params: The CLT conditional probability tables (CPTs), as a (N, 2, 2) Numpy array in logarithmic scale.
                       Note that params[i, l, k] = log P(X_i=k | Pa(X_i)=l).
        :raises ValueError: If the root variable is not in scope.
        :raises ValueError: If the tree structure is not compatible with the number of variables and root node.
        :raises ValueError: If the CPTs parameters are invalid.
        """
        super().__init__(scope)

        if tree is not None:
            if isinstance(tree, list):
                tree = np.array(tree, dtype=np.int32)

            # Check tree structure with respect to the scope
            if len(tree) != len(self.scope):
                raise ValueError("Invalid tree structure's number of variables")

            # Check root node with respect to the tree structure
            if root is None:
                root, = np.argwhere(tree == -1)
                if len(root) != 1:
                    raise ValueError("Invalid tree structure's root node")
                root = root.item()
            elif root not in self.scope:
                raise ValueError("The root variable must be in scope")
            else:
                root = self.scope.index(root)
            if tree[root] != -1:
                raise ValueError("Invalid tree structure's root node")

            # Compute BFS variable ordering
            bfs = compute_bfs_ordering(tree)
        else:
            bfs = None
            # Check root node with respect to the scope
            if root is not None:
                if root not in self.scope:
                    raise ValueError("The root variable must be in scope")
                root = self.scope.index(root)
        self.root = root
        self.tree = tree
        self.bfs = bfs

        # Initialize the parameters
        if isinstance(params, list):
            params = np.array(params, dtype=np.float32)
            if params.shape != (len(self.scope), 2, 2):
                raise ValueError("Invalid conditional probability table (CPT) shape")
            if not np.allclose(np.exp(params).sum(axis=2), 1.0):
                raise ValueError("Invalid conditional probability table (CPT) values")
        self.params = params

    @staticmethod
    def compute_clt_parameters(
        bfs: np.ndarray,
        tree: np.ndarray,
        priors: np.ndarray,
        joints: np.ndarray
    ) -> np.ndarray:
        """
        Compute the parameters of the CLTree given the tree structure and the priors and joints distributions.

        This function returns the conditional probability tables (CPTs) in a tensorized form.
        Note that params[i, l, k] = P(X_i=k | Pa(X_i)=l).
        A special case is made for the root distribution which is not conditioned.
        Note that params[root, :, k] = P(X_root=k).

        :param bfs: The bfs structure, i.e. a sequence of successors in a breadth-first traversal.
        :param tree: The tree structure, i.e. a sequence of predecessors in a tree structure.
        :param priors: The priors distributions.
        :param joints: The joints distributions.
        :return: The conditional probability tables (CPTs) in a tensorized form.
        """
        root_id = bfs[0]
        n_features = len(bfs)
        vs = np.arange(n_features)

        # Compute the conditional probabilities (by einsum operation)
        params = np.einsum('ikl,il->ilk', joints[vs, tree], np.reciprocal(priors[tree]))
        params[root_id] = priors[root_id]

        # Re-normalize the factors, because there can be FP32 approximation errors
        params /= np.sum(params, axis=2, keepdims=True)
        return params

    def em_init(self, random_state: np.random.RandomState):
        if self.tree is None:
            raise ValueError("The CLT's structure must be already initialized")

        probs = random_state.rand(len(self.scope), 2)
        probs[self.root, 0] = probs[self.root, 1]
        self.params[:, :, 1] = probs
        self.params[:, :, 0] = 1.0 - probs
        self.params = np.log(self.params)

    def em_step(self, stats: np.ndarray, data: np.ndarray, step_size: float):
        if self.tree is None:
            raise ValueError("The CLT's structure must be already initialized")

        alpha = np.finfo(np.float16).eps  # Use a very small Laplace smoothing factor
        total_stats = np.sum(stats)
        weighted_features = np.expand_dims(stats, axis=1) * data

        # Compute prior distributions
        priors_stats = np.sum(weighted_features, axis=0)
        priors = np.empty(shape=(len(self.scope), 2), dtype=np.float32)
        priors[:, 1] = (priors_stats + 2.0 * alpha) / (total_stats + 4.0 * alpha)
        priors[:, 0] = 1.0 - priors[:, 1]

        # Compute conditional sufficient statistics
        conditional_stats = np.empty(shape=(len(self.scope), 2), dtype=np.float32)
        conditional_stats[:, 1] = np.sum(weighted_features * data[:, self.tree], axis=0)
        conditional_stats[:, 0] = priors_stats - conditional_stats[:, 1]

        # Update the parameters
        params = np.empty_like(self.params)
        params[:, :, 1] = (conditional_stats + alpha) / (total_stats * priors[self.tree] + 4.0 * alpha)
        params[:, :, 0] = 1.0 - params[:, :, 1]
        params[self.root, 0] = params[self.root, 1] = priors[self.root]
        params = (1.0 - step_size) * np.exp(self.params) + step_size * params

        # Re-normalize the factors, because there can be FP32 approximation errors
        params /= np.sum(params, axis=2, keepdims=True)
        self.params = np.log(params)

    def fit(
        self,
        data: np.ndarray,
        domain: List[list],
        alpha: float = 0.1,
        random_state: Optional[RandomState] = None,
        **kwargs
    ):
        """
        Fit the distribution parameters (and structure if necessary) given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: The Laplace smoothing factor.
        :param random_state: The random state. It can be either None, a seed integer or a Numpy RandomState.
        :param kwargs: Optional parameters.
        :raises ValueError: If the random state is not valid.
        :raises ValueError: If a parameter is out of domain.
        """
        _, n_features = data.shape
        if len(domain) != n_features:
            raise ValueError("Each data column should correspond to a random variable having a domain")
        if not all(d == [0, 1] for d in domain):
            raise ValueError("The domains must be binary for a Binary CLT distribution")
        if alpha < 0.0:
            raise ValueError("The Laplace smoothing factor must be non-negative")

        # Check the random state
        random_state = check_random_state(random_state)

        # Choose a root variable randomly, if not specified
        if self.root is None:
            self.root = random_state.choice(len(self.scope))

        # Estimate the priors and joints probabilities
        priors, joints = estimate_priors_joints(data, alpha=alpha)

        if self.tree is None:
            # Compute the mutual information
            mutual_info = compute_mutual_information(priors, joints)

            # Compute the CLT structure
            self.bfs, self.tree = maximum_spanning_tree(self.root, mutual_info)

        # Compute the CLT parameters (in log-space), using the joints and priors probabilities
        params = self.compute_clt_parameters(self.bfs, self.tree, priors, joints)
        self.params = np.log(params)

    def message_passing(
        self, x: np.ndarray,
        obs_mask: np.ndarray,
        return_lls: bool = True,
        reduce: str = 'mar'
    ) -> np.ndarray:
        """
        Compute the messages passed from the leaves to the root node.

        :param x: The input data.
        :param obs_mask: The mask of observed values.
        :param return_lls: Whether to compute and return the log-likelihoods.
        :param reduce: The method used to reduce the messages of missing values.
                       It can be either 'mar' (marginalize the message) or 'mpe' (maximum probable explanation).
        :return: The messages array if return_lls is False.
                 The log-likelihoods if return_lls is True.
        """
        n_samples, n_features = x.shape
        messages = np.zeros(shape=(n_features, n_samples, 2), dtype=np.float32)

        # Let's proceed bottom-up
        for j in reversed(self.bfs[1:]):
            mask = obs_mask[:, j]
            mis_mask = ~mask
            obs_values = x[mask, j].astype(np.int64)
            msg = np.expand_dims(messages[j], axis=1)

            # Compute the messages for observed data
            messages[self.tree[j], mask] += self.params[j, :, obs_values] + msg[mask, :, obs_values]

            # Compute the messages for unobserved data
            if np.any(mis_mask):
                parent_msg = self.params[j] + msg[mis_mask]
                if reduce == 'mar':
                    messages[self.tree[j], mis_mask] += logsumexp(parent_msg, axis=2)
                elif reduce == 'mpe':
                    messages[self.tree[j], mis_mask] += np.max(parent_msg, axis=2)
                else:
                    raise ValueError("Unknown reduce method called {}".format(reduce))

        if not return_lls:
            return messages

        lls = np.empty(n_samples, dtype=np.float32)
        mask = obs_mask[:, self.root]
        mis_mask = ~mask
        obs_values = x[mask, self.root].astype(np.int64)
        msg = messages[self.root]

        # Compute the messages for observed data at root node
        lls[mask] = self.params[self.root, 0, obs_values] + msg[mask, obs_values]

        # Compute the messages for unobserved data at root node
        if np.any(mis_mask):
            lls[mis_mask] = logsumexp(self.params[self.root, 0] + msg[mis_mask], axis=1)

        return lls

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        n_samples, n_features = x.shape

        # Build the mask of samples with missing values (used for marginalization)
        mis_mask = np.isnan(x)
        mar_mask = np.any(mis_mask, axis=1)

        if np.any(mar_mask):
            evi_mask = ~mar_mask
            obs_mask = ~mis_mask
            lls = np.empty(n_samples, dtype=np.float32)

            # Vectorized implementation of full-evidence inference
            vs = np.arange(n_features)
            z = x[evi_mask]
            z_cond = z[:, self.tree].astype(np.int64, copy=False)
            z_vals = z[:, vs].astype(np.int64, copy=False)
            lls[evi_mask] = np.sum(self.params[vs, z_cond, z_vals], axis=1)

            # Semi-vectorized implementation of marginal inference
            z = x[mar_mask]
            lls[mar_mask] = self.message_passing(z, obs_mask[mar_mask], return_lls=True, reduce='mar')
            return np.expand_dims(lls, axis=1)

        # Vectorized implementation (without masking) of full-evidence inference
        vs = np.arange(n_features)
        x_cond = x[:, self.tree].astype(np.int64, copy=False)
        x_vals = x[:, vs].astype(np.int64, copy=False)
        lls = np.sum(self.params[vs, x_cond, x_vals], axis=1, keepdims=True)
        return lls

    def mpe(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mis_mask = np.isnan(x)
        obs_mask = ~mis_mask

        # Semi-vectorized implementation of MPE inference
        messages = self.message_passing(x, obs_mask, return_lls=False, reduce='mpe')

        # Compute MPE at the root feature
        mask = mis_mask[:, self.root]
        msg = self.params[self.root, 0] + messages[self.root, mask]
        x[mask, self.root] = np.argmax(msg, axis=1)

        # Compute MPE at the other features, by using the accumulated messages
        for j in self.bfs[1:]:
            mask = mis_mask[:, j]
            obs_parent_values = x[mask, self.tree[j]].astype(np.int64)
            msg = self.params[j, obs_parent_values] + messages[j, mask]
            x[mask, j] = np.argmax(msg, axis=1)
        return x

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        mis_mask = np.isnan(x)
        obs_mask = ~mis_mask

        # Semi-vectorized implementation of conditional sampling
        messages = self.message_passing(x, obs_mask, return_lls=False, reduce='mar')

        # Sample the root feature
        mask = mis_mask[:, self.root]
        log_probs = self.params[self.root, 0, 1] + messages[self.root, mask, 1]
        x[mask, self.root] = ss.bernoulli.rvs(np.exp(log_probs))

        # Sample the other features, by using the accumulated messages
        for j in self.bfs[1:]:
            mask = mis_mask[:, j]
            obs_parent_values = x[mask, self.tree[j]].astype(np.int64)
            log_probs = self.params[j, obs_parent_values, 1] + messages[j, mask, obs_parent_values]
            x[mask, j] = ss.bernoulli.rvs(np.exp(log_probs))
        return x

    def moment(self, k: int = 1) -> float:
        raise NotImplementedError("Computation of moments on Binary CLTs not yet implemented")

    def params_count(self) -> int:
        return 1 + len(self.tree) + self.params.size

    def params_dict(self) -> dict:
        return {
            'root': None if self.root is None else self.scope[self.root],
            'tree': self.tree,
            'params': self.params
        }

    def to_pc(self) -> Node:
        """
        Convert a Chow-Liu Tree into a smooth, deterministic and structured-decomposable PC

        :return: A smooth, deterministic and structured-decomposable PC.
        """
        # Build the tree structure
        root = build_tree_structure(self.tree, scope=self.scope)

        # Build the factors dictionary
        factors = {self.scope[i]: np.exp(self.params[i]) for i in range(len(self.tree))}

        # Post-Order exploration
        neg_buffer, pos_buffer = [], []
        nodes_stack = [root]
        last_node_visited = None
        while nodes_stack:
            node = nodes_stack[-1]
            if node.is_leaf() or (last_node_visited in node.get_children()):
                leaves: List[Union[Bernoulli, Sum]] = [
                    Bernoulli(node.get_id(), p=0.0),
                    Bernoulli(node.get_id(), p=1.0)
                ]
                if not node.is_leaf():
                    neg_prod = Product(children=[leaves[0]] + neg_buffer[-len(node.get_children()):])
                    pos_prod = Product(children=[leaves[1]] + pos_buffer[-len(node.get_children()):])
                    del neg_buffer[-len(node.get_children()):]
                    del pos_buffer[-len(node.get_children()):]
                    sum_children = [neg_prod, pos_prod]
                else:
                    sum_children = leaves
                weights = factors[node.get_id()]
                neg_buffer.append(
                    Sum(children=sum_children, weights=weights[0])
                )
                pos_buffer.append(
                    Sum(children=sum_children, weights=weights[1])
                )
                last_node_visited = nodes_stack.pop()
            else:
                nodes_stack.extend(node.get_children())
        # Equivalently, pos = neg_buffer[0]
        pc = pos_buffer[0]
        return assign_ids(pc)

    def get_scopes(self):
        """
        Return a list containing the scope of every node in the PC equivalent to the
        current CLTree (see to_pc() method). Every scope occurs once in the list.

        :return: The list of scopes.
        """
        scopes = []
        scopes_stack = []

        # Post-Order exploration
        root = build_tree_structure(self.tree, scope=self.scope)
        nodes_stack = [root]
        last_node_visited = None
        while nodes_stack:
            node = nodes_stack[-1]
            if node.is_leaf() or (last_node_visited in node.get_children()):
                if node.is_leaf():
                    scopes_stack.append([node.get_id()])
                else:
                    scopes_temp = scopes_stack[-len(node.get_children()):]
                    del scopes_stack[-len(node.get_children()):]
                    scopes_temp.append([node.get_id()])
                    merged_scope = [var for scope in scopes_temp for var in scope]
                    scopes_stack.append(merged_scope)
                    scopes.append(merged_scope)
                last_node_visited = nodes_stack.pop()
            else:
                nodes_stack.extend(node.get_children())

        return scopes
