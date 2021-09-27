import numpy as np

from typing import Tuple

from deeprob.utils.data import check_data_dtype


def compute_mean_quantiles(data: np.ndarray, n_quantiles: int) -> np.ndarray:
    """
    Compute the mean quantiles of a dataset (Poon-Domingos).

    :param data: The data.
    :param n_quantiles: The number of quantiles.
    :return: The mean quantiles.
    :raises ValueError: If the number of quantiles is not valid.
    """
    n_samples = len(data)
    if n_quantiles <= 0 or n_quantiles > n_samples:
        raise ValueError("The number of quantiles must be positive and less or equal than the number of samples")

    # Split the dataset in quantiles regions
    data = np.sort(data, axis=0)
    values_per_quantile = np.array_split(data, n_quantiles, axis=0)

    # Compute the mean quantiles
    mean_per_quantiles = [np.mean(x, axis=0) for x in values_per_quantile]
    return np.stack(mean_per_quantiles, axis=0)


def compute_mutual_information(priors: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """
    Compute the mutual information between each features, given priors and joints distributions.

    :param priors: The priors probability distributions, as a (N, D) Numpy array
                   having priors[i, k] = P(X_i=k).
    :param joints: The joints probability distributions, as a (N, N, D, D) Numpy array
                   having joints[i, j, k, l] = P(X_i=k, X_j=l).
    :return: The mutual information between each pair of features, as a (N, N) Numpy symmetric matrix.
    :raises ValueError: If there are inconsistencies between priors and joints arrays.
    :raises ValueError: If joints array is not symmetric.
    :raises ValueError: If priors or joints arrays don't encode valid probability distributions.
    """
    n_variables, n_values = priors.shape
    if joints.shape != (n_variables, n_variables, n_values, n_values):
        raise ValueError("There are inconsistencies between priors and joints distributions")
    if not np.all(joints == joints.transpose([1, 0, 3, 2])):
        raise ValueError("The joints probability distributions are expected to be symmetric")
    if not np.allclose(np.sum(priors, axis=1), 1.0):
        raise ValueError("The priors probability distributions are not valid")
    if not np.allclose(np.sum(joints, axis=(2, 3)), 1.0):
        raise ValueError("The joints probability distributions are not valid ")

    outers = np.multiply.outer(priors, priors).transpose([0, 2, 1, 3])
    # Ignore warnings of logarithm at zero (because NaNs on the diagonal will be zeroed later anyway)
    with np.errstate(divide='ignore', invalid='ignore'):
        mutual_info = np.sum(joints * (np.log(joints) - np.log(outers)), axis=(2, 3))
    np.fill_diagonal(mutual_info, 0.0)
    return mutual_info


def estimate_priors_joints(data: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate both priors and joints probability distributions from binary data.

    This function returns both the prior distributions and the joint distributions.
    Note that priors[i, k] = P(X_i=k) and joints[i, j, k, l] = P(X_i=k, X_j=l).

    :param data: The binary data matrix.
    :param alpha: The Laplace smoothing factor.
    :return: A pair of priors and joints distributions.
    :raises ValueError: If the Laplace smoothing factor is out of domain.
    """
    if alpha < 0.0:
        raise ValueError("The Laplace smoothing factor must be non-negative")

    # Check the data dtype
    data = check_data_dtype(data, dtype=np.float32)

    # Compute the counts
    n_samples, n_features = data.shape
    counts_ones = np.dot(data.T, data)
    counts_features = np.diag(counts_ones)
    counts_cols = counts_features * np.ones_like(counts_ones)
    counts_rows = np.transpose(counts_cols)

    # Compute the prior probabilities
    priors = np.empty(shape=(n_features, 2), dtype=data.dtype)
    priors[:, 1] = (counts_features + 2 * alpha) / (n_samples + 4 * alpha)
    priors[:, 0] = 1.0 - priors[:, 1]

    # Compute the joints probabilities
    joints = np.empty(shape=(n_features, n_features, 2, 2), dtype=data.dtype)
    joints[:, :, 0, 0] = n_samples - counts_cols - counts_rows + counts_ones
    joints[:, :, 0, 1] = counts_cols - counts_ones
    joints[:, :, 1, 0] = counts_rows - counts_ones
    joints[:, :, 1, 1] = counts_ones
    joints = (joints + alpha) / (n_samples + 4 * alpha)

    # Correct smoothing on the diagonal of joints array
    idx_features = np.arange(n_features)
    joints[idx_features, idx_features, 0, 0] = priors[:, 0]
    joints[idx_features, idx_features, 0, 1] = 0.0
    joints[idx_features, idx_features, 1, 0] = 0.0
    joints[idx_features, idx_features, 1, 1] = priors[:, 1]

    return priors, joints
