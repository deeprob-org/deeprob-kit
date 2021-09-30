import numpy as np
import torch
import torchvision.utils as utils

from typing import Optional, Union, Tuple

from deeprob.utils.random import RandomState, check_random_state
from deeprob.spn.structure.node import Node
from deeprob.spn.algorithms.inference import log_likelihood
from deeprob.torch.base import ProbabilisticModel
#from deeprob.torch.routines import train_model, test_model
from deeprob.torch.datasets import UnsupervisedDataset, SupervisedDataset, WrappedDataset


def evaluate_log_likelihoods(
    spn: Node,
    x: np.ndarray,
    batch_size: int = 2048
) -> Tuple[float, float]:
    """
    Evaluate the average log-likelihood and two standard deviations.
    This function is implemented in batch mode in order to use less memory.

    :param spn: The SPN root node.
    :param x: The test data.
    :param batch_size: The size of each batch.
    :return: The average log-likelihoods and two standard deviations.
    """
    n_samples = len(x)
    ll = np.zeros(n_samples, dtype=np.float32)
    for i in range(0, n_samples - batch_size, batch_size):
        ll[i:i + batch_size] = log_likelihood(spn, x[i:i + batch_size])

    n_remaining_samples = n_samples % batch_size
    if n_remaining_samples > 0:
        ll[-n_remaining_samples:] = log_likelihood(spn, x[-n_remaining_samples:])

    mean_ll = np.mean(ll).item()
    stddev_ll = 2.0 * np.std(ll).item() / np.sqrt(n_samples)
    return mean_ll, stddev_ll


def collect_results_generative(
    model: ProbabilisticModel,
    data_train: Union[UnsupervisedDataset, WrappedDataset],
    data_valid: Union[UnsupervisedDataset, WrappedDataset],
    data_test: Union[UnsupervisedDataset, WrappedDataset],
    compute_bpp: bool = False,
    **kwargs
) -> Tuple[float, float, Optional[float]]:
    """
    Train and test a model in generative setting.

    :param model: The model.
    :param data_train: The train data.
    :param data_valid: The validation data.
    :param data_test: The test data.
    :param compute_bpp: Whether to compute bits-per-pixel (useful for images).
    :param kwargs: Other arguments to pass to train_model.
    :return: The average log-likelihoods with two standard deviation and an optional bits-per-pixel value.
    """
    # Train the model
    train_model(model, data_train, data_valid, setting='generative', **kwargs)

    # Test the model
    (mu_ll, sigma_ll) = test_model(model, data_test, setting='generative')

    # Compute the bits per pixel, if specified
    bpp = None
    if compute_bpp:
        dims = np.prod(data_train.features_shape)
        bpp = -(mu_ll / np.log(2)) / dims
    return mu_ll, sigma_ll, bpp


def collect_results_discriminative(
    model: ProbabilisticModel,
    data_train: Union[SupervisedDataset, WrappedDataset],
    data_valid: Union[SupervisedDataset, WrappedDataset],
    data_test: Union[SupervisedDataset, WrappedDataset],
    **kwargs
) -> Tuple[float, dict]:
    """
    Train and test a model in discriminative setting.

    :param model: The Torch model.
    :param data_train: The train data.
    :param data_valid: The validation data.
    :param data_test: The test data.
    :param kwargs: Other arguments to pass to train_model.
    :return: The negative log-likelihood (the loss value) and a dictionary representing a classification report.
    """
    # Train the model
    train_model(model, data_train, data_valid, setting='discriminative', **kwargs)

    # Test the model
    nll, metrics = test_model(model, data_test, setting='discriminative')
    return nll, metrics


def collect_samples(model: ProbabilisticModel, n_samples: int = 1) -> torch.Tensor:
    """
    Collect some samples given by a model.

    :param model: The model.
    :param n_samples: The number of samples.
    """
    # Make sure to switch to evaluation mode
    model.eval()

    # Sample some values
    return model.sample(n_samples).cpu()


def collect_image_completions(
    model: ProbabilisticModel,
    data_test: Union[UnsupervisedDataset, WrappedDataset],
    n_samples: int = 1,
    random_state: Optional[RandomState] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Sample some images, fill it with missing values and collect the completions.

    :param model: The model.
    :param data_test: The test data.
    :param n_samples: The number of samples.
    :param random_state: An optional random state.
    :param device: The device used for completions. If it's None 'cuda' will be used, if available.
    :return: A tensor consisting of original images and their artificial completions.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Make sure to switch to evaluation mode
    model.eval()

    # Check the random state
    random_state = check_random_state(random_state)

    # Sample some data points
    channels, height, width = data_test.features_shape
    samples_idx = random_state.choice(len(data_test), size=n_samples, replace=False)
    samples = torch.stack([data_test[i] for i in samples_idx]).to(device)

    # Compute image tensors with some missing data patterns
    half_width, half_height = width // 2, height // 2
    samples_up = torch.clone(samples)
    samples_up[:, :, :half_height, :] = np.nan
    samples_down = torch.clone(samples)
    samples_down[:, :, half_height:, :] = np.nan
    samples_left = torch.clone(samples)
    samples_left[:, :, :, :half_width] = np.nan
    samples_right = torch.clone(samples)
    samples_right[:, :, :, half_width:] = np.nan

    # Complete the images by most probable explanation (MPE) query
    uncomplete_samples = torch.cat([samples_up, samples_down, samples_left, samples_right])
    complete_samples = model.mpe(uncomplete_samples)
    samples = torch.cat([samples, complete_samples])
    samples = samples.reshape(5, n_samples, channels, height, width)
    samples = samples.permute(1, 0, 2, 3, 4)
    samples = samples.reshape(n_samples * 5, channels, height, width)
    return samples.cpu()


def save_grid_images(
    images: Union[np.ndarray, torch.Tensor],
    filepath: str,
    nrow: Optional[int] = None
):
    """
    Compose and save several images in a grid-like image.

    :param images: A Numpy array or Torch tensor of shape (N, C, H, W).
                   Each pixel must be in the normalized range [0, 1].
    :param nrow: Number of images displayed in each row of the grid.
                 If None, then floor(sqrt(len(images))) will be used.
    :param filepath: The filepath where to save the resulting image.
    """
    if isinstance(images, np.ndarray):
        images = torch.tensor(images)
    if nrow is None:
        nrow = int(np.sqrt(len(images)))
    utils.save_image(images, filepath, nrow=nrow, padding=0)
