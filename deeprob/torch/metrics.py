# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Optional, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision import models
from tqdm import tqdm

from deeprob.utils.statistics import compute_fid


class RunningAverageMetric:
    def __init__(self):
        """
        Initialize a running average metric object.
        """
        self.__samples_counter = 0
        self.__metric_accumulator = 0.0

    def __call__(self, metric: float, num_samples: int):
        """
        Accumulate a metric value.

        :param metric: The metric value.
        :param num_samples: The number of samples from which the metric is estimated.
        :raises ValueError: If the number of samples is not positive.
        """
        if num_samples <= 0:
            raise ValueError("The number of samples must be positive")
        self.__samples_counter += num_samples
        self.__metric_accumulator += metric * num_samples

    def reset(self):
        """
        Reset the running average metric accumulator.
        """
        self.__samples_counter = 0
        self.__metric_accumulator = 0.0

    def average(self) -> float:
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.__metric_accumulator / self.__samples_counter


def fid_score(
    dataset1: Union[data.Dataset, torch.Tensor],
    dataset2: Union[data.Dataset, torch.Tensor],
    model: Optional[nn.Module] = None,
    transform: Optional[Any] = None,
    batch_size: int = 100,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> float:
    """
    Compute the Frechet Inception Distance (FID) between two data samples.
    This implementation has been readapted from https://github.com/mseitzer/pytorch-fid.
    IMPORTANT NOTE: The computed FID score is not comparable with other FID scores based on Tensorflow's InceptionV3.

    :param dataset1: The first samples data set.
    :param dataset2: The second samples data set.
    :param model: The model to use to extract the features.
                  If None the Torchvision's InceptionV3 model pretrained on ImageNet will be used.
    :param transform: An optional transformation to apply to every sample. If transform and model are both None,
                      then the transformation resizes to 3x299x299 and normalizes values from (0, 1) to (-1, 1).
    :param batch_size: The batch size to use when extracting features.
    :param num_workers: The number of workers used for the data loaders.
    :param device: The device used to run the model. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :return The FID score.
    """
    if model is None:
        # Load the InceptionV3 model pretrained on ImageNet
        model = models.inception_v3(pretrained=True, aux_logits=False, transform_input=False)

        # Remove dropout and fully-connected layers (we are interested in extracted features)
        model.dropout = nn.Identity()
        model.fc = nn.Identity()

        # Set the transformation
        if transform is None:
            transform = transforms.Compose([
                transforms.Lambda(lambda x: F.interpolate(x, (299, 299), mode='bilinear', align_corners=False)),
                transforms.Normalize((0.5,), (0.5,))
            ])

    # Extract the features of the two data sets
    features1 = extract_features(
        model, dataset1, transform, device=device, verbose=verbose,
        batch_size=batch_size, num_workers=num_workers
    )
    features2 = extract_features(
        model, dataset2, transform, device=device, verbose=verbose,
        batch_size=batch_size, num_workers=num_workers
    )

    # Compute the statistics (mean and covariance of the features)
    features1, features2 = features1.cpu().numpy(), features2.cpu().numpy()
    mean1, cov1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mean2, cov2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Compute and return the FID score
    return compute_fid(mean1, cov1, mean2, cov2)


def extract_features(
    model: nn.Module,
    dataset: Union[data.Dataset, torch.Tensor],
    transform: Optional[Any] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Extract the features produced by a model using a data set.

    :param model: The model to use to extract the features.
    :param dataset: The data set.
    :param transform: An optional transformation to apply to every sample.
    :param device: The device used to run the model. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :param kwargs: Additional parameters to pass to the data loader.
    :return: The extracted features for each data sample.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extract features using device: {}".format(device))

    # Instantiate the data loader
    data_loader = data.DataLoader(dataset, **kwargs)
    if verbose:
        data_loader = tqdm(
            data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}', unit='batch'
        )

    # Make sure the model is in evaluation mode
    # Moreover, move it to the desired device
    model.eval()
    model.to(device)

    # Extract the features
    with torch.no_grad():
        features = list()
        for batch in data_loader:
            if transform is not None:
                batch = transform(batch)
            batch = batch.to(device)
            batch_features = model(batch)
            features.append(batch_features)
    return torch.cat(features, dim=0)
