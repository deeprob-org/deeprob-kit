from typing import Optional, Union

import torch
from torch import nn
from torch import hub
from torch.utils import data
from torchvision import transforms

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
    resizer: Optional[nn.Module] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute the Frechet Inception Distance (FID) between two data samples.

    :param dataset1: The first samples data set.
    :param dataset2: The second samples data set.
    :param model: The model to use to extract the features.
                  If None the Torchvision's InceptionV3 model pretrained on ImageNet will be used.
    :param resizer: The input images resizer. It can be None if no resizing must be applied.
                    If resizer is None and model is also None, then the resizer is instantiated
                    in order to resize to 3x299x299.
    :param batch_size: The batch size to use when extracting features.
    :param num_workers: The number of workers used for the data loaders.
    :param device: The device used to run the model. If it's None 'cuda' will be used, if available.
    :return The FID score.
    """
    if model is None:
        # Load the InceptionV3 model pretrained on ImageNet
        model = hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

        # Remove dropout and fully-connected layers (we are interested in features)
        model.dropout = nn.Identity()
        model.fc = nn.Identity()

        # Set the resizer
        if resizer is None:
            resizer = transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR)

    # Extract the features of the two data sets
    features1 = extract_features(
        model, dataset1, resizer,
        device=device, batch_size=batch_size, num_workers=num_workers
    )
    features2 = extract_features(
        model, dataset2, resizer,
        device=device, batch_size=batch_size, num_workers=num_workers
    )

    # Compute the statistics (mean and covariance of the features)
    mean1, cov1 = torch.mean(features1, dim=0), torch.cov(features1.T)
    mean2, cov2 = torch.mean(features2, dim=0), torch.cov(features2.T)

    # Compute and return the FID score
    mean1, cov1 = mean1.cpu().numpy(), cov1.cpu().numpy()
    mean2, cov2 = mean2.cpu().numpy(), cov2.cpu().numpy()
    return compute_fid(mean1, cov1, mean2, cov2)


def extract_features(
    model: nn.Module,
    dataset: data.Dataset,
    resizer: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> torch.Tensor:
    """
    Extract the features produced by a model using a data set.

    :param model: The model to use to extract the features.
    :param dataset: The data set.
    :param resizer: The input images resizer. It can be None if no resizing must be applied.
    :param device: The device used to run the model. If it's None 'cuda' will be used, if available.
    :param kwargs: Additional parameters to pass to the data loader.
    :return: The extracted features for each data sample.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extracting features using device: {}".format(device))

    # Instantiate the data loader
    data_loader = data.DataLoader(dataset, **kwargs)

    # Make sure the model is in evaluation mode
    # Moreover, move it to the desired device
    model.eval()
    model.to(device)

    # Extract the features
    with torch.no_grad():
        features = list()
        for batch in data_loader:
            if resizer is not None:
                batch = resizer(batch)
            batch = batch.to(device)
            batch_features = model(batch)
            features.append(batch_features)
    return torch.cat(features, dim=0)
