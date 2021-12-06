# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from typing import Union, Optional, Tuple, List

import numpy as np
import torch
from torch.utils import data

from deeprob.torch.transforms import Transform


class UnsupervisedDataset(data.Dataset):
    def __init__(
        self,
        dataset: Union[np.ndarray, torch.Tensor],
        transform: Optional[Union[Transform]] = None
    ):
        """
        Initialize an unsupervised dataset.

        :param dataset: The dataset.
        :param transform: An optional transformation to apply.
        """
        super().__init__()
        if isinstance(dataset, np.ndarray):
            dataset = torch.tensor(dataset, dtype=torch.float32)
        elif dataset.dtype != torch.float32:
            dataset = dataset.float()

        self.dataset = dataset
        self.transform = transform

        # Compute the features shape
        if self.transform is None:
            shape = tuple(self.dataset.shape[1:])
        else:
            shape = tuple(self.transform(self.dataset[0]).shape)
        self.shape = shape[0] if len(shape) == 1 else shape

    @property
    def features_shape(self) -> Union[int, tuple]:
        """Get the dataset features shape."""
        return self.shape

    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.dataset)

    def __getitem__(self, i) -> torch.Tensor:
        """
        Retrive the example at a specified index.

        :param i: The index of the example.
        :return: The example features.
        """
        x = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        return x


class SupervisedDataset(data.Dataset):
    def __init__(
        self,
        dataset: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        transform: Optional[Union[Transform]] = None
    ):
        """
        Initialize a supervised dataset.

        :param dataset: The dataset.
        :param targets: The targets.
        :param transform: An optional transformation to apply.
        """
        super().__init__()
        if isinstance(dataset, np.ndarray):
            dataset = torch.tensor(dataset, dtype=torch.float32)
        elif dataset.dtype != torch.float32:
            dataset = dataset.float()
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.int64)
        elif targets.dtype != torch.int64:
            targets = targets.long()

        self.dataset = dataset
        self.targets = targets
        self.transform = transform

        # Compute the features shape
        if self.transform is None:
            shape = tuple(self.dataset[0].shape)
        else:
            shape = tuple(self.transform(self.dataset[0]).shape)
        self.shape = shape[0] if len(shape) == 1 else shape

        # Obtain the classes
        self.classes = torch.unique(self.targets).tolist()

    @property
    def features_shape(self) -> Union[int, tuple]:
        """Get the dataset features shape."""
        return self.shape

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.classes)

    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.dataset)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrive the example at a specified index.

        :param i: The index of the example.
        :return: The example features and the target.
        """
        x, y = self.dataset[i], self.targets[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class WrappedDataset(data.Dataset):
    def __init__(
        self,
        dataset: data.Dataset,
        unsupervised: bool = True,
        classes: Optional[List[int]] = None,
        transform: Optional[Union[Transform]] = None
    ):
        """
        Initialize a wrapped dataset (either unsupervised or supervised).

        :param dataset: The dataset (assumed to be supervised).
        :param unsupervised: Whether to treat the dataset as unsupervised.
        :param classes: The class domain. It can be None if unsupervised is True.
        :param transform: An optional transformation to apply.
        """
        super().__init__()
        self.dataset = dataset
        self.unsupervised = unsupervised
        self.transform = transform

        # Compute the features shape
        if self.transform is None:
            shape = tuple(self.dataset[0][0].shape)
        else:
            shape = tuple(self.transform(self.dataset[0][0]).shape)
        self.shape = shape[0] if len(shape) == 1 else shape

        # Set the classes
        self.classes = [0] if classes is None else classes

    @property
    def features_shape(self) -> Union[int, tuple]:
        """Get the dataset features shape."""
        return self.shape

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.classes)

    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.dataset)

    def __getitem__(self, i) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrive the example at a specified index.

        :param i: The index of the example.
        :return: If unsupervised is False, then the pair of example features and target.
                 If unsupervised is True, then the example features only.
        """
        x, y = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        if self.unsupervised:
            return x
        return x, y
