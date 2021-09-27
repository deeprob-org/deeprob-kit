import os
import csv
import h5py
import torch
import torchvision
import numpy as np
import pandas as pd

from collections import Counter
from typing import Optional, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from deeprob.utils.random import RandomState, check_random_state
from deeprob.torch.transforms import TransformList
from deeprob.torch.transforms import Flatten, Normalize, RandomHorizontalFlip
from deeprob.torch.datasets import UnsupervisedDataset, SupervisedDataset, WrappedDataset

#: A list of 29 binary datasets names (sorted by number of features).
BINARY_DATASETS = [
    'nltcs',
    'msnbc',
    'kdd',
    'plants',
    'baudio',
    'jester',
    'bnetflix',
    'accidents',
    'mushrooms',
    'adult',
    'connect4',
    'ocr_letters',
    'rcv1',
    'tretail',
    'pumsb_star',
    'dna',
    'kosarek',
    'msweb',
    'nips',
    'book',
    'tmovie',
    'cwebkb',
    'cr52',
    'c20ng',
    'moviereview',
    'bbc',
    'voting',
    'ad',
    'binarized_mnist'
]

#: A list of 5 continuous datasets names (well known in Normalizing Flows papers).
CONTINUOUS_DATASETS = [
    'power',
    'gas',
    'hepmass',
    'miniboone',
    'BSDS300'
]

#: Computer Vision datasets, also suitable for classification tasks.
VISION_DATASETS = [
    'mnist',
    'cifar10',
    'olivetti-faces'
]


def csv_to_numpy(filepath: str, sep: str = ',', dtype=np.uint8) -> np.ndarray:
    """
    Read a CSV and convert it into a Numpy array.

    :param filepath: The CSV filepath.
    :param sep: The CSV separator string.
    :param dtype: The output ndarray data type.
    :return: The CSV read into a Numpy array.
    """
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        dataset = np.array(list(reader), dtype=dtype)
        return dataset


def load_binary_dataset(
    root: str,
    name: str,
    raw: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[UnsupervisedDataset, UnsupervisedDataset, UnsupervisedDataset]
]:
    """
    Load a binary dataset.

    :param root: The datasets root directory.
    :param name: The name of the dataset.
    :param raw: Whether to return Numpy arrays instead of Torch Datasets.
    :return: The train, validation and test dataset splits.
    """
    # binarized_mnist CSV have a whitespace separator for some reason
    sep = ' ' if name == 'binarized_mnist' else ','

    # Load the CSV files to Numpy arrays
    directory = os.path.join(root, name)
    data_train = csv_to_numpy(os.path.join(directory, name + '.train.data'), sep=sep)
    data_valid = csv_to_numpy(os.path.join(directory, name + '.valid.data'), sep=sep)
    data_test = csv_to_numpy(os.path.join(directory, name + '.test.data'), sep=sep)

    # Return raw Numpy arrays, if specified
    if raw:
        return data_train, data_valid, data_test

    # Wrap and return the datasets
    data_train = UnsupervisedDataset(data_train)
    data_valid = UnsupervisedDataset(data_valid)
    data_test = UnsupervisedDataset(data_test)
    return data_train, data_valid, data_test


def load_continuous_dataset(
    root: str,
    name: str,
    raw: bool = False,
    random_state: Optional[RandomState] = None
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[UnsupervisedDataset, UnsupervisedDataset, UnsupervisedDataset]
]:
    """
    Load a continuous dataset.
    All the datasets are preprocessed as in the original MAF paper repository.
    See https://github.com/gpapamak/maf/tree/master/datasets for details.

    :param root: The datasets root directory.
    :param name: The name of the dataset.
    :param raw: Whether to return unpreprocessed Numpy arrays instead of Torch Datasets.
                Torch Datasets will have standardization as data transformation.
    :param random_state: The random state to use for shuffling and transforming the data.
                         It can be either None, a seed integer or a Numpy RandomState.
    :return: The train, validation and test dataset splits.
    :raise ValueError: If the continuous dataset name is not known.
    """
    # Check the random state
    random_state = check_random_state(random_state)

    directory = os.path.join(root, name)
    if name == 'power':
        # Load the dataset
        data = np.load(os.path.join(directory, 'data.npy'))
        random_state.shuffle(data)
        n_samples = len(data)
        data = np.delete(data, [1, 3], axis=1)

        # Add noise as in original datasets preprocessing (MAF paper)
        voltage_noise = 0.01 * random_state.rand(n_samples, 1)
        gap_noise = 0.001 * random_state.rand(n_samples, 1)
        sm_noise = random_state.rand(n_samples, 3)
        time_noise = np.zeros(shape=(n_samples, 1))
        data = data + np.hstack([gap_noise, voltage_noise, sm_noise, time_noise])

        # Split the dataset
        n_test = int(0.1 * len(data))
        data_test = data[-n_test:]
        data = data[:-n_test]
        n_valid = int(0.1 * len(data))
        data_valid = data[-n_valid:]
        data_train = data[:-n_valid]
    elif name == 'gas':
        # Load the dataset
        data = pd.read_pickle(os.path.join(directory, 'ethylene_CO.pickle'))
        data.drop(['Meth', 'Eth', 'Time'], axis=1, inplace=True)

        # Remove uninformative features
        uninformative_idx = (data.corr() > 0.98).to_numpy().sum(axis=1)
        while np.any(uninformative_idx > 1):
            col_to_remove = np.where(uninformative_idx > 1)[0][0]
            data.drop(data.columns[col_to_remove], axis=1, inplace=True)
            uninformative_idx = (data.corr() > 0.98).to_numpy().sum(axis=1)
        data = data.to_numpy()
        random_state.shuffle(data)

        # Split the dataset
        n_test = int(0.1 * len(data))
        data_test = data[-n_test:]
        data = data[:-n_test]
        n_valid = int(0.1 * len(data))
        data_valid = data[-n_valid:]
        data_train = data[:-n_valid]
    elif name == 'hepmass':
        # Load the dataset
        data_train = pd.read_csv(os.path.join(directory, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(os.path.join(directory, "1000_test.csv"), index_col=False)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        data_test = data_test.drop(data_test.columns[-1], axis=1)
        data_train, data_test = data_train.to_numpy(), data_test.to_numpy()

        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for i, feature in enumerate(data_train.T):
            c = Counter(feature)
            max_count = next(v for k, v in sorted(c.items()))
            if max_count > 5:
                features_to_remove.append(i)
        features_to_keep = [i for i in range(data_train.shape[1]) if i not in features_to_remove]
        data_train = data_train[:, features_to_keep]
        data_test = data_test[:, features_to_keep]
        random_state.shuffle(data_train)

        # Split the train dataset
        n_valid = int(len(data_train) * 0.1)
        data_valid = data_train[-n_valid:]
        data_train = data_train[:-n_valid]
    elif name == 'miniboone':
        # Load the dataset
        data = np.load(os.path.join(directory, 'data.npy'))
        random_state.shuffle(data)

        # Split the dataset
        n_test = int(0.1 * len(data))
        data_test = data[-n_test:]
        data = data[:-n_test]
        n_valid = int(0.1 * len(data))
        data_valid = data[-n_valid:]
        data_train = data[:-n_valid]
    elif name == 'BSDS300':
        # Load the dataset
        with h5py.File(os.path.join(directory, 'BSDS300.hdf5'), 'r') as file:
            data_train = file['train'][:]
            data_valid = file['validation'][:]
            data_test = file['test'][:]
    else:
        raise ValueError("Unknown continuous dataset called {}".format(name))

    # Return raw Numpy arrays, if specified
    if raw:
        return data_train, data_valid, data_test

    # Instantiate the standardize transform
    mean = torch.tensor(np.mean(data_train, axis=0), dtype=torch.float32)
    std = torch.tensor(np.std(data_train, axis=0), dtype=torch.float32)
    transform = Normalize(mean, std)

    # Wrap and return the datasets
    data_train = UnsupervisedDataset(data_train, transform)
    data_valid = UnsupervisedDataset(data_valid, transform)
    data_test = UnsupervisedDataset(data_test, transform)
    return data_train, data_valid, data_test


def load_vision_dataset(
    root: str,
    name: str,
    unsupervised: bool = True,
    standardize: bool = False,
    flatten: bool = True,
    random_hflip: bool = False,
    random_state: Optional[RandomState] = None
) -> Tuple[WrappedDataset, WrappedDataset, WrappedDataset]:
    """
    Load a computer vision dataset.

    :param root: the datasets root directory.
    :param name: The name of the dataset.
    :param unsupervised: Whether to load the unsupervised version (i.e. without labels).
    :param standardize: Whether to standardize the image dataset.
    :param flatten: Whether to flatten the image features.
    :param random_hflip: Whether to apply a random horizontal flip transformation.
    :param random_state: The random state to use for shuffling and splitting.
    :return: The train, validation and test dataset splits.
    :raise ValueError: If the vision dataset name is not known.
    :raises ValueError: If the optional transformation (preproc) to apply is out of domain.
    """
    # Check the random state
    random_state = check_random_state(random_state)

    # Instantiate the ToTensor() transform
    transform = torchvision.transforms.ToTensor()

    # Load the vision train and test datasets
    if name == 'mnist':
        data_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
        data_test = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
        classes = list(range(10))
        data_mean, data_std = (0.1307,), (0.3081,)

        # Split the train dataset
        data_full = data_train
        train_idx, valid_idx = train_test_split(
            np.arange(len(data_full)), test_size=1.0 / 6.0, random_state=random_state,
            shuffle=True, stratify=data_full.targets if not unsupervised else None
        )
        data_train = torch.utils.data.Subset(data_full, train_idx)
        data_valid = torch.utils.data.Subset(data_full, valid_idx)
    elif name == 'cifar10':
        data_train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
        data_test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
        classes = list(range(10))
        data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

        # Split the train dataset
        data_full = data_train
        train_idx, valid_idx = train_test_split(
            np.arange(len(data_full)), test_size=0.1, random_state=random_state,
            shuffle=True, stratify=data_full.targets if not unsupervised else None
        )
        data_train = torch.utils.data.Subset(data_full, train_idx)
        data_valid = torch.utils.data.Subset(data_full, valid_idx)
    elif name == 'olivetti-faces':
        if not unsupervised:
            raise ValueError("Olivetti-Faces dataset can be used only in unsupervised mode")

        # Fetch the dataset
        data, targets = fetch_openml(data_id=41083, data_home=root, return_X_y=True, as_frame=True)
        data, targets = data.to_numpy().astype(np.float32), targets.to_numpy().astype(np.int64)
        data = data.reshape([400, 1, 64, 64])
        unique_targets = np.unique(targets)
        classes = list(range(len(unique_targets)))

        # Obtain the test set according to the subjects ids
        n_test = int(0.1 * len(unique_targets))
        test_targets = random_state.choice(unique_targets, size=n_test, replace=False)
        mask = np.logical_or.reduce(targets == test_targets[:, np.newaxis], axis=0)
        data_train, data_test = data[~mask], data[mask]
        targets_train, targets_test = targets[~mask], targets[mask]

        # Compute mean and standard deviation
        data_mean, data_std = (np.mean(data_train).item(),), (np.std(data_train).item(),)

        # Split the train dataset furthermore to obtain the validation set
        n_val = int(0.1 * len(unique_targets))
        rest_unique_targets = np.array([t for t in unique_targets if t not in test_targets])
        valid_targets = random_state.choice(rest_unique_targets, size=n_val, replace=False)
        mask = np.logical_or.reduce(targets_train == valid_targets[:, np.newaxis], axis=0)
        data_train, data_valid = data_train[~mask], data_train[mask]
        targets_train, targets_valid = targets_train[~mask], targets_train[mask]

        # Instantiate the supervised datasets
        data_train = SupervisedDataset(data_train, targets_train)
        data_valid = SupervisedDataset(data_valid, targets_valid)
        data_test = SupervisedDataset(data_test, targets_test)
    else:
        raise ValueError("Unknown vision dataset called {}".format(name))

    # Build the transforms
    transform_train = TransformList()
    transform_test = TransformList()
    if standardize:
        mean = torch.tensor(data_mean, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        std = torch.tensor(data_std, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        normalize = Normalize(mean, std)
        transform_train.append(normalize)
        transform_test.append(normalize)

    if random_hflip:
        # Append random horizontal flip transformation
        transform_train.append(RandomHorizontalFlip())

    if flatten:
        # Append flatten transformation
        shape = data_train[0][0].shape
        transform_train.append(Flatten(shape))
        transform_test.append(Flatten(shape))

    # Prevent empty transform lists
    if not transform_train:
        transform_train = None
    if not transform_test:
        transform_test = None

    if unsupervised:
        # Instantiate and return unsupervised wrappers
        data_train = WrappedDataset(data_train, unsupervised=True, transform=transform_train)
        data_valid = WrappedDataset(data_valid, unsupervised=True, transform=transform_train)
        data_test = WrappedDataset(data_test, unsupervised=True, transform=transform_test)
        return data_train, data_valid, data_test

    # Instantiate and return supervised wrappers
    data_train = WrappedDataset(data_train, unsupervised=False, classes=classes, transform=transform_train)
    data_valid = WrappedDataset(data_valid, unsupervised=False, classes=classes, transform=transform_train)
    data_test = WrappedDataset(data_test, unsupervised=False, classes=classes, transform=transform_test)
    return data_train, data_valid, data_test
