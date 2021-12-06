# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import os
import time
from typing import Union, Optional, Tuple, Dict

import numpy as np
import torch
from torch import optim
from torch.utils import data
from tqdm import tqdm
from sklearn import metrics

from deeprob.torch.base import ProbabilisticModel
from deeprob.torch.utils import get_optimizer_class
from deeprob.torch.callbacks import EarlyStopping
from deeprob.torch.metrics import RunningAverageMetric
from deeprob.flows.models.base import NormalizingFlow


def train_model(
    model: ProbabilisticModel,
    data_train: Union[np.ndarray, data.Dataset],
    data_valid: Union[np.ndarray, data.Dataset],
    setting: str,
    lr: float = 1e-3,
    batch_size: int = 100,
    epochs: int = 1000,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[dict] = None,
    patience: int = 20,
    checkpoint: Union[os.PathLike, str] = 'checkpoint.pt',
    train_base: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Union[Dict[str, list], Dict[str, Dict[str, list]]]:
    """
    Train a Torch model.

    :param model: The model to train.
    :param data_train: The train dataset.
    :param data_valid: The validation dataset.
    :param setting: The train setting. It can be either 'generative' or 'discriminative'.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param optimizer: The optimizer to use.
    :param optimizer_kwargs: A dictionary containing additional optimizer parameters.
    :param patience: The epochs patience for early stopping.
    :param checkpoint: The checkpoint filepath used for early stopping.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param drop_last: Whether to drop the last train data batch having size less than the specified batch size.
    :param num_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :return: The train history.
    :raises ValueError: If a parameter is out of domain.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Train using device: {}".format(device))

    # Setup the data loaders
    train_loader = data.DataLoader(data_train, batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers)
    valid_loader = data.DataLoader(data_valid, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # Move the model to device
    model.to(device)

    # Instantiate the optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = get_optimizer_class(optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, **optimizer_kwargs
    )

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(model, patience=patience, filepath=checkpoint)

    # Train the model
    if setting == 'generative':
        return train_generative(
            model, train_loader, valid_loader, optimizer, device,
            early_stopping, epochs, train_base, verbose
        )
    if setting == 'discriminative':
        return train_discriminative(
            model, train_loader, valid_loader, optimizer, device,
            early_stopping, epochs, train_base, verbose
        )
    raise ValueError("Unknown train setting called {}".format(setting))


def train_generative(
    model: ProbabilisticModel,
    train_loader: data.DataLoader,
    valid_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    early_stopping: EarlyStopping,
    epochs: int = 1000,
    train_base: bool = True,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train a Torch model in generative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param valid_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param device: The device to use for training.
    :param epochs: The number of epochs.
    :param early_stopping: The early stopping callback object.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param verbose: Whether to enable verbose mode.
    :return: The train history with keys 'train' and 'validation'.
    :raises ValueError: If a parameter is out of domain.
    """
    if epochs <= 0:
        raise ValueError("The number of epochs must be positve")

    # Instantiate the train history
    history = {'train': [], 'valid': []}

    # Instantiate the running average metrics
    running_train_loss = RunningAverageMetric()
    running_valid_loss = RunningAverageMetric()

    for epoch in range(1, epochs + 1):
        # Reset the metrics
        running_train_loss.reset()
        running_valid_loss.reset()

        # Get the starting time
        start_time = time.perf_counter()

        # Wrap the train loader in a tqdm bar, if specified
        if verbose:
            data_loader = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Train Epoch {}/{}'.format(epoch, epochs), unit='batch'
            )
        else:
            data_loader = train_loader

        # Make sure the model is set to train mode
        if isinstance(model, NormalizingFlow):
            model.train(base_mode=train_base)
        else:
            model.train()

        # Training phase
        for inputs in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs)
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            running_train_loss(loss.item(), num_samples=inputs.shape[0])

        # Wrap the validation loader in a tqdm bar, if specified
        if verbose:
            data_loader = tqdm(
                valid_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Valid Epoch {}/{}'.format(epoch, epochs), unit='batch'
            )
        else:
            data_loader = valid_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = model.loss(outputs)
                running_valid_loss(loss.item(), num_samples=inputs.shape[0])

        # Compute the elapsed time per epoch
        elapsed_time = int(time.perf_counter() - start_time)

        # Get the average train and validation losses and print it
        train_loss = running_train_loss.average()
        valid_loss = running_valid_loss.average()
        print("Epoch {}/{} - train_loss: {:.4f}, valid_loss: {:.4f} [{}s]".format(
            epoch, epochs, train_loss, valid_loss, elapsed_time if elapsed_time > 0 else '<1'
        ))

        # Append losses to history data
        history['train'].append(train_loss)
        history['valid'].append(valid_loss)

        # Check if training should stop according to early stopping
        early_stopping(valid_loss, epoch)
        if early_stopping.should_stop:
            print("Early Stopping... {}".format(early_stopping))
            break

    # Load the best parameters state according to early stopping
    model.load_state_dict(early_stopping.get_best_state())
    return history


def train_discriminative(
    model: ProbabilisticModel,
    train_loader: data.DataLoader,
    valid_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    early_stopping: EarlyStopping,
    epochs: int = 1000,
    train_base: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, list]]:
    """
    Train a Torch model in discriminative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param valid_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param device: The device to use for training.
    :param early_stopping: The early stopping callback object.
    :param epochs: The number of epochs.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param verbose: Whether to enable verbose mode.
    :return: The train history with keys 'train' and 'validation' and for both keys 'loss' and 'accuracy'.
    :raises ValueError: If a parameter is out of domain.
    """
    if epochs <= 0:
        raise ValueError("The number of epochs must be positve")

    # Instantiate the train history
    history = {
        'train': {'loss': [], 'accuracy': []},
        'valid': {'loss': [], 'accuracy': []}
    }

    # Instantiate the running average metrics
    running_train_loss = RunningAverageMetric()
    running_train_hits = RunningAverageMetric()
    running_valid_loss = RunningAverageMetric()
    running_valid_hits = RunningAverageMetric()

    for epoch in range(1, epochs + 1):
        # Reset the metrics
        running_train_loss.reset()
        running_train_hits.reset()
        running_valid_loss.reset()
        running_valid_hits.reset()

        # Get the starting time
        start_time = time.perf_counter()

        # Wrap the train loader in a tqdm bar, if specified
        if verbose:
            data_loader = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Train Epoch {}/{}'.format(epoch, epochs), unit='batch'
            )
        else:
            data_loader = train_loader

        # Make sure the model is set to train mode
        if isinstance(model, NormalizingFlow):
            model.train(base_mode=train_base)
        else:
            model.train()

        # Training phase
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, y=targets)
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            running_train_loss(loss.item(), num_samples=inputs.shape[0])
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).float().mean()
                running_train_hits(hits.item(), num_samples=inputs.shape[0])

        # Wrap the validation loader in a tqdm bar, if specified
        if verbose:
            data_loader = tqdm(
                valid_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Valid Epoch {}/{}'.format(epoch, epochs), unit='batch'
            )
        else:
            data_loader = valid_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = model.loss(outputs, y=targets)
                running_valid_loss(loss.item(), num_samples=inputs.shape[0])
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).float().mean()
                running_valid_hits(hits.item(), num_samples=inputs.shape[0])

        # Compute the elapsed time per epoch
        elapsed_time = int(time.perf_counter() - start_time)

        # Get the average train and validation losses and accuracies and print it
        train_loss = running_train_loss.average()
        train_acc = running_train_hits.average()
        valid_loss = running_valid_loss.average()
        valid_acc = running_valid_hits.average()
        print("Epoch {}/{} - train_loss: {:.4f}, valid_loss: {:.4f}, ".format(
            epoch, epochs, train_loss, valid_loss
        ), end="")
        print("train_acc: {:.1f}%, valid_acc: {:.1f}% [{}s]".format(
            train_acc * 100, valid_acc * 100, elapsed_time if elapsed_time > 0 else '<1'
        ))

        # Append losses and accuracies to history data
        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_acc)
        history['valid']['loss'].append(valid_loss)
        history['valid']['accuracy'].append(valid_acc)

        # Check if training should stop according to early stopping
        early_stopping(valid_loss, epoch)
        if early_stopping.should_stop:
            print("Early Stopping... {}".format(early_stopping))
            break

    # Load the best parameters state according to early stopping
    model.load_state_dict(early_stopping.get_best_state())
    return history


def test_model(
    model: ProbabilisticModel,
    data_test: Union[np.ndarray, data.Dataset],
    setting: str,
    batch_size: int = 100,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Union[Tuple[float, float], Tuple[float, dict]]:
    """
    Test a Torch model.

    :param model: The model to test.
    :param data_test: The test dataset.
    :param setting: The test setting. It can be either 'generative' or 'discriminative'.
    :param batch_size: The batch size for testing.
    :param num_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :return: The mean log-likelihood and two standard deviations if setting='generative'.
             The negative log-likelihood and classification metrics if setting='discriminative'.
    :raises ValueError: If a parameter is out of domain.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: {}'.format(device))

    # Setup the data loader
    test_loader = data.DataLoader(data_test, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # Move the model to device
    model.to(device)

    # Test the model
    if setting == 'generative':
        return test_generative(model, test_loader, device, verbose)
    if setting == 'discriminative':
        return test_discriminative(model, test_loader, device, verbose)
    raise ValueError("Unknown test setting called {}".format(setting))


def test_generative(
    model: ProbabilisticModel,
    test_loader: data.DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Test a Torch model in generative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :param verbose: Whether to enable verbose mode.
    :return: The mean log-likelihood and two standard deviations.
    """
    # Wrap the test loader in a tqdm bar, if specified
    if verbose:
        data_loader = tqdm(
            test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
            desc='Test', unit='batch'
        )
    else:
        data_loader = test_loader

    # Make sure the model is set to evaluation mode
    model.eval()

    test_lls = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            ll = model(inputs).cpu().tolist()
            test_lls.extend(ll)
    mean_ll = np.mean(test_lls)
    stddev_ll = 2.0 * np.std(test_lls) / np.sqrt(len(test_lls))
    return mean_ll.item(), stddev_ll.item()


def test_discriminative(
    model: ProbabilisticModel,
    test_loader: data.DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, dict]:
    """
    Test a Torch model in discriminative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :param verbose: Whether to enable verbose mode.
    :return: The negative log-likelihood and classification report dictionary.
    """
    # Wrap the test loader in a tqdm bar, if specified
    if verbose:
        data_loader = tqdm(
            test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
            desc='Test', unit='batch'
        )
    else:
        data_loader = test_loader

    # Make sure the model is set to evaluation mode
    model.eval()

    y_true = []
    y_pred = []
    running_loss = RunningAverageMetric()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.loss(outputs, y=targets)
            running_loss(loss.item(), num_samples=inputs.shape[0])
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
    return running_loss.average(), metrics.classification_report(y_true, y_pred, output_dict=True)
