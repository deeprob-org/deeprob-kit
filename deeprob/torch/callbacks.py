# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import os
from typing import Union
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class EarlyStopping:
    def __init__(
        self,
        model: nn.Module,
        patience: int = 1,
        filepath: Union[os.PathLike, str] = 'checkpoint.pt',
        delta: float = 1e-3
    ):
        """
        Early stops the training if validation loss doesn't improve after a given number of consecutive epochs.

        :param model: The model to monitor.
        :param patience: The number of consecutive epochs to wait.
        :param filepath: The checkpoint filepath where to save the model state dictionary.
        :param delta: The minimum change of the monitored quantity.
        :raises ValueError: If the patience or delta values are out of domain.
        """
        if patience <= 0:
            raise ValueError("The patience value must be positive")
        if delta <= 0.0:
            raise ValueError("The delta value must be positive")
        self.model = model
        self.patience = patience
        self.filepath = filepath
        self.delta = delta
        self.__best_loss = np.inf
        self.__best_epoch = None
        self.__counter = 0

    @property
    def should_stop(self) -> bool:
        """
        Check if the training process should stop.
        """
        return self.__counter >= self.patience

    def get_best_state(self) -> OrderedDict:
        """
        Get the best model's state dictionary.
        """
        with open(self.filepath, 'rb') as f:
            best_state = torch.load(f)
        return best_state

    def __call__(self, loss: float, epoch: int):
        """
        Update the state of early stopping.

        :param loss: The validation loss measured.
        :param epoch: The current epoch.
        """
        # Check if an __best_loss of the loss happened
        if loss < self.__best_loss - self.delta:
            self.__best_loss = loss
            self.__best_epoch = epoch
            self.__counter = 0

            # Save the best model state parameters
            with open(self.filepath, 'wb') as f:
                torch.save(self.model.state_dict(), f)
        else:
            self.__counter += 1

    def __format__(self, format_spec) -> str:
        return "Best Loss: {:.4f} at Epoch: {}".format(self.__best_loss, self.__best_epoch)
