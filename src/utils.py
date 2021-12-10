"""Utility functions and classes"""

import errno
import json
import logging
import os
import shutil

import torch


class Params:
    """Class to load hyperparameters from a json file."""

    def __init__(self, json_path):
        with open(json_path, encoding="utf-8") as fptr:
            params = json.load(fptr)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters to json file at json_path"""
        with open(json_path, "w", encoding="utf-8") as fptr:
            json.dump(self.__dict__, fptr, indent=4)

    def update(self, json_path):
        """Loads parameters from json file at json_path"""
        with open(json_path, encoding="utf-8") as fptr:
            params = json.load(fptr)
            self.__dict__.update(params)


class RunningAverage:
    """Maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() == 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        """Update the running average"""
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: (string) location of log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(data, json_path):
    """Saves a dict of floats to json file
    Args:
        data: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w", encoding="utf-8") as fptr:
        data = {k: float(v) for k, v in data.items()}
        json.dump(data, fptr, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model at checkpoint
    Args:
        state: (dict) contains model's state_dict, epoch, optimizer state_dict etc.
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    safe_makedir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model state_dict from checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint


def safe_makedir(path):
    """Make directory given the path if it doesn't already exist
    Args:
        path: path of the directory to be made
    """
    if not os.path.exists(path):
        print(f"Directory doesn't exist! Making directory {path}.")
        os.makedirs(path)
    else:
        print(f"Directory {path} Exists!")
