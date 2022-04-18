"""
Author: Smeet Shah

Description: Function list to check working of different modules.

- One check function for each module (functions/classes)
- Random inputs to check working
"""

import random

import torch
import numpy as np

from config import args
from models import MyNet
from utils import MyDataset
from essentials.pprocs import preprocess_sample
from essentials.losses import MyLoss, L2Regularizer
from essentials.decoders import decode
from essentials.metrics import compute_metric


def check_mynet():
    """
    Check function for MyNet()
    """
    return


def check_mydataset():
    """
    Check function for MyDataset()
    """
    return


def check_decode():
    """
    Check function for decode()
    """
    return


def check_compute_metric():
    """
    Check function for compute_metric()
    """
    return


def check_preprocess_sample():
    """
    Check function for preprocess_sample()
    """
    return


def check_myloss():
    """
    Check function for MyLoss()
    """
    return


def check_l2regularizer():
    """
    Check function for L2Regularizer()
    """
    return


if __name__ == "__main__":
    check_preprocess_sample()
    check_mydataset()
    check_myloss()
    check_mynet()
    check_l2regularizer()
    check_decode()
    check_compute_metric()
