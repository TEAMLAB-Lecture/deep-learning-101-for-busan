"""
Author: Smeet Shah

Description: Set of tools for data processing.

- prepare_input(): Data sample to tensor
"""

import sys

import torch
import numpy as np


def prepare_input(file):

    """
    Documentation: description, input, output
    """

    inp, trgt = file
    return inp, trgt
