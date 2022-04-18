"""
Author: Smeet Shah

Description: Custom dataset class definitions.

- MyDataset class definition
"""

import sys

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import torch


class CSVDataset(Dataset):
    """
    Documentation: description, input, output
    """

    def __init__(self, dataset, data_directroy, data_filename):
        super(CSVDataset, self).__init__()
        self.dataset = dataset

        data_filepath = os.path.join(data_directroy, data_filename)
        df = pd.read_csv(data_filepath, header=None)
        x_data = df.values[:, :-1]
        y_data = df.values[:, [-1]]
        self.x_dataset = torch.FloatTensor(x_data)
        self.y_dataset = torch.FloatTensor(y_data)

    def __getitem__(self, idx):
        x_data, y_data = self.x_dataset[idx], self.y_dataset[idx]
        return x_data, y_data

    def __len__(self):
        return len(self.y_dataset)
