"""
Author: Smeet Shah

Description: DL model class definition.

- MyNet model architecture with forward pass function
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifierModel(nn.Module):

    """
    Documentation: Task, Architecture, Input, Output
    """

    def __init__(self):
        super(BinaryClassifierModel, self).__init__()
        super().__init__()
        self.linear = nn.Linear(8, 1)  # 8개의 element 받아서 0인지 1인지 예측하는 모델
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass function
        """
        return self.sigmoid(self.linear(x))  # 모델은 self.linear을 한 후 self.sigmoid를 하는 모델임을 정의
