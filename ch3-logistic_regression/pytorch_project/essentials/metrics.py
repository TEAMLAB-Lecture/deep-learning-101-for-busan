import torch
import numpy as np



def compute_metric(y_pred, y_target):

    """
    Documentation: description, input, output
    """
    return torch.sum(y_pred == y_target)
