"""
Author: Smeet Shah

Description: All essential functions.

- num_params(): Number of total parameters and trainable parameters
- train(): Train the model for one epoch
- evaluate(): Evaluate model performance over validation/test set
"""

import sys

import torch
import numpy as np
from tqdm import tqdm

from .decoders import decode
from .metrics import compute_metric


def num_params(model):

    """
    Function that outputs the number of total and trainable paramters in the model.
    """

    numTotalParams = sum([p.numel() for p in model.parameters()])
    numTrainableParams = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    return numTotalParams, numTrainableParams


def train(model, trainLoader, optimizer, criterion, device):

    """
    Function to train the model for one epoch.
    """

    trainingLoss = 0
    trainingMetric = 0
    number_of_data = 0
    for batch, (inputBatch, targetBatch) in enumerate(
        tqdm(trainLoader, leave=False, desc="Train", ncols=75)
    ):

        inputBatch, targetBatch = (
            (inputBatch.float()).to(device),
            targetBatch.to(device),
        )

        optimizer.zero_grad()
        model.train()
        outputBatch = model(inputBatch)
        loss = criterion(outputBatch, targetBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch = decode(outputBatch.detach())
        trainingMetric = trainingMetric + compute_metric(
            predictionBatch, targetBatch
        )
        number_of_data = number_of_data + batch

    trainingLoss = trainingLoss / number_of_data
    trainingMetric = trainingMetric / number_of_data
    return trainingLoss, trainingMetric


def evaluate(model, evalLoader, criterion, device):

    """
    Function to evaluate the model over validation/test set.
    """

    evalLoss = 0
    evalMetric = 0
    number_of_data = 0
    for batch, (inputBatch, targetBatch) in enumerate(
        tqdm(evalLoader, leave=False, desc="Eval", ncols=75)
    ):

        inputBatch, targetBatch = (
            (inputBatch.float()).to(device),
            targetBatch.to(device),
        )

        model.eval()
        with torch.no_grad():
            outputBatch = model(inputBatch)
            loss = criterion(outputBatch, targetBatch)

        evalLoss = evalLoss + loss.item()
        predictionBatch = decode(outputBatch)
        evalMetric = evalMetric + compute_metric(predictionBatch, targetBatch)
        number_of_data += len(inputBatch)
    evalLoss = evalLoss / number_of_data
    evalMetric = evalMetric / number_of_data
    return evalLoss, evalMetric
