"""
Author: Smeet Shah

Description: Python script to test the DL model.

- Loads the test dataset
- Loads the trained model weights
- Tests the model over the test set and prints the loss and metric values
"""

import random

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from config import args
from models import MyNet
from utils import MyDataset
from essentials import evaluate
from essentials.losses import MyLoss, L2Regularizer


def main():

    """
    Main function wrapper for testing script.
    """

    random.seed(args["SEED"])
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True}
    else:
        device = torch.device("cpu")
        kwargs = {}

    if args["TRAINED_WEIGHTS_FILE"] is not None:

        testData = MyDataset("test", datadir=args["DATA_DIRECTORY"])
        testLoader = DataLoader(
            testData, batch_size=args["BATCH_SIZE"], shuffle=True, **kwargs
        )

        print("Trained Weights File: %s" % (args["TRAINED_WEIGHTS_FILE"]))

        model = MyNet()
        model.load_state_dict(
            torch.load(
                args["CODE_DIRECTORY"] + args["TRAINED_WEIGHTS_FILE"],
                map_location=device,
            )
        )
        model.to(device)

        criterion = MyLoss()
        regularizer = L2Regularizer(lambd=args["LAMBDA"])

        print("Testing the trained model ....")

        testLoss, testMetric = evaluate(
            model, testLoader, criterion, regularizer, device
        )

        print(
            "| Test Loss: %.6f || Test Metric: %.3f |" % (testLoss, testMetric)
        )
        print("Testing Done.")

    else:
        print("Path to the trained weights file not specified.")

    return


if __name__ == "__main__":
    main()
