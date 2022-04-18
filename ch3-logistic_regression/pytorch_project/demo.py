"""
Author: Smeet Shah

Description: Python script to run a demo of the trained model on data samples.

- Loads the trained model weights
- Iterates over all data samples in demo directory
- Preprocesses each data sample and runs the model
- Decodes the model outputs and gives predictions
"""

import os
import random

import torch
import numpy as np

from config import args
from models import MyNet
from utils.tools import prepare_input
from essentials.pprocs import preprocess_sample
from essentials.decoders import decode


def main():

    """
    Main function wrapper for demo script
    """

    random.seed(args["SEED"])
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args["TRAINED_WEIGHTS_FILE"] is not None:

        print("Trained Weights File: %s" % (args["TRAINED_WEIGHTS_FILE"]))
        print("Demo Directory: %s" % (args["DEMO_DIRECTORY"]))

        model = MyNet()
        model.load_state_dict(
            torch.load(
                args["CODE_DIRECTORY"] + args["TRAINED_WEIGHTS_FILE"],
                map_location=device,
            )
        )
        model.to(device)

        print("Running Demo ....")

        for root, dirs, files in os.walk(args["DEMO_DIRECTORY"]):
            for file in files:

                sampleFile = os.path.join(root, file)

                preprocess_sample(sampleFile)

                inp, _ = prepare_input(sampleFile)
                inputBatch = torch.unsqueeze(inp, dim=0)

                inputBatch = (inputBatch.float()).to(device)

                model.eval()
                with torch.no_grad():
                    outputBatch = model(inputBatch)

                predictionBatch = decode(outputBatch)
                pred = predictionBatch[0][:]

                print("File: %s" % (file))
                print("Prediction: %s" % (pred))
                print("\n")

        print("Demo Completed.")

    else:
        print("Path to trained weights file not specified.")

    return


if __name__ == "__main__":
    main()
