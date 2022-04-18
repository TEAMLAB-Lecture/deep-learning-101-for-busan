"""
Author: Smeet Shah

Description: Python script to preprocess all samples in the dataset.

- Obtains a list of all samples in the dataset
- Preprocesses each sample in the dataset
"""

import os
import random

import torch
import numpy as np
from tqdm import tqdm

from config import args
from essentials.pprocs import preprocess_sample


def main():

    """
    Main function wrapper for preprocessing script.
    """

    random.seed(args["SEED"])
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])

    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        for file in files:
            filesList.append(os.path.join(root, file))

    print("Number of data samples to be processed = %d" % (len(filesList)))
    print("Starting preprocessing ....")

    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file)

    print("Preprocessing Done.")

    return


if __name__ == "__main__":
    main()
