"""
Author: Smeet Shah

Description: File for setting values for the configuration options.

- Project structure paths
- Hyperparameters
"""

import os

args = dict()

# project structure
args["CODE_DIRECTORY"] = os.getcwd()
args["DATA_DIRECTORY"] = os.path.join(args["CODE_DIRECTORY"], "data")
args["DATA_FILENAME"] = "data-03-diabetes.csv"

args["DEMO_DIRECTORY"] = "absolute path to directory containing demo samples"
args["PRETRAINED_WEIGHTS_FILE"] = None
# args["PRETRAINED_WEIGHTS_FILE"] = "/saved/weights/pretrained_weights.pt"

args["TRAINED_WEIGHTS_FILE"] = "/saved/weights/trained_weights.pt"

# data
args["VALIDATION_SPLIT"] = 0.1
args["NUM_WORKERS"] = 4

# training
args["SEED"] = 10
args["BATCH_SIZE"] = 20
args["NUM_EPOCHS"] = 500
args["SAVE_FREQUENCY"] = 5

# optimizer and scheduler
args["LEARNING_RATE"] = 0.001
args["MOMENTUM1"] = 0.9
args["MOMENTUM2"] = 0.999
args["LR_DECAY"] = 1

# loss
args["LAMBDA"] = 0.03


if __name__ == "__main__":

    for key, value in args.items():
        print(str(key) + " : " + str(value))
