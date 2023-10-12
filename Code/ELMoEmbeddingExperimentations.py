#!/usr/bin/env python3

import argparse
import os
import random as rn


import tensorflow as tf
import numpy as np


from experiments.embedding_to_train.elmo.ELMoExperiment import ELMoExperiment
from utils import *

DEBUG_MODE = False
SEED = 7

# Set the seed for hash based operations in python
os.environ["PYTHONHASHSEED"] = "0"
# Fix the random seed for tensorflow
tf.random.set_seed(SEED)
# Fix the random seed for numpy
np.random.seed(SEED)
# Fix the random seed for random module
rn.seed(SEED)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    strategy = tf.distribute.MirroredStrategy()

    # Set and parse the arguments list
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=""
    )
    p.add_argument("--d", dest="data", action="store", default="", help="dataset name")
    p.add_argument(
        "--e",
        dest="experiment",
        action="store",
        default="",
        help="dataset name",
        required=True,
    )

    p.add_argument(
        "--c",
        dest="config",
        action="store",
        default="",
        help="config_file",
        required=True,
    )

    args = p.parse_args()

    data = str(args.data)
    experiement = str(args.experiment)
    config_path = str(args.config)

    # Load the config file
    config = load_config(config_path)

    # Load data
    train_x = load_train_data_elmo(data)

    

    with strategy.scope():
        if experiement == "elmo":
            exp = ELMoExperiment(data, train_x, config)
        elif experiement == "elmo_sep":
            exp = ELMoExperiment(data, train_x, config)
        else:
            print("WRONG EXPERIMENT CHOICE")
            exit()

        exp.DEBUG = DEBUG_MODE

        exp.start()

    # Save experiment config
    exp.save_config()
