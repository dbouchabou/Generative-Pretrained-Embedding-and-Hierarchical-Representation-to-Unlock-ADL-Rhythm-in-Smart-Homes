#!/usr/bin/env python3

from utils import *

import argparse

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import random as rn

from experiments.embedding_to_train.gpt.GPTExperiment2 import GPTExperiment2

DEBUG_MODE = True
SEED = 7

# Set the seed for hash based operations in python
os.environ["PYTHONHASHSEED"] = "0"
# Fix the random seed for tensorflow
tf.random.set_seed(SEED)
# Fix the random seed for numpy
np.random.seed(SEED)
# Fix the random seed for random module
rn.seed(SEED)







def load_vocabulary(path):
    with open(path) as json_file:
        vocabulary = json.load(json_file)

    return vocabulary


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
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
    p.add_argument(
        "--d", 
        dest="data", 
        action="store", 
        default="", 
        help="dataset name"
    )

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

    # vocab
    vocab = None

    print(data)

    with strategy.scope():

        if  experiement == "gpt2":
            # Load data
            train_x, train_y = load_train_data_gpt(data)
            exp = GPTExperiment2(data, train_x, train_y, vocab, config)

        elif experiement == "gpt2_sep":
            # Load data
            train_x, train_y = load_train_data_gpt_sep(data)
            exp = GPTExperiment2(data, train_x, train_y, vocab, config)
    

        # Save experiment config
        # exp.save_config()

        exp.DEBUG = False

        exp.start()
