#!/usr/bin/env python3
from utils import *


from experiments.embedding_pre_trained.gpt.GPTBiLSTMContextExperiment2 import (
    GPTBiLSTMContextExperiment2,
)
from experiments.embedding_pre_trained.gpt.GPTBiLSTMSEPExperiment2 import (
    GPTBiLSTMSEPExperiment2,
)
from experiments.embedding_pre_trained.gpt.GPTBiLSTMHierarchyExperiment2 import (
    GPTBiLSTMHierarchyExperiment2,
)

from experiments.embedding_pre_trained.gpt.GPTBiLSTMHierarchyHourExperiment2 import (
    GPTBiLSTMHierarchyHourExperiment2,
)
from experiments.embedding_pre_trained.gpt.GPTBiLSTMExperiment2 import (
    GPTBiLSTMExperiment2,
)
from experiments.embedding_pre_trained.elmo.ELMoBiLSTMHierarchyExperiment import (
    ELMoBiLSTMHierarchyExperiment,
)
from experiments.embedding_pre_trained.elmo.ELMoBiLSTMHierarchyHourExperiment import (
    ELMoBiLSTMHierarchyHourExperiment,
)
from experiments.embedding_pre_trained.elmo.ELMoBiLSTMExperiment import (
    ELMoBiLSTMExperiment,
)
from experiments.embedding_pre_trained.elmo.ELMoBiLSTMContextExperiment import (
    ELMoBiLSTMContextExperiment,
)

import os
import json
import argparse
import numpy as np
import random as rn

import tensorflow as tf


SEED = 7
DEBUG_MODE = False

# Set the seed for hash based operations in python
os.environ["PYTHONHASHSEED"] = "0"
# Fix the random seed for tensorflow
tf.random.set_seed(SEED)
# Fix the random seed for numpy
np.random.seed(SEED)
# Fix the random seed for random module
rn.seed(SEED)


def load_config(config_path):
    f = open(
        config_path,
    )

    # returns JSON object as
    # a dictionary
    return json.load(f)


milan_dict = {
    "Other": "Other",
    "Master_Bedroom_Activity": "Other",
    "Meditate": "Other",
    "Chores": "Work",
    "Desk_Activity": "Work",
    "Morning_Meds": "Take_medicine",
    "Eve_Meds": "Take_medicine",
    "Sleep": "Sleep",
    "Read": "Relax",
    "Watch_TV": "Relax",
    "Leave_Home": "Leave_Home",
    "Dining_Rm_Activity": "Eat",
    "Kitchen_Activity": "Cook",
    "Bed_to_Toilet": "Bed_to_toilet",
    "Master_Bathroom": "Bathing",
    "Guest_Bathroom": "Bathing",
}

cairo_dict = {
    "Other": "Other",
    "R1_wake": "Other",
    "R2_wake": "Other",
    "Night_wandering": "Other",
    "R1_work_in_office": "Work",
    "Laundry": "Work",
    "R2_take_medicine": "Take_medicine",
    "R1_sleep": "Sleep",
    "R2_sleep": "Sleep",
    "Leave_home": "Leave_Home",
    "Breakfast": "Eat",
    "Dinner": "Eat",
    "Lunch": "Eat",
    "Bed_to_toilet": "Bed_to_toilet",
}


aruba_dict = {
    "Other": "Other",
    "Wash_Dishes": "Work",
    "Sleeping": "Sleep",
    "Respirate": "Other",
    "Relax": "Relax",
    "Meal_Preparation": "Cook",
    "Housekeeping": "Work",
    "Enter_Home": "Enter_Home",
    "Leave_Home": "Leave_Home",
    "Eating": "Eat",
    "Bed_to_Toilet": "Bed_to_toilet",
    "Work": "Work",
}

if __name__ == "__main__":
    # Specify the Tensorflow environment
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
    p.add_argument(
        "--d",
        dest="data",
        action="store",
        default="",
        help="dataset name",
        required=True,
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
    p.add_argument(
        "--nb",
        dest="nb_run",
        action="store",
        default="1",
        help="number of repitition",
        required=False,
    )

    p.add_argument(
        "--cv",
        dest="cross_val",
        action="store",
        default="True",
        help="cross validation training",
        required=False,
    )

    args = p.parse_args()

    data = str(args.data)
    experiment = str(args.experiment)
    config_path = str(args.config)
    nb_run = int(args.nb_run)
    cross_val = str(args.cross_val)

    if cross_val == "True":
        cross_val = True
    else:
        cross_val = False

    # Load the config file
    config = load_config(config_path)

    # Load data
    train_x = load_train_data_from_dataframe_time(data)

    if not cross_val:
        test_x = load_test_data_from_dataframe_time(data)
    else:
        test_x = None

    tab_acc = []
    tab_bal_acc = []

    with strategy.scope():
        print("NB RUNS = {}".format(nb_run))
        for i in range(nb_run):
            print("\nRUN = {}/{}\n".format(i + 1, nb_run))

            if experiment == "elmo_bi_lstm":
                exp = ELMoBiLSTMExperiment(
                    data, train_x, test_x, config, cross_val)

            elif experiment == "gpt_bi_lstm":
                exp = GPTBiLSTMExperiment2(
                    data, train_x, test_x, config, cross_val)

            elif experiment == "elmo_bi_lstm_hierarchy":
                exp = ELMoBiLSTMHierarchyExperiment(
                    data, train_x, test_x, config, cross_val
                )
            elif experiment == "elmo_bi_lstm_hierarchy_hour":
                exp = ELMoBiLSTMHierarchyHourExperiment(
                    data, train_x, test_x, config, cross_val
                )
            elif experiment == "elmo_bi_lstm_context":
                exp = ELMoBiLSTMContextExperiment(
                    data, train_x, test_x, config, cross_val
                )

            elif experiment == "gpt_bi_lstm_hierarchy":
                exp = GPTBiLSTMHierarchyExperiment2(
                    data, train_x, test_x, config, cross_val
                )
            elif experiment == "gpt_bi_lstm_hierarchy_hour":
                exp = GPTBiLSTMHierarchyHourExperiment2(
                    data, train_x, test_x, config, cross_val
                )
            elif experiment == "gpt_bi_lstm_sep":
                exp = GPTBiLSTMSEPExperiment2(
                    data, train_x, test_x, config, cross_val)
            elif experiment == "gpt_bi_lstm_context":
                exp = GPTBiLSTMContextExperiment2(
                    data, train_x, test_x, config, cross_val
                )

            exp.DEBUG = DEBUG_MODE

            exp.start()

            # Save word dict
            exp.save_word_dict()

            # Save activity dict
            exp.save_activity_dict()

            # Save metrics
            exp.save_metrics()

            # Save experiment config
            exp.save_config()

            print(
                "Accuracy run {}: {:.2f}% (+/- {:.2f}%)".format(
                    i + 1,
                    np.mean(exp.global_classifier_accuracy) * 100,
                    np.std(exp.global_classifier_accuracy),
                )
            )

            print(
                "Balanced Accuracy run {}: {:.2f}% (+/- {:.2f}%)".format(
                    i + 1,
                    np.mean(exp.global_classifier_balance_accuracy) * 100,
                    np.std(exp.global_classifier_balance_accuracy),
                )
            )

            tab_acc.append(np.mean(exp.global_classifier_accuracy))
            tab_bal_acc.append(np.mean(exp.global_classifier_balance_accuracy))

    print(
        "Average Accuracy over all runs: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(tab_acc) * 100, np.std(tab_acc)
        )
    )

    print(
        "Average Balanced Accuracy over all runs: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(tab_bal_acc) * 100, np.std(tab_bal_acc)
        )
    )
