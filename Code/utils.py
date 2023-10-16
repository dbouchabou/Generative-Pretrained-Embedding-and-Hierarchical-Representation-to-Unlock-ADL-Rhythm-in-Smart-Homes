#!/usr/bin/env python3
import pickle
import json


def load_config(config_path):
    f = open(
        config_path,
    )

    # returns JSON object as
    # a dictionary
    return json.load(f)


def load_train_data_gpt(data):
    train_x = None
    train_y = None

    # Load the list of DataFrames from the pickle file
    with open("datasets/" + data + "_train_data_time.pickle", "rb") as f:
        train_x = pickle.load(f)

    return train_x, train_y


def load_train_data_gpt_sep(data):
    train_x = None
    train_y = None

    # Load the list of DataFrames from the pickle file
    with open("datasets/" + data + "_train_gpt_with_sep_X.pickle", "rb") as f:
        train_x = pickle.load(f)

    with open("datasets/" + data + "_train_gpt_with_sep_Y.pickle", "rb") as f:
        train_y = pickle.load(f)

    return train_x, train_y


def load_train_data_elmo(dataset_name):
    # Load train data X from the pickle file
    with open(
        "datasets/" + dataset_name + "_train_classification_data_X.pickle", "rb"
    ) as f:
        train_x = pickle.load(f)

    return train_x


def load_train_data_from_dataframe_time(dataset_name):
    # Load train data X from the pickle file
    with open(
        "datasets/" + dataset_name + "_train_classification_data_time_dataframe.pickle",
        "rb",
    ) as f:
        train_x = pickle.load(f)

    return train_x


def load_test_data_from_dataframe_time(dataset_name):
    # Load train data X from the pickle file
    with open(
        "datasets/" + dataset_name + "_test_classification_data_time_dataframe.pickle",
        "rb",
    ) as f:
        test_x = pickle.load(f)

    return test_x
