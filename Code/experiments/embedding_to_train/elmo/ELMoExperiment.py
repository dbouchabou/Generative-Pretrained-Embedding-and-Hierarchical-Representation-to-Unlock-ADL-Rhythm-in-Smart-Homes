# coding: utf-8
# !/usr/bin/env python3

import os
import csv
import time
import json
import numpy as np


from SmartHomeHARLib.embedding import ELMoEventEmbedder


class ELMoExperiment:
    def __init__(self, dataset_name, train_x, experiment_parameters):
        super().__init__()

        self.experiment_parameters = experiment_parameters

        self.experiment_tag = (
            "Dataset_{}_Encoding_{}_WindowsSize_{}_EmbeddingSize_{}_NbEpochs_{}".format(
                dataset_name,
                self.experiment_parameters["encoding"],
                self.experiment_parameters["window_size"],
                self.experiment_parameters["embedding_size"],
                self.experiment_parameters["epoch_number"],
            )
        )

        # General
        self.global_classifier_accuracy = []
        self.global_classifier_balance_accuracy = []
        self.current_time = None
        self.dataset_name = dataset_name

        self.elmo_data_train = train_x

    def start(self):
        # Star time of the experiment
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(
            self.experiment_parameters["name"],
            self.experiment_parameters["model_type"],
            self.dataset_name,
            "run_" + "_" + str(current_time) + "_" + self.experiment_tag,
        )

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        filename_base = "model_6_basic_raw_{}_{}_{}_backward".format(
            self.dataset_name,
            self.experiment_parameters["window_size"],
            self.experiment_parameters["embedding_size"],
        )

        self.elmo_model = ELMoEventEmbedder(
            sentences=self.elmo_data_train,
            embedding_size=self.experiment_parameters["embedding_size"],
            window_size=self.experiment_parameters["window_size"],
            nb_epoch=self.experiment_parameters["epoch_number"],
            batch_size=self.experiment_parameters["batch_size"],
        )

        self.elmo_model.tokenize()

        if self.DEBUG:
            print("")
            print(self.elmo_model.vocabulary)

            input("Press Enter to continue...")

        self.elmo_model.prepare_4()

        if self.DEBUG:
            print("")
            print(self.elmo_model.forward_inputs.shape)
            print(self.elmo_model.backward_inputs.shape)
            print(self.elmo_model.forward_outputs.shape)
            print(self.elmo_model.forward_inputs[0])
            print(self.elmo_model.backward_inputs[0])
            print(self.elmo_model.forward_outputs[0])

            print(self.elmo_model.forward_inputs[1])
            print(self.elmo_model.backward_inputs[1])
            print(self.elmo_model.forward_outputs[1])

            print(np.sort(np.unique(self.elmo_model.forward_outputs)))

            input("Press Enter to continue...")

        picture_name = "ELMo_" + self.experiment_tag + ".png"
        picture_path = os.path.join(self.experiment_result_path, picture_name)

        self.elmo_model.compile(picture_path=picture_path)

        print("Start Training...")

        model_filename = filename_base + "_elmo_model.h5"
        final_model_path = os.path.join(self.experiment_result_path, model_filename)

        root_logdir = os.path.join(
            self.experiment_parameters["name"],
            "logs_{}_{}".format(self.experiment_parameters["name"], self.dataset_name),
        )

        run_id = "ELMo_" + self.experiment_tag + "_" + str(current_time)
        log_dir = os.path.join(root_logdir, run_id)

        csv_name = "ELMo_" + self.experiment_tag + ".csv"
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        self.elmo_model.train(final_model_path, log_dir=log_dir, csv_path=csv_path)

        print("Training Finish")

        # Save the vocabulary dict
        vocabulary = self.elmo_model.vocabulary

        vocabulary_filename = filename_base + "_elmo_dict_vocabulary.json"
        final_vocabulary_path = os.path.join(
            self.experiment_result_path, vocabulary_filename
        )

        with open(final_vocabulary_path, "w") as save_vocab_file:
            json.dump(vocabulary, save_vocab_file, indent=4)

    def __save_dict_to_json(self, where_to_save, dict_to_save):
        with open(where_to_save, "w") as json_dict_file:
            json.dump(dict_to_save, json_dict_file, indent=4)

    def save_config(self):
        experiment_parameters_name = "experiment_parameters.json"
        experiment_parameters_path = os.path.join(
            self.experiment_result_path, experiment_parameters_name
        )

        self.__save_dict_to_json(experiment_parameters_path, self.experiment_parameters)
