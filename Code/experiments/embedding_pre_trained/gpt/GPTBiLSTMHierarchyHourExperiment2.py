# coding: utf-8
# !/usr/bin/env python3

import os
import csv
import time
import json
import numpy as np
import h5py

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras_nlp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


from SmartHomeHARLib.custom_layers import TokenAndPositionEmbedding
from SmartHomeHARLib.custom_layers import GPT_Block
from SmartHomeHARLib.custom_layers import TransformerBlock
from SmartHomeHARLib.custom_layers.transfomers import padding_attention_mask_3


from SmartHomeHARLib.utils import Evaluator


def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True

    return False


def isDataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True

    return False


def getDatasetFromGroup(datasets, obj):
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets, x)
    else:
        datasets.append(obj)


def getWeightsForLayer(layerName, fileName):
    weights = []
    with h5py.File(fileName, mode="r") as f:
        model = f["/model_weights"]
        for key in model:
            if layerName in key:
                obj = model[key]
                datasets = []
                getDatasetFromGroup(datasets, obj)

                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights


def load_config(config_path):
    f = open(
        config_path,
    )

    # returns JSON object as
    # a dictionary
    return json.load(f)


class GPTBiLSTMHierarchyHourExperiment2:
    def __init__(
        self,
        dataset_name,
        train_x,
        test_x,
        experiment_parameters,
        cross_validation=True,
    ):
        super().__init__()

        self.experiment_parameters = experiment_parameters
        self.embedding_parameters = load_config(
            self.experiment_parameters["embedding_parameters"]
        )

        self.experiment_tag = "Dataset_{}_Encoding_{}_Segmentation_{}_Batch_{}_Patience_{}_SeqLenght_{}_EmbDim_{}_NbUnits_{}".format(
            dataset_name,
            self.experiment_parameters["encoding"],
            self.experiment_parameters["segmentation"],
            self.experiment_parameters["batch_size"],
            self.experiment_parameters["patience"],
            self.experiment_parameters["sequence_lenght"],
            self.embedding_parameters["embedding_size"],
            self.experiment_parameters["nb_units"],
        )

        # General
        self.global_classifier_accuracy = []
        self.global_classifier_balance_accuracy = []
        self.current_time = None
        self.wordDict = self.load_vocabulay_file(
            self.experiment_parameters["word_dict"]
        )
        self.actDict = {}
        self.train_x = train_x
        self.train_y = None
        self.train_x_encoded = None
        self.train_y_encoded = None
        self.dataset_name = dataset_name
        self.cross_validation = cross_validation

        # Classifier
        self.classifier_dataset_encoder = None
        self.classifier_segmentator = None

        self.classifier_model = None
        self.classifier_best_model_path = None
        self.classifier_data_X = []
        self.classifier_data_Y = []
        self.classifier_data_X_train = []
        self.classifier_data_Y_train = []
        self.classifier_data_X_test = []
        self.classifier_data_Y_test = []
        self.classifier_data_X_val = []
        self.classifier_data_Y_val = []

        self.test_x = test_x

        # Fix the random seed for tensorflow
        tf.random.set_seed(self.experiment_parameters["seed"])
        # Fix the random seed for numpy
        np.random.seed(self.experiment_parameters["seed"])
        # Enable mixed precision, which will speed up training by running most of our computations with 16 bit (instead of 32 bit) floating point numbers.

    def load_vocabulay_file(self, vocab_filename):
        with open(vocab_filename) as json_file:
            return json.load(json_file)

    def data_preprocessing(self):
        # extract labels
        self.train_y = self.train_x.labels.values

        # extract inputs
        x1 = self.train_x.input_1.values
        x2 = self.train_x.input_2.values

        x3 = self.train_x.input_6.values
        x4 = self.train_x.input_7.values

        x5 = self.train_x.input_11.values
        x6 = self.train_x.input_12.values

        x_all = x1 + x3 + x5

        # encode tokens

        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(x_all)
        tokenizer.word_index = self.wordDict

        # replace words into sentences by their index token
        x1_encoded = np.array(tokenizer.texts_to_sequences(x1), dtype=object)
        x1_encoded = pad_sequences(
            x1_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x3_encoded = np.array(tokenizer.texts_to_sequences(x3), dtype=object)
        x3_encoded = pad_sequences(
            x3_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x5_encoded = np.array(tokenizer.texts_to_sequences(x5), dtype=object)
        x5_encoded = pad_sequences(
            x5_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x2_encoded = pad_sequences(
            x2, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        x4_encoded = pad_sequences(
            x4, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        x6_encoded = pad_sequences(
            x6, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        # create the time distributed input
        reshaped_input_1 = []
        reshaped_input_2 = []

        for i in range(len(x1_encoded)):
            input_1 = []
            input_1.append(x1_encoded[i])
            input_1.append(x3_encoded[i])
            input_1.append(x5_encoded[i])

            input_2 = []
            input_2.append(x2_encoded[i])
            input_2.append(x4_encoded[i])
            input_2.append(x6_encoded[i])

            reshaped_input_1.append(input_1)
            reshaped_input_2.append(input_2)

        self.train_x_encoded = np.array(
            np.stack((reshaped_input_1, reshaped_input_2), axis=-1)
        )

        if self.DEBUG:
            print(self.train_x_encoded.shape)
            input("Press Enter to continue...")

        # encode labels

        le = preprocessing.LabelEncoder()
        le.fit(self.train_y)

        self.label_encoder = le

        self.train_y_encoded = le.transform(self.train_y)

        self.actDict = dict(zip(le.classes_, le.transform(le.classes_)))

        # cast in int
        for keys in self.actDict:
            self.actDict[keys] = int(self.actDict[keys])

    def data_preprocessing_test(self):
        # extract labels
        self.test_y = self.test_x.labels.values

        # extract inputs
        x1 = self.test_x.input_1.values
        x2 = self.test_x.input_2.values

        x3 = self.test_x.input_6.values
        x4 = self.test_x.input_7.values

        x5 = self.test_x.input_11.values
        x6 = self.test_x.input_12.values

        x_all = x1 + x3 + x5

        # encode tokens

        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(x_all)
        tokenizer.word_index = self.wordDict

        # replace words into sentences by their index token
        x1_encoded = np.array(tokenizer.texts_to_sequences(x1), dtype=object)
        x1_encoded = pad_sequences(
            x1_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x3_encoded = np.array(tokenizer.texts_to_sequences(x3), dtype=object)
        x3_encoded = pad_sequences(
            x3_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x5_encoded = np.array(tokenizer.texts_to_sequences(x5), dtype=object)
        x5_encoded = pad_sequences(
            x5_encoded,
            maxlen=self.experiment_parameters["sequence_lenght"],
            padding="post",
        )

        x2_encoded = pad_sequences(
            x2, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        x4_encoded = pad_sequences(
            x4, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        x6_encoded = pad_sequences(
            x6, maxlen=self.experiment_parameters["sequence_lenght"], padding="post"
        )

        # create the time distributed input
        reshaped_input_1 = []
        reshaped_input_2 = []

        for i in range(len(x1_encoded)):
            input_1 = []
            input_1.append(x1_encoded[i])
            input_1.append(x3_encoded[i])
            input_1.append(x5_encoded[i])

            input_2 = []
            input_2.append(x2_encoded[i])
            input_2.append(x4_encoded[i])
            input_2.append(x6_encoded[i])

            reshaped_input_1.append(input_1)
            reshaped_input_2.append(input_2)

        self.test_x_encoded = np.array(
            np.stack((reshaped_input_1, reshaped_input_2), axis=-1)
        )

        if self.DEBUG:
            print(self.test_x_encoded.shape)
            input("Press Enter to continue...")

        # encode labels

        self.test_y_encoded = self.label_encoder.transform(self.test_y)

    def model_selection(self):
        if self.cross_validation:
            with tqdm(total=2, desc="Dataset Split for cross validation") as pbar:
                kfold = StratifiedKFold(
                    n_splits=self.experiment_parameters["nb_splits"],
                    shuffle=True,
                    random_state=self.experiment_parameters["seed"],
                )

                for train, test in kfold.split(
                    self.train_x_encoded, self.train_y_encoded, groups=None
                ):
                    self.classifier_data_X_train.append(
                        np.array(self.train_x_encoded)[train]
                    )
                    self.classifier_data_Y_train.append(
                        np.array(self.train_y_encoded)[train]
                    )

                    print(np.array(self.train_x_encoded)[train].shape)
                    print(np.array(self.train_y_encoded)[train].shape)

                    self.classifier_data_X_test.append(
                        np.array(self.train_x_encoded)[test]
                    )
                    self.classifier_data_Y_test.append(
                        np.array(self.train_y_encoded)[test]
                    )

                pbar.update(1)

                self.classifier_data_X_train = np.array(self.classifier_data_X_train)
                self.classifier_data_Y_train = np.array(self.classifier_data_Y_train)

                self.classifier_data_X_test = np.array(self.classifier_data_X_test)
                self.classifier_data_Y_test = np.array(self.classifier_data_Y_test)
                pbar.update(1)

                if self.DEBUG:
                    print("")
                    print(self.classifier_data_X_train.shape)
                    print(self.classifier_data_Y_train.shape)
                    print(self.classifier_data_X_test.shape)
                    print(self.classifier_data_Y_test.shape)

                    input("Press Enter to continue...")
        else:
            with tqdm(total=1, desc="Dataset Split train and validation") as pbar:
                (
                    self.classifier_data_X_train,
                    self.classifier_data_X_val,
                    self.classifier_data_Y_train,
                    self.classifier_data_Y_val,
                ) = train_test_split(
                    self.train_x_encoded,
                    self.train_y_encoded,
                    test_size=0.2,
                    shuffle=True,
                    stratify=self.train_y_encoded,
                    random_state=self.experiment_parameters["seed"],
                )

                pbar.update(1)

                self.classifier_data_X_test = np.array(self.test_x_encoded)
                self.classifier_data_Y_test = np.array(self.test_y_encoded)

                if self.DEBUG:
                    print("Data processed")
                    print(self.classifier_data_X_train.shape)
                    print(self.classifier_data_Y_train.shape)
                    print(self.classifier_data_X_val.shape)
                    print(self.classifier_data_Y_val.shape)
                    print(self.classifier_data_X_test.shape)
                    print(self.classifier_data_Y_test.shape)

                    input("Press Enter to continue...")

    def build_model_classifier(self, run_number=0):
        nb_timesteps = 3
        nb_classes = len(list(self.actDict.keys()))
        embed_dim = self.embedding_parameters["embedding_size"]
        num_heads = self.embedding_parameters["num_heads"]
        dropout_rate = self.embedding_parameters["dropout"]
        num_of_layers = self.embedding_parameters["num_layers"]
        vocab_size = len(list(self.wordDict.keys())) + 1
        output_dim = self.experiment_parameters["nb_units"]
        hour_emb_dim = self.experiment_parameters["hour_emb_dim"]
        output_embedding_layer_nomalized = self.experiment_parameters[
            "output_embedding_layer_nomalized"
        ]

        if self.DEBUG:
            print("")
            print(vocab_size)

            input("Press Enter to continue...")

        # build the model

        # classifier
        input_sensor = Input(
            shape=(
                (
                    nb_timesteps,
                    None,
                )
            )
        )
        input_hour = Input(
            shape=(
                (
                    nb_timesteps,
                    None,
                )
            )
        )

        # GPT Embedding

        model_base = load_model(
            self.experiment_parameters["pre_train_embedding"],
            custom_objects={
                "TokenAndPositionEmbedding": keras_nlp.layers.TokenAndPositionEmbedding,
                # "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerDecoder": keras_nlp.layers.TransformerDecoder,
                "Perplexity": keras_nlp.metrics.Perplexity,
            },
        )
        print(model_base.summary())

        # new_model_base = tf.keras.Sequential()
        # new_model_base.add(model_base.layers[0])
        # new_model_base.add(model_base.layers[1])
        # new_model_base.add(model_base.layers[2])
        # new_model_base.add(model_base.layers[3])
        # new_model_base.add(model_base.layers[4])
        # new_model_base.add(model_base.layers[5])
        # new_model_base.add(model_base.layers[6])
        # new_model_base.add(model_base.layers[7])

        # new_model_base.trainable = self.experiment_parameters["trainable"]
        # output_gpt_embedding = new_model_base(
        #    inputs, training=self.experiment_parameters["trainable"])

        # print(new_model_base.summary())
        output_gpt_embedding = TimeDistributed(
            model_base.layers[1], trainable=self.experiment_parameters["trainable"]
        )(input_sensor)
        # gpt_embedding = TimeDistributed(
        #     model_base.layers[2], trainable=self.experiment_parameters["trainable"]
        # )(gpt_embedding)
        # gpt_embedding = TimeDistributed(
        #     model_base.layers[3], trainable=self.experiment_parameters["trainable"]
        # )(gpt_embedding)
        # output_gpt_embedding = TimeDistributed(
        #     model_base.layers[4], trainable=self.experiment_parameters["trainable"]
        # )(gpt_embedding)

        for i in range(2, num_of_layers + 2):
            # new_model_base.add(model_base.layers[i + 1])
            decoder_layer = TimeDistributed(
                model_base.layers[i], trainable=self.experiment_parameters["trainable"]
            )
            output_gpt_embedding = decoder_layer(output_gpt_embedding)

        if output_embedding_layer_nomalized:
            sensor_embedding = TimeDistributed(LayerNormalization(epsilon=1e-5))(
                output_gpt_embedding
            )

            sensor_embedding = TimeDistributed(Bidirectional(LSTM(output_dim)))(
                sensor_embedding
            )
        else:
            sensor_embedding = TimeDistributed(Bidirectional(LSTM(output_dim)))(
                output_gpt_embedding
            )

        # Time Embedding

        embedding_hour = Embedding(
            input_dim=24 + 1,
            output_dim=hour_emb_dim,
            input_length=nb_timesteps,
            mask_zero=True,
        )

        hour_embedding = TimeDistributed(embedding_hour)(input_hour)

        if output_embedding_layer_nomalized:
            hour_embedding = TimeDistributed(LayerNormalization(epsilon=1e-5))(
                hour_embedding
            )
            hour_embedding = TimeDistributed(Bidirectional(LSTM(output_dim)))(
                hour_embedding
            )
        else:
            hour_embedding = TimeDistributed(Bidirectional(LSTM(output_dim)))(
                hour_embedding
            )

        ####

        seq_embedding = Concatenate()([sensor_embedding, hour_embedding])
        # seq_embedding = Concatenate()([output_gpt_embedding, hour_embedding])

        if output_embedding_layer_nomalized:
            # seq_embedding = TimeDistributed(LayerNormalization(epsilon=1e-5))(
            #     seq_embedding
            # )

            # seq_embedding = TimeDistributed(Bidirectional(LSTM(448)))(seq_embedding)
            seq_embedding = TimeDistributed(
                keras_nlp.layers.TransformerEncoder(
                    intermediate_dim=448,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    activation="relu",
                    normalize_first=True,
                )
            )(seq_embedding)

            # seq_embedding = TimeDistributed(GlobalAveragePooling1D())(seq_embedding)

            # seq_embedding = TimeDistributed(LayerNormalization(epsilon=1e-5))(
            #     seq_embedding
            # )

            seq_embedding = TimeDistributed(Bidirectional(LSTM(224)))(seq_embedding)

            # seq_embedding = TimeDistributed(BatchNormalization())(seq_embedding)

        lstm_1 = Bidirectional(LSTM(128))(seq_embedding)

        # lstm_1 = BatchNormalization()(lstm_1)

        output_layer = Dense(nb_classes, activation="softmax")(lstm_1)

        if output_embedding_layer_nomalized:
            self.classifier_model = Model(
                inputs=[input_sensor, input_hour],
                outputs=output_layer,
                # name="GPT_Hierachy_Hour_BiLSTM_NORM_V1_Classifier",
                name="GPT_Hierachy_Hour_BiLSTM_NORM_V4_224_Encoder_Classifier",
            )
        else:
            self.classifier_model = Model(
                inputs=[input_sensor, input_hour],
                outputs=output_layer,
                name="GPT_Hierachy_Hour_BiLSTM_Classifier",
            )

        # ceate a picture of the model
        picture_name = (
            self.classifier_model.name
            + "_"
            + self.experiment_tag
            + "_"
            + str(run_number)
            + ".png"
        )
        picture_path = os.path.join(self.experiment_result_path, picture_name)

        plot_model(self.classifier_model, show_shapes=True, to_file=picture_path)

    def train(
        self, X_train_input, Y_train_input, X_val_input, Y_val_input, run_number=0
    ):
        root_logdir = os.path.join(
            self.experiment_parameters["name"],
            "logs_{}_{}".format(self.experiment_parameters["name"], self.dataset_name),
        )

        run_id = (
            self.classifier_model.name
            + "_"
            + self.experiment_tag
            + "_"
            + str(self.current_time)
            + str(run_number)
        )
        log_dir = os.path.join(root_logdir, run_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        best_model_name_saved = (
            self.classifier_model.name
            + "_"
            + self.experiment_tag
            + "_BEST_"
            + str(run_number)
            + ".h5"
        )
        self.classifier_best_model_path = os.path.join(
            self.experiment_result_path, best_model_name_saved
        )

        csv_name = (
            self.classifier_model.name
            + "_"
            + self.experiment_tag
            + "_"
            + str(run_number)
            + ".csv"
        )
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        # create a callback for the tensorboard
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

        # callbacks
        csv_logger = CSVLogger(csv_path)

        # simple early stopping
        es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=self.experiment_parameters["patience"],
        )
        mc = ModelCheckpoint(
            self.classifier_best_model_path,
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True,
        )
        # mc = ModelCheckpoint(self.classifier_best_model_path,
        #                     mode='auto', verbose=1, save_best_only=True)
        # mc = ModelCheckpoint(self.classifier_best_model_path, monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)

        # cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
        cbs = [csv_logger, tensorboard_cb, mc, es]

        if self.cross_validation:
            self.classifier_model.fit(
                X_train_input,
                Y_train_input,
                epochs=self.experiment_parameters["nb_epochs"],
                batch_size=self.experiment_parameters["batch_size"],
                verbose=self.experiment_parameters["verbose"],
                callbacks=cbs,
                validation_split=0.2,
                shuffle=True,
            )
        else:
            self.classifier_model.fit(
                X_train_input,
                Y_train_input,
                epochs=self.experiment_parameters["nb_epochs"],
                batch_size=self.experiment_parameters["batch_size"],
                verbose=self.experiment_parameters["verbose"],
                callbacks=cbs,
                validation_data=(X_val_input, Y_val_input),
                shuffle=True,
            )

    def check_input_model(self, run_number=0):
        X_val_input = []
        Y_val_input = []

        if self.DEBUG:
            print("check input")
            print(self.classifier_data_X_train.ndim)
            print(self.classifier_data_X_train.shape)
            print(self.classifier_data_X_test.shape)
            if self.classifier_data_X_val != []:
                print(self.classifier_data_X_val.shape)
            else:
                print("None")
            input("Press Enter to continue...")

        # Check number size of exemples
        if self.classifier_data_X_train.ndim < 5:
            data_X_train = self.classifier_data_X_train
            data_Y_train = self.classifier_data_Y_train
        else:
            data_X_train = self.classifier_data_X_train[run_number]
            data_Y_train = self.classifier_data_Y_train[run_number]

        if self.classifier_data_X_test.ndim < 5:
            data_X_test = self.classifier_data_X_test
            data_Y_test = self.classifier_data_Y_test
        else:
            data_X_test = self.classifier_data_X_test[run_number]
            data_Y_test = self.classifier_data_Y_test[run_number]

        if self.classifier_data_X_val != []:
            if self.classifier_data_X_val.ndim < 5:
                data_X_val = self.classifier_data_X_val
                data_Y_val = self.classifier_data_Y_val
            else:
                data_X_val = self.classifier_data_X_val[run_number]
                data_Y_val = self.classifier_data_Y_val[run_number]

        # Nb features depends on data shape
        if data_X_train.ndim > 3:
            nb_features = data_X_train.shape[3]
        else:
            nb_features = 1

        if self.DEBUG:
            print(len(data_X_train))
            print(len(data_X_val))
            print(len(data_X_test))
            print(data_X_train.shape)
            print(data_X_val.shape)
            print(data_X_test.shape)
            input("Press Enter to continue...")

        X_train_input = [data_X_train[:, :, :, 0], data_X_train[:, :, :, 1]]
        X_test_input = [data_X_test[:, :, :, 0], data_X_test[:, :, :, 1]]

        if self.classifier_data_X_val != []:
            X_val_input = [data_X_val[:, :, :, 0], data_X_val[:, :, :, 1]]

        # X_train_input = data_X_train.transpose(3, 0, 1, 2)
        # X_test_input = data_X_test.transpose(3, 0, 1, 2)

        # if self.classifier_data_X_val != []:
        #     X_val_input = data_X_val.transpose(3, 0, 1, 2)

        Y_train_input = data_Y_train
        Y_test_input = data_Y_test

        if self.classifier_data_X_val != []:
            Y_val_input = data_Y_val

        if self.DEBUG:
            print("Train {}:".format(np.array(X_train_input).shape))
            if self.classifier_data_X_val != []:
                print("Val : {}".format(np.array(X_val_input).shape))
            else:
                print("Val : None")
            print("Test : {}".format(np.array(X_test_input).shape))

            input("Press Enter to continue...")

        return (
            X_train_input,
            Y_train_input,
            X_val_input,
            Y_val_input,
            X_test_input,
            Y_test_input,
            nb_features,
        )

    def compile_model(self):
        self.classifier_model.compile(
            loss="sparse_categorical_crossentropy",
            # optimizer = tf.keras.optimizers.Adam(learning_rate=cyclical_learning_rate),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["sparse_categorical_accuracy"],
        )

        # print summary
        print(self.classifier_model.summary())

    def evaluate(self, X_test_input, Y_test_input, run_number=0):
        if self.DEBUG:
            print("")
            print("EVALUATION")
            print(np.array(X_test_input).shape)
            print(np.array(Y_test_input).shape)
            print(self.classifier_best_model_path)
            input("Press Enter to continue...")

        evaluator = Evaluator(X_test_input, Y_test_input, model=self.classifier_model)

        evaluator.simpleEvaluation(
            self.experiment_parameters["batch_size"], Y_test_input=Y_test_input
        )
        self.global_classifier_accuracy.append(evaluator.ascore)

        evaluator.evaluate()

        listActivities = list(self.actDict.keys())
        indexLabels = list(self.actDict.values())
        evaluator.classificationReport(listActivities, indexLabels)
        # print(evaluator.report)

        report_name = (
            self.classifier_model.name
            + "_repport_"
            + self.experiment_tag
            + "_"
            + str(run_number)
            + ".csv"
        )
        report_path = os.path.join(self.experiment_result_path, report_name)
        evaluator.saveClassificationReport(report_path)

        evaluator.confusionMatrix()
        # print(evaluator.cm)

        confusion_name = (
            self.classifier_model.name
            + "_confusion_matrix_"
            + self.experiment_tag
            + "_"
            + str(run_number)
            + ".csv"
        )
        confusion_path = os.path.join(self.experiment_result_path, confusion_name)
        evaluator.saveConfusionMatrix(confusion_path)

        evaluator.balanceAccuracyCompute()
        self.global_classifier_balance_accuracy.append(evaluator.bscore)

    def start(self):
        # Star time of the experiment
        self.current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(
            self.experiment_parameters["name"],
            self.experiment_parameters["model_type"],
            "run_" + self.experiment_tag + "_" + str(self.current_time),
        )

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        self.data_preprocessing()

        if not self.cross_validation:
            self.data_preprocessing_test()

        # Split the dataset into train, val and test examples
        self.model_selection()

        if self.cross_validation:
            nb_runs = self.experiment_parameters["nb_splits"]
        else:
            nb_runs = 1

        if self.DEBUG:
            print("")
            print("NB RUN: {}".format(nb_runs))

        for run_number in range(nb_runs):
            # prepare input according to the model type
            (
                X_train_input,
                Y_train_input,
                X_val_input,
                Y_val_input,
                X_test_input,
                Y_test_input,
                nb_features,
            ) = self.check_input_model(run_number)

            self.build_model_classifier(run_number)

            # compile the model
            self.compile_model()

            self.train(
                X_train_input, Y_train_input, X_val_input, Y_val_input, run_number
            )

            self.evaluate(X_test_input, Y_test_input, run_number)

    def __save_dict_to_json(self, where_to_save, dict_to_save):
        with open(where_to_save, "w") as json_dict_file:
            json.dump(dict_to_save, json_dict_file, indent=4)

    def save_word_dict(self):
        word_dict_name = "wordDict.json"
        word_dict_path = os.path.join(self.experiment_result_path, word_dict_name)

        self.__save_dict_to_json(word_dict_path, self.wordDict)

    def save_activity_dict(self):
        activity_dict_name = "activityDict.json"
        activity_dict_path = os.path.join(
            self.experiment_result_path, activity_dict_name
        )

        self.__save_dict_to_json(activity_dict_path, self.actDict)

    def save_config(self):
        experiment_parameters_name = "experiment_parameters.json"
        experiment_parameters_path = os.path.join(
            self.experiment_result_path, experiment_parameters_name
        )

        self.__save_dict_to_json(experiment_parameters_path, self.experiment_parameters)

    def save_metrics(self):
        csv_name = (
            "cv_scores"
            + self.classifier_model.name
            + "_"
            + self.experiment_tag
            + "_"
            + str(self.current_time)
            + ".csv"
        )
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        with open(csv_path, "w") as output:
            writer = csv.writer(output, lineterminator="\n")

            writer.writerow(["accuracy score :"])
            for val in self.global_classifier_accuracy:
                writer.writerow([val * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_accuracy)])

            writer.writerow([])
            writer.writerow(["balanced accuracy score :"])

            for val2 in self.global_classifier_balance_accuracy:
                writer.writerow([val2 * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_balance_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_balance_accuracy)])
