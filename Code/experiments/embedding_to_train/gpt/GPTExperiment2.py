# coding: utf-8
# !/usr/bin/env python3

import os
import time
import json
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from SmartHomeHARLib.custom_layers import TokenAndPositionEmbedding
from SmartHomeHARLib.custom_layers import GPT_Block
from SmartHomeHARLib.custom_layers import causal_attention_mask

import keras_nlp
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from tensorflow.keras import backend as K


def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


class GPTExperiment2:
    def __init__(self, dataset_name, train_x, train_y, vocab, experiment_parameters):
        super().__init__()

        self.experiment_parameters = experiment_parameters
        self.dataset_name = dataset_name
        self.dataset = None

        self.experiment_tag = "Dataset_{}_Encoding_{}_WindowsSize_{}_EmbeddingSize_{}_BatchSize_{}_NbEpochs_{}_Head{}_Layers_{}_Stride_{}".format(
            self.dataset_name,
            self.experiment_parameters["encoding"],
            self.experiment_parameters["bloc_size"],
            self.experiment_parameters["embedding_size"],
            self.experiment_parameters["batch_size"],
            self.experiment_parameters["epoch_number"],
            self.experiment_parameters["num_heads"],
            self.experiment_parameters["num_layers"],
            self.experiment_parameters["stride"],
        )

        # Embedding
        self.data_train_X = []
        self.data_train_y = []
        self.model = None
        # self.wordDict = vocab
        self.wordDict = {}
        self.train_x = train_x
        self.train_y = train_y

        self.train_x_encoded = train_x
        self.train_y_encoded = train_y

        self.current_time = None

    def encode(self, df):
        df["merge"] = df["sensor"].astype(str) + df["value"].astype(str)

        feature_1 = df["merge"].values.astype(str)
        # activity = df['activity'].values[0]
        # activity = df['activity'].value_counts().idxmax()

        return feature_1

    def split_array(self, arr, window_size=1024, stride=1):
        # window_size = 1024
        # stride = int(window_size / 2)
        n_windows = int(np.ceil((len(arr) - window_size) / stride)) + 1
        input_seq = []
        output_seq = []
        if n_windows > 0:
            if n_windows % 2 == 0:
                for i in range(n_windows):
                    start = i * stride
                    end = start + window_size
                    if end > len(arr):
                        start -= end - len(arr)
                        end = len(arr)

                    if i % 2 == 0:
                        input_seq.append(arr[start:end])
                    else:
                        output_seq.append(arr[start:end])
            else:
                for i in range(n_windows - 1):
                    start = i * stride
                    end = start + window_size
                    if end > len(arr):
                        start -= end - len(arr)
                        end = len(arr)

                    if i % 2 == 0:
                        input_seq.append(arr[start:end])
                    else:
                        output_seq.append(arr[start:end])

                last_seg = arr[-(window_size + 1) :]
                input_seq.append(last_seg[:window_size])
                output_seq.append(last_seg[-window_size:])
        else:
            print("windows size superiror to input lenght")

        return input_seq, output_seq

    def data_preprocessing(self):
        if self.DEBUG:
            print("")
            print("Vocabulary generation")

            input("Press Enter to continue...")

        # encode tokens
        print("Create vocab")
        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(self.train_x)
        self.wordDict = tokenizer.word_index

        if self.DEBUG:
            print("")
            print("Turn sequence into integers")

            input("Press Enter to continue...")
        # replace words into sentences by their index token
        print("Replace words into sentences by their index token")
        self.train_x_encoded = np.array(
            tokenizer.texts_to_sequences(self.train_x), dtype=object
        )
        self.train_y_encoded = np.array(
            tokenizer.texts_to_sequences(self.train_y), dtype=object
        )

        print("Pad sequences")
        self.train_x_encoded = pad_sequences(
            self.train_x_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
        )
        self.train_y_encoded = pad_sequences(
            self.train_y_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
        )

    def data_preprocessing_2(self):
        self.train_x_encoded = pad_sequences(
            self.train_x_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
        )
        self.train_y_encoded = pad_sequences(
            self.train_y_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
        )

    def split(self, list_a, chunk_size):
        seqs = []
        for i in range(0, len(list_a), chunk_size):
            seq = list_a[i : i + chunk_size]
            x = seq[:-1]
            y = seq[1:]
            return x, y

    def sliding_window(self, elements, window_size, stride):
        chunks_x = []
        chunks_y = []

        if len(elements) < window_size:
            x = elements[:-1]
            y = elements[1:]

            # print("")
            # print("INFERIEUR !!!!!!")
            # print("")
            # print(len(elements))
            # print("")
            # print(elements)
            # print("")
            # print(x)
            # print("")
            # print(y)
            # input("Press Enter to continue...")

            chunks_x.append(x)
            chunks_y.append(y)

            return chunks_x, chunks_y

        n_windows = int(np.ceil((len(elements) - window_size) / stride)) + 1

        for i in tqdm(range(n_windows)):
            start = i * stride
            end = start + window_size
            if end > len(elements):
                start -= end - len(elements)
                end = len(elements)

            x = elements[start:end][:-1]
            y = elements[start:end][1:]

            chunks_x.append(x)
            chunks_y.append(y)

        return chunks_x, chunks_y

    def data_preprocessing_4(self):
        print("Turn sensor into Words")
        training_weeks = []

        for df in tqdm(self.train_x):
            x = self.encode(df)
            training_weeks.append(x)

        print("Turn sensor words into sentences")
        training_weeks_text = []
        for s in tqdm(training_weeks):
            training_weeks_text.append(" ".join(s))

        # Tokenize
        print("Tokenize and create vocabulary")
        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(training_weeks_text)
        self.wordDict = tokenizer.word_index

        print("Encode sentences")
        training_weeks_encoded = tokenizer.texts_to_sequences(training_weeks_text)

        print("Create Input and Output to train the model")
        input_sequences_encoded = []
        output_sequences_encoded = []

        for week in tqdm(training_weeks_encoded):
            # x,y = self.split(week, 1025)
            # print("Shpae week : {}" .format(np.array(week).shape))

            x, y = self.sliding_window(week, 1025, 512)

            input_sequences_encoded = input_sequences_encoded + x
            output_sequences_encoded = output_sequences_encoded + y

            # print("Shpae week : {}" .format(np.array(input_sequences_encoded).shape))

        self.dataset_size = len(input_sequences_encoded)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences_encoded, output_sequences_encoded)
        )
        print("shuffle")
        self.dataset = self.dataset.shuffle(10).batch(
            self.experiment_parameters["batch_size"]
        )
        # print("Pad sequences in case")
        # self.train_x_encoded = pad_sequences(input_sequences_encoded, maxlen = self.experiment_parameters["bloc_size"], padding = 'post', value=0.0)
        # self.train_y_encoded = pad_sequences(output_sequences_encoded, maxlen = self.experiment_parameters["bloc_size"], padding = 'post', value=0.0)

    def data_preprocessing_3(self):
        print("Turn sensor into Words")
        training_weeks = []

        for df in tqdm(self.train_x):
            x = self.encode(df)
            training_weeks.append(x)

        print("Turn sensor words into sentences")
        training_weeks_text = []
        for s in tqdm(training_weeks):
            training_weeks_text.append(" ".join(s))

        # Tokenize
        print("Tokenize and create vocabulary")
        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(training_weeks_text)
        self.wordDict = tokenizer.word_index

        print("Encode sentences")
        training_weeks_encoded = tokenizer.texts_to_sequences(training_weeks_text)

        print("Create Input and Output to train the model")
        input_sequences_encoded = []
        output_sequences_encoded = []

        for week in tqdm(training_weeks_encoded):
            inp_s, out_s = self.split_array(week, 1024, 512)

            input_sequences_encoded = input_sequences_encoded + inp_s
            output_sequences_encoded = output_sequences_encoded + out_s

        print("Pad sequences in case")
        self.train_x_encoded = pad_sequences(
            input_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )
        self.train_y_encoded = pad_sequences(
            output_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )

    def data_preprocessing_5(self):
        print("Turn sensor into Words")
        training_weeks = []

        for df in tqdm(self.train_x):
            x = self.encode(df)
            training_weeks.append(x)

        print("Turn sensor words into sentences")
        training_weeks_text = []
        for s in tqdm(training_weeks):
            training_weeks_text.append(" ".join(s))

        # Tokenize
        print("Tokenize and create vocabulary")
        # print(len(training_weeks_text))
        tokenizer = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tokenizer.fit_on_texts(training_weeks_text)
        self.wordDict = tokenizer.word_index

        print("Encode sentences")
        training_weeks_encoded = tokenizer.texts_to_sequences(training_weeks_text)

        print("Create Input and Output to train the model")
        input_sequences_encoded = []
        output_sequences_encoded = []

        for week in tqdm(training_weeks_encoded):
            x, y = self.sliding_window(
                week,
                self.experiment_parameters["bloc_size"] + 1,
                self.experiment_parameters["stride"],
            )

            print(np.array(x).shape)

            # if len(x) < 1024:
            #    print(x)
            #    print("")
            #    print(y)
            #    input("Press Enter to continue...")
            input_sequences_encoded = input_sequences_encoded + x
            output_sequences_encoded = output_sequences_encoded + y

        # print(input_sequences_encoded[0])
        # print(output_sequences_encoded[0])
        # print(np.array(input_sequences_encoded).shape)
        # print(np.array(output_sequences_encoded).shape)

        # print(input_sequences_encoded)
        # for i,s in enumerate (input_sequences_encoded):
        #    print(s)
        #    print("")
        #    print(i)
        #    input("Press Enter to continue...")

        # for s in output_sequences_encoded:
        #    if len(s) < 1024:
        #        print(s)
        # input("Press Enter to continue...")

        print("Pad sequences in case")
        self.train_x_encoded = pad_sequences(
            input_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )
        # input("Press Enter to continue...")
        self.train_y_encoded = pad_sequences(
            output_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )

    def load_vocabulay_file(self, vocab_filename):
        with open(vocab_filename) as json_file:
            return json.load(json_file)

    def data_preprocessing_6(self):
        input_sequences_encoded = self.train_x
        output_sequences_encoded = self.train_y

        self.wordDict = self.load_vocabulay_file(
            self.experiment_parameters["word_dict"]
        )

        print("Pad sequences in case")
        self.train_x_encoded = pad_sequences(
            input_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )
        # input("Press Enter to continue...")
        self.train_y_encoded = pad_sequences(
            output_sequences_encoded,
            maxlen=self.experiment_parameters["bloc_size"],
            padding="post",
            value=0.0,
        )

        print(self.train_x_encoded.shape)
        print(self.train_y_encoded.shape)

        input("Press Enter to continue...")

    def create_model(self):
        vocab_size = (
            len(list(self.wordDict.keys())) + 1
        )  # Only consider the top 20k words
        # vocab_size = 193
        # print(vocab_size)
        # input("Press Enter to continue...")
        maxlen = self.experiment_parameters["bloc_size"]  # Max sequence size
        embed_dim = self.experiment_parameters[
            "embedding_size"
        ]  # Embedding size for each token
        num_heads = self.experiment_parameters["num_heads"]  # Number of attention heads
        dropout_rate = self.experiment_parameters["dropout"]
        num_layers = self.experiment_parameters["num_layers"]
        activation = self.experiment_parameters["activation"]
        normalize_first = self.experiment_parameters["normalize_first"]

        inputs = Input(shape=(maxlen,))
        # embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=maxlen,
            embedding_dim=embed_dim,
            mask_zero=True,
        )
        ln_f = LayerNormalization(epsilon=1e-5)

        x = embedding_layer(inputs)

        for _ in range(num_layers):
            decoder_layer = keras_nlp.layers.TransformerDecoder(
                intermediate_dim=embed_dim * 4,
                num_heads=num_heads,
                dropout=dropout_rate,
                activation=activation,
                normalize_first=normalize_first,
            )
            x = decoder_layer(x)  # Giving one argument only skips cross-attention.

        x = ln_f(x)

        outputs = Dense(vocab_size)(x)

        model = Model(inputs=inputs, outputs=outputs, name="GPT")

        # ceate a picture of the model
        picture_name = model.name + "_" + self.experiment_tag + ".png"
        picture_path = os.path.join(self.experiment_result_path, picture_name)

        plot_model(model, show_shapes=True, to_file=picture_path)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=cyclical_learning_rate)
        optimizer = tf.keras.optimizers.Adam()
        perp = keras_nlp.metrics.Perplexity(from_logits=True, name="perplexity")

        model.compile(
            # optimizer =optimizer, loss=[loss_fn, None], metrics = [perplexity]
            optimizer=optimizer,
            loss=[loss_fn, None],
            metrics=[perp],
        )  # No loss and optimization based on word embeddings from transformer block

        # print summary
        print(model.summary())

        return model

    def train(self, final_model_path):
        root_logdir = os.path.join(
            self.experiment_parameters["name"],
            "logs_{}_{}".format(self.experiment_parameters["name"], self.dataset_name),
        )

        run_id = (
            self.model.name + "_" + self.experiment_tag + "_" + str(self.current_time)
        )
        log_dir = os.path.join(root_logdir, run_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        csv_name = self.model.name + "_" + self.experiment_tag + ".csv"
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
            final_model_path,
            monitor="val_perplexity",
            mode="min",
            verbose=1,
            save_best_only=True,
        )

        # cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
        cbs = [csv_logger, tensorboard_cb, mc, es]

        if self.dataset == None:
            self.model.fit(
                self.train_x_encoded,
                self.train_y_encoded,
                epochs=self.experiment_parameters["epoch_number"],
                batch_size=self.experiment_parameters["batch_size"],
                verbose=self.experiment_parameters["verbose"],
                callbacks=cbs,
                validation_split=0.2,
                shuffle=True,
            )
        else:
            train_size = int(0.8 * self.dataset_size)

            train_dataset = self.dataset.take(train_size)
            test_dataset = self.dataset.skip(train_size)
            self.model.fit(
                train_dataset,
                epochs=self.experiment_parameters["epoch_number"],
                verbose=self.experiment_parameters["verbose"],
                callbacks=cbs,
                validation_data=test_dataset,
                batch_size=self.experiment_parameters["batch_size"],
            )

    def start(self):
        # Star time of the experiment
        self.current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(
            self.experiment_parameters["name"],
            self.experiment_parameters["model_type"],
            self.dataset_name,
            "run_" + str(self.current_time) + "_" + self.experiment_tag,
        )

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        filename_base = "GPT_basic_raw_{}_{}_{}".format(
            self.dataset_name,
            self.experiment_parameters["bloc_size"],
            self.experiment_parameters["embedding_size"],
        )

        # self.data_preprocessing()
        # self.data_preprocessing_2()
        # self.data_preprocessing_3()
        # self.data_preprocessing_4()
        self.data_preprocessing_5()  # the one to use for normal GPT
        # self.data_preprocessing_6()  # with separator

        vocabulary_filename = filename_base + "_dict_vocabulary.json"
        final_vocabulary_path = os.path.join(
            self.experiment_result_path, vocabulary_filename
        )

        with open(final_vocabulary_path, "w") as save_vocab_file:
            json.dump(self.wordDict, save_vocab_file, indent=4)

        self.save_config()

        self.model = self.create_model()

        if self.DEBUG:
            print("")
            print(list(self.wordDict.keys()))

            input("Press Enter to continue...")

        print("Start Training...")

        model_filename = filename_base + "_model.h5"
        final_model_path = os.path.join(self.experiment_result_path, model_filename)

        self.train(final_model_path)

        print("Training Finish")

    def __save_dict_to_json(self, where_to_save, dict_to_save):
        with open(where_to_save, "w") as json_dict_file:
            json.dump(dict_to_save, json_dict_file, indent=4)

    def save_config(self):
        experiment_parameters_name = "experiment_parameters.json"
        experiment_parameters_path = os.path.join(
            self.experiment_result_path, experiment_parameters_name
        )

        self.__save_dict_to_json(experiment_parameters_path, self.experiment_parameters)
