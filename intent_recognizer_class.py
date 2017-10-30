import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np

import sklearn.model_selection

import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import json
from keras import backend as K
import fasttext

from metrics import fmeasure
from fasttext_embeddings import text2embeddings
from intent_models import cnn_word_model, cnn_word_model_ner, cnn_word_model_with_sent_emb, cnn_word_model_glove
from report_intent import report
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from save_predictions import save_predictions

class IntentRecognizer(object):

    # IF to_use_kfold = True:
    # data - list or array of strings-request len = N_samples
    # classes - np.array of one-hot classes N_samples x n_classes

    # IF to_use_kfold = False:
    # data - list of lists or arrays of strings-request len = N_samples
    # classes - list of arrays of one-hot classes N_samples x n_classes

    def __init__(self, intents, data, classes, model_function, to_use_kfold=False, n_splits=None, fasttext_embedding_model=None):

        self.intents = intents
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.network_parameters = None
        self.learning_parameters = None
        self.n_classes = len(intents)

        if to_use_kfold == True:
            self.n_splits = n_splits
            print("___Stratified splitting data___")
            stratif_y = [np.nonzero(classes[j].values)[0][0] for j in range(data.shape[0])]
            kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
            kf_split.get_n_splits(data, stratif_y)
            for train_index, test_index in kf_split.split(data, stratif_y):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = classes[train_index], classes[test_index]
                self.X_train.append(X_train)
                self.X_test.append(X_test)
                self.y_train.append(y_train)
                self.y_test.append(y_test)
        else:
            #this way data is a list of dataframes
            self.n_splits = len(data)
            print("___Given %d splits of train data___" % self.n_splits)
            for i in range(self.n_splits):
                X_train = data[i]
                y_train = classes[i]
                self.X_train.append(X_train)
                self.y_train.append(y_train)

        if fasttext_embedding_model is not None:
            print("___Fasttext embedding model is loaded___")
            self.fasttext_embedding_model = fasttext_embedding_model

        self.model_function = model_function
        print("___Initialized____")

    def gener_network_parameters(self, **kwargs):
        print("___Considered network parameters___")
        self.network_parameters = []
        for i in range(self.n_splits):
            self.network_parameters.append(param_gen(**kwargs))    #generated dict
        return True

    def gener_learning_parameters(self, **kwargs):
        print("___Considered learning parameters___")
        self.learning_parameters = []
        for i in range(self.n_splits):
            self.learning_parameters.append(param_gen(**kwargs))   #generated dict
        return True

    def init_network_parameters(self, **kwargs):
        print("___Considered network parameters___")
        self.network_parameters = [kwargs]                 #dict
        return True

    def init_learning_parameters(self, **kwargs):
        print("___Considered learning parameters___")
        self.learning_parameters = [kwargs]                #dict
        return True

    def fit_model(self, text_size, embedding_size, kernel_sizes, verbose=True):
        print("___Fitting model___")
        if self.network_parameters is None or self.learning_parameters is None:
            print("___ERROR: network and learning parameters are not given___")
            exit(1)

        self.text_size = text_size
        self.embedding_size = embedding_size
        self.kernel_sizes = kernel_sizes
        self.histories = []
        self.models = []

        for model_ind in range(self.n_splits):
            if self.fasttext_embedding_model is not None:
                X_train_embed = text2embeddings(self.X_train[model_ind], self.fasttext_embedding_model, self.text_size, self.embedding_size)
            self.models.append(init_from_scratch(self.model_function, text_size=self.text_size, n_classes=self.n_classes,
                                                 embedding_size=self.embedding_size,
                                                 kernel_sizes=self.kernel_sizes, **(self.network_parameters[model_ind])))
            optimizer = Adam(lr=self.learning_parameters[model_ind]['lear_rate'], decay=self.learning_parameters[model_ind]['lear_rate_decay'])
            self.models[model_ind].compile(loss='categorical_crossentropy',
                                           optimizer=optimizer,
                                           metrics=['categorical_accuracy',
                                           fmeasure])
            self.histories.append(self.models[model_ind].fit(X_train_embed, self.y_train[model_ind].reshape(-1, 7),
                                                            batch_size=self.learning_parameters[model_ind]['batch_size'],
                                                            epochs=self.learning_parameters[model_ind]['epochs'],
                                                            validation_split=0.1,
                                                            verbose=2 * verbose,
                                                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0)]))
        return True

    def predict(self, data=None):
        print("___Predictions___")
        if data is not None:
            X_test = data

        predictions = []

        if self.fasttext_embedding_model is not None:
            for model_ind in range(self.n_splits):
                X_test_embed = text2embeddings(X_test[model_ind], self.fasttext_embedding_model,
                                               self.text_size, self.embedding_size)
                predictions.append(self.models[model_ind].predict(X_test_embed).reshape(-1, self.n_classes))
            return predictions
        else:
            ('Error: No embeddings provided\n')
            return False

    def report(self, true, predicts, mode=None):
        print("___Report___")
        if mode is not None:
            print("___MODE is %s___" % mode)
        print("%s \t %s \t%s \t %s \t %s" % ('type', 'precision', 'recall', 'f1-score', 'support'))
        f1_scores = []
        for ind, intent in enumerate(self.intents):
            scores = np.asarray(precision_recall_fscore_support(true[:, ind], predicts[:, ind]))[:, 1]
            print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))
            f1_scores.append(scores[2])
        return(f1_scores)












