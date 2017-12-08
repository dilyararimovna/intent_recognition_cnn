from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))
import fasttext
from intent_models import cnn_word_model
from intent_recognizer_class import IntentRecognizer
import os
import numpy as np
import re

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

class IntentRecognizerDSTC2(object):

    def __init__(self, dpath):
        fasttext_model_file = os.path.join(dpath, 'dstc2_fasttext_model_100.bin')
        self.fasttext_model = fasttext.load_model(fasttext_model_file)
        self.confident_threshold = 0.5
        self.text_size = 15
        self.embedding_size = 100
        self.kernel_sizes = [1, 2, 3]
        f = open(os.path.join(dpath, 'intents.txt'), 'r')
        lines = f.readlines()
        self.intents = np.array([line.rstrip() for line in lines])
        self.n_classes = self.intents.shape[0]
        f.close()
        print('\n___Considered intents___\n', self.intents)
        self.Recognizer = IntentRecognizer(self.intents, fasttext_embedding_model=self.fasttext_model, n_splits=1)

        self.Recognizer.init_network_parameters([{'coef_reg_cnn': 0.00028397089135487048,
                                                    'coef_reg_den': 0.0048999053790332739,
                                                    'filters_cnn': 170,
                                                    'dense_size': 141, 'dropout_rate': 0.4051841045170547}])
        self.Recognizer.init_learning_parameters([{'batch_size': 30,
                                                     'lear_rate': 0.24946701435260424,
                                                     'lear_rate_decay': 0.36829336803894619,
                                                     'epochs': 68}])

        self.Recognizer.init_model(cnn_word_model, self.text_size, self.embedding_size,
                                   self.kernel_sizes, add_network_params=None)
        self.Recognizer.load_model(fname=os.path.join(dpath, 'best_model__dstc2_1'),
                                   loss='binary_crossentropy')
        print('___Intent recognizer initialized___')

    def predict(self, requests):
        train_predictions = self.Recognizer.predict([requests])[0]
        return train_predictions

    def _predictions2text(self, predictions):
        y = []
        for sample in predictions:
            to_add = np.where(sample > self.confident_threshold)[0]
            if len(to_add) > 0:
                y.append(self.intents[to_add])
            else:
                y.append([self.intents[np.argmax(sample)]])
        y = np.asarray(y)
        return y

    def _text2predictions(self, predictions):
        eye = np.eye(self.n_classes)
        y = []
        for sample in predictions:
            curr = np.zeros(self.n_classes)
            for intent in sample:
                curr += eye[np.where(self.intents == intent)[0]].reshape(-1)
            y.append(curr)
        y = np.asarray(y)
        return y

    def _proba2labels(self, predictions):
        labels = predictions.copy()
        for i in range(predictions.shape[0]):
            labels[i, :] = 1 * (labels[i, :] > self.confident_threshold)
        return labels

