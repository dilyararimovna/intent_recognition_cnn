from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from dstc2_intent_prediction_function import IntentRecognizerDSTC2
import re

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

def predictions2text(predictions):
    global intents
    y = []
    for sample in predictions:
        to_add = np.where(sample > confident_threshold)[0]
        if len(to_add) > 0:
            y.append(intents[to_add])
        else:
            y.append([intents[np.argmax(sample)]])
    y = np.asarray(y)
    return y

def text2predictions(predictions):
    global intents, n_classes
    eye = np.eye(n_classes)
    y = []
    for sample in predictions:
        curr = np.zeros(n_classes)
        for intent in sample:
            curr += eye[np.where(intents == intent)[0]].reshape(-1)
        y.append(curr)
    y = np.asarray(y)
    return y

def proba2labels(predictions):
    global confident_threshold
    labels = predictions.copy()
    for i in range(predictions.shape[0]):
        labels[i, :] = 1 * (labels[i, :] > confident_threshold)
    return labels

def report(true, predicts):
    global intents
    f1_scores = []
    for i in range(intents.shape[0]):
        m = precision_recall_fscore_support(true[:, i], predicts[:, i], average='binary', pos_label=1)
        f1_scores.append(m[2])
    return f1_scores

dpath = '/home/dilyara/data/data_files/dstc2/dstc2_intent_model'
test_data = pd.read_csv(os.path.join(dpath, 'dstc2_test.csv'))
print(test_data.head())

f = open(os.path.join(dpath, 'intents.txt'), 'r')
lines = f.readlines()
intents = np.array([line.rstrip() for line in lines])
n_classes = intents.shape[0]
f.close()

test_intents = []
for i in range(test_data.shape[0]):
    split = test_data.loc[i, 'intents'].split(' ')
    for j in range(len(split)):
        if split[j] not in intents:
            split[j] = 'unknown'
    test_intents.append(split)

test_requests = test_data.loc[:, 'text'].values
test_classes = text2predictions(test_intents)

text_size = 15
embedding_size = 300
kernel_sizes=[1,2,3]
confident_threshold = 0.5

recognizer = IntentRecognizerDSTC2(dpath=dpath)

test_predictions = recognizer.predict(requests=test_requests)
print(test_classes)
f1_test = report(test_classes, proba2labels(test_predictions))
print('Test  F1-scores per intent: ', f1_test)
mean_f1 = np.mean(f1_test)
print('Test mean F1:', mean_f1)



