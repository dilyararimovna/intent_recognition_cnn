from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics import precision_recall_fscore_support
from metrics import fmeasure
from intent_models import cnn_word_model,cnn_word_model_ner, lstm_word_model
from intent_recognizer_class import IntentRecognizer


import sys, os
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from save_predictions import save_predictions
from sklearn.utils.class_weight import compute_class_weight

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 16
VERSION = '_dstc2_2'

dpath = '/home/dilyara/data/data_files/dstc2'

train_data = pd.read_csv(os.path.join(dpath, 'dstc2_train.csv'))
print(train_data.head())
valid_data = pd.read_csv(os.path.join(dpath, 'dstc2_valid.csv'))
print(valid_data.head())
test_data = pd.read_csv(os.path.join(dpath, 'dstc2_test.csv'))
print(test_data.head())

train_data = pd.concat([train_data, valid_data], ignore_index=True)
# low_data_intents = ['ack', 'confirm_area', 'confirm_pricerange',
#                     'deny_food', 'deny_name', 'hello',
#                     'inform_name', 'repeat', 'reqmore',
#                     'request_signature', 'restart']
#
# low_data = test_data.loc[test_data['intents'] == low_data_intents[0], :]
# for low_data_intent in low_data_intents[1:]:
#     low_data = pd.concat([low_data, test_data.loc[test_data['intents'] == low_data_intent, :]], ignore_index=True)
# test_data = low_data
# test_data.index = np.arange(test_data.shape[0])
#
# print(test_data.head())

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/dstc2_fasttext_model_300.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

intents = []
train_intents = []
valid_intents = []
test_intents = []

for i in range(train_data.shape[0]):
    split = train_data.loc[i, 'intents'].split(' ')
    intents.extend(split)
    train_intents.append(split)

# for i in range(valid_data.shape[0]):
#     split = valid_data.loc[i, 'intents'].split(' ')
#     intents.extend(split)
#     valid_intents.append(split)
#
list_of_intent = intents
intents.append('unknown')
intents = np.array(list(set(intents)))
print('Considered intents:', intents)
confident_threshold = 0.5
n_classes = intents.shape[0]

class_weight = [compute_class_weight('balanced',
                                     intents,
                                     list_of_intent)]
intents, counts = np.unique(list_of_intent, return_counts=True)
print(counts)
print(intents)
print(intents[np.where(counts < 50)[0]])
# class_weight = None

f = open(os.path.join(dpath, 'intents.txt'), 'w')
f.write('\n'.join(intents))
f.close()

for i in range(test_data.shape[0]):
    split = test_data.loc[i, 'intents'].split(' ')
    for j in range(len(split)):
        if split[j] not in intents:
            split[j] = 'unknown'
    test_intents.append(split)

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

train_requests = train_data.loc[:,'text'].values
train_classes = text2predictions(train_intents)
print('Train:', train_classes.shape, train_requests.shape)
# valid_requests = train_data.loc[:,'text'].values
# valid_classes = text2predictions(valid_intents)
# print('Valid:', valid_classes.shape)
test_requests = test_data.loc[:, 'text'].values
test_classes = text2predictions(test_intents)
print(test_classes)
print('Test:', test_classes.shape)

text_size = 15
embedding_size = 300
kernel_sizes=[1,2,3]

def report(true, predicts):
    global intents
    f1_scores = []
    for i in range(intents.shape[0]):
        m = precision_recall_fscore_support(true[:,i], predicts[:,i], average='binary', pos_label=1)
        f1_scores.append(m[2])
    return f1_scores


if FIND_BEST_PARAMS:
    print("___TO FIND APPROPRIATE PARAMETERS____")

    FindBestRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=1)

    best_mean_f1 = 0.
    best_network_params = dict()
    best_learning_params = dict()
    params_f1 = []

    for p in range(100):
        FindBestRecognizer.gener_network_parameters(coef_reg_cnn={'range': [0.0001,0.01], 'scale': 'log'},
                                                    coef_reg_den={'range': [0.0001,0.01], 'scale': 'log'},
                                                    filters_cnn={'range': [100,300], 'discrete': True},
                                                    dense_size={'range': [100,200], 'discrete': True},
                                                    dropout_rate={'range': [0.4,0.6]})
        FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                     lear_rate={'range': [0.01,1], 'scale': 'log'},
                                                     lear_rate_decay={'range': [0.01,1], 'scale': 'log'},
                                                     epochs={'range': [50, 100], 'discrete': True, 'scale': 'log'})
        FindBestRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

        FindBestRecognizer.fit_model([train_requests],
                                     [train_classes],
                                     verbose=True, to_use_kfold=False,
                                     loss='binary_crossentropy', shuffle=True,
                                     class_weight=class_weight,
                                     patience=5)

        train_predictions = FindBestRecognizer.predict([train_requests])[0]
        f1_scores = report(train_classes, proba2labels(train_predictions))
        print('Train F1-scores per intent: ', f1_scores)
        mean_f1 = np.mean(f1_scores)
        print('Train mean F1:', mean_f1)
        test_predictions = FindBestRecognizer.predict([test_requests])[0]
        # test_l = predictions2text(test_predictions)
        # test_t = test_intents
        # for k in range(test_classes.shape[0]):
        #     print(test_l[k], test_t[k])
        f1_test = report(test_classes, proba2labels(test_predictions))
        print('Test  F1-scores per intent: ', f1_test)
        mean_f1 = np.mean(f1_test)
        print('Test mean F1:', mean_f1)

        params_dict = FindBestRecognizer.all_params_to_dict()

        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)

        params_f1_dataframe = pd.DataFrame(params_f1)
        params_f1_dataframe.to_csv("/home/dilyara/data/outputs/intent_snips/depend_" + VERSION + '.txt')

        if mean_f1 > best_mean_f1:
            FindBestRecognizer.save_models(fname='/home/dilyara/data/models/intent_models/dstc2/best_model_' + VERSION)
            print('___BETTER PARAMETERS FOUND!___\n')
            print('___THESE PARAMETERS ARE:___', params_dict)
            best_mean_f1 = mean_f1


if AVERAGE_FOR_PARAMS:
    print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")

    AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=1)
    f1_scores_for_intents = []

    for p in range(NUM_OF_CALCS):
        # for cnn model
        AverageRecognizer.init_network_parameters([{'coef_reg_cnn': 0.00028397089135487048,
                                                    'coef_reg_den': 0.0048999053790332739,
                                                    'filters_cnn': 170,
                                                    'dense_size': 141, 'dropout_rate': 0.4051841045170547}])
        AverageRecognizer.init_learning_parameters([{'batch_size': 30,
                                                     'lear_rate': 0.24946701435260424,
                                                     'lear_rate_decay': 0.36829336803894619,
                                                     'epochs': 68}])

        AverageRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)
        AverageRecognizer.load_model(fname='/home/dilyara/data/models/intent_models/dstc2/best_model__dstc2_1',
                                     loss='binary_crossentropy')
        # AverageRecognizer.fit_model(train_requests, train_classes, to_use_kfold=False, verbose=True)

        train_predictions = AverageRecognizer.predict([train_requests])[0]
        f1_scores = report(train_classes, proba2labels(train_predictions))
        print('Train F1-scores per intent: ', f1_scores)
        mean_f1 = np.mean(f1_scores)
        print('Train mean F1:', mean_f1)
        test_predictions = AverageRecognizer.predict([test_requests])[0]
        f1_test = report(test_classes, proba2labels(test_predictions))
        print('Test  F1-scores per intent: ', f1_test)
        mean_f1 = np.mean(f1_test)
        print('Test mean F1:', mean_f1)

        f1_scores_for_intents.append(f1_test)

    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))


