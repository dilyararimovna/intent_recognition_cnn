import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
import fasttext
from sklearn.model_selection import train_test_split

from metrics import fmeasure
from intent_models import cnn_word_model_sigmoid
from intent_recognizer_class import IntentRecognizer

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from save_predictions import save_predictions

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 16
VERSION = '_cnn_word_1'

train_data = []
test_data = []

data = pd.read_csv("/home/dilyara/data/data_files/fact_opinion/data.csv", sep='\t')

train, test = train_test_split(data, test_size=0.2)
train_data.append(train.set_index(np.arange(train.shape[0])))

test_data.append(test.set_index(np.arange(test.shape[0])))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/fasttext_model_twitter.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

print('Train data', train_data[0].head())
print('Test data: ', test_data[0].head())

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 1
kernel_sizes=[1,2,3]

intents = ['is_fact', 'is_opinion', 'ignore']

train_requests = [train_data[i].loc[:,'sentence'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:, intents].values for i in range(n_splits)]
test_requests = [test_data[i].loc[:, 'sentence'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:, intents].values for i in range(n_splits)]

if FIND_BEST_PARAMS:
    print("___TO FIND APPROPRIATE PARAMETERS____")

    FindBestRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)

    best_mean_f1 = 0.
    best_network_params = dict()
    best_learning_params = dict()
    params_f1 = []

    for p in range(100):
        FindBestRecognizer.gener_network_parameters(coef_reg_cnn={'range': [0.001,0.1], 'scale': 'log'},
                                                    coef_reg_den={'range': [0.001,0.1], 'scale': 'log'},
                                                    filters_cnn={'range': [100,300], 'discrete': True},
                                                    dense_size={'range': [50,150], 'discrete': True},
                                                    dropout_rate={'range': [0.4,0.6]})
        FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                     lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                                     lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                                     epochs={'range': [100,500], 'discrete': True, 'scale': 'log'})
        FindBestRecognizer.init_model(cnn_word_model_sigmoid, text_size, embedding_size, kernel_sizes,
                                      add_network_params=None)

        FindBestRecognizer.fit_model(train_requests, train_classes, verbose=True, to_use_kfold=False)

        train_predictions = FindBestRecognizer.predict(train_requests)
        FindBestRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                  np.vstack([train_predictions[i] for i in range(n_splits)]),
                                  mode='TRAIN')

        test_predictions = FindBestRecognizer.predict(test_requests)

        f1_test, f1_macro, f1_weighted = FindBestRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                            np.vstack([test_predictions[i] for i in range(n_splits)]),
                                            mode='TEST')
        mean_f1 = np.mean(f1_test)

        params_dict = FindBestRecognizer.all_params_to_dict()

        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)

        params_f1_dataframe = pd.DataFrame(params_f1)
        params_f1_dataframe.to_csv("/home/dilyara/data/outputs/fact_opinion/depend_" + VERSION + '.txt')

        if mean_f1 > best_mean_f1:
            FindBestRecognizer.save_models(fname='/home/dilyara/data/models/fact_opinion/best_model_' + VERSION)
            print('___BETTER PARAMETERS FOUND!___\n')
            print('___THESE PARAMETERS ARE:___', params_dict)
            best_mean_f1 = mean_f1

if AVERAGE_FOR_PARAMS:
    print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")

    AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)
    f1_scores_for_intents = []

    for p in range(NUM_OF_CALCS):
        # for cnn model
        # {'coef_reg_cnn_0': 0.0033318201134740549, 'coef_reg_den_0': 0.0054373095552282986, 'filters_cnn_0': 256,
        #  'dense_size_0': 207, 'dropout_rate_0': 0.5406982976365785, 'batch_size_0': 63,
        #  'lear_rate_0': 0.062165154413520614, 'lear_rate_decay_0': 0.080905421964017704, 'epochs_0': 477,
        #  'mean_f1': 0.64817978951014055} __

        AverageRecognizer.init_network_parameters([{'coef_reg_cnn': 0.00036855026787845302,
                                                    'coef_reg_den': 0.00029720938577019042,
                                                    'filters_cnn': 250,
                                                    'dense_size': 60, 'dropout_rate': 0.4428009151691538},
                                                   {'coef_reg_cnn': 0.00021961271422818286,
                                                    'coef_reg_den': 0.00038807312004670983,
                                                    'filters_cnn': 213,
                                                    'dense_size': 73, 'dropout_rate': 0.4163792890042266},
                                                   {'coef_reg_cnn': 0.0032453791711150072,
                                                    'coef_reg_den': 0.002292512102049529,
                                                    'filters_cnn': 263,
                                                    'dense_size': 56, 'dropout_rate': 0.480199139584279}])
        AverageRecognizer.init_learning_parameters([{'batch_size': 60,
                                                     'lear_rate': 0.0200569566892299,
                                                     'lear_rate_decay': 0.046826433619590428,
                                                     'epochs': 25},
                                                    {'batch_size': 18,
                                                     'lear_rate': 0.047765133582360196,
                                                     'lear_rate_decay': 0.08370914599522486, 'epochs': 38},
                                                    {'batch_size': 21,
                                                     'lear_rate': 0.016858859621336184,
                                                     'lear_rate_decay': 0.037668912128691327,
                                                     'epochs': 45}])

        AverageRecognizer.init_model(lstm_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

        AverageRecognizer.fit_model(train_requests, train_classes, to_use_kfold=False, verbose=True)

        train_predictions = AverageRecognizer.predict(train_requests)
        AverageRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                 np.vstack([train_predictions[i] for i in range(n_splits)]),
                                 mode='TRAIN')

        test_predictions = AverageRecognizer.predict(test_requests)

        f1_test = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                           np.vstack([test_predictions[i] for i in range(n_splits)]),
                                           mode='TEST')
        f1_scores_for_intents.append(f1_test)
    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))

