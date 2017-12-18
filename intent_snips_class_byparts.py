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

from metrics import fmeasure
from intent_models import cnn_word_model,cnn_word_model_ner
from intent_recognizer_class import IntentRecognizer

import sys, os
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from save_predictions import save_predictions

SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = False
AVERAGE_FOR_PARAMS = True
NUM_OF_CALCS = 16
VERSION = '_findbest_byparts_paraphrases_2_nobpe'

path = '/home/dilyara/data/data_files/snips'

train_data = []

train_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_train_0"))

test_data = []

test_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_test_0"))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)
#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 1
kernel_sizes=[1,2,3]
train_sizes = [10, 25, 50, 100, 200, 500, 1000] # per intent
# train_sizes = [10, 25, 50] # per intent


intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

train_requests = [train_data[i].loc[:,'request'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:,intents].values for i in range(n_splits)]
test_requests = [test_data[i].loc[:2761, 'request'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:2761, intents].values for i in range(n_splits)]

f1_mean_per_size = []
f1_std_per_size = []

for n_size, train_size in enumerate(train_sizes):
    print("\n\n______NUMBER OF TRAIN SAMPLES PER INTENT = %d___________" % train_size)
    train_index_parts = []
    for model_ind in range(n_splits):
        train_part = []
        for i, intent in enumerate(intents):
            samples_intent = np.nonzero(train_classes[model_ind][:,i])[0]
            train_part.extend(list(np.random.choice(samples_intent, size=train_size)))
        train_index_parts.append(train_part)

    train_requests_part = [train_requests[model_ind][train_index_parts[model_ind]] for model_ind in range(n_splits)]
    train_classes_part = [train_classes[model_ind][train_index_parts[model_ind]] for model_ind in range(n_splits)]

    if FIND_BEST_PARAMS:
        print("___TO FIND APPROPRIATE PARAMETERS____")

        FindBestRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)

        best_mean_f1 = 0.
        best_network_params = dict()
        best_learning_params = dict()
        params_f1 = []

        for p in range(20):
            FindBestRecognizer.gener_network_parameters(coef_reg_cnn={'range': [0.0001,0.01], 'scale': 'log'},
                                                        coef_reg_den={'range': [0.0001,0.01], 'scale': 'log'},
                                                        filters_cnn={'range': [50,200], 'discrete': True},
                                                        dense_size={'range': [50,200], 'discrete': True},
                                                        dropout_rate={'range': [0.4,0.6]})
            FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                         lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                                         lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                                         epochs={'range': [50,100], 'discrete': True, 'scale': 'log'})
            FindBestRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

            FindBestRecognizer.fit_model(train_requests_part, train_classes_part, verbose=True, to_use_kfold=False)

            train_predictions = FindBestRecognizer.predict(train_requests_part)
            FindBestRecognizer.report(np.vstack([train_classes_part[i] for i in range(n_splits)]),
                                      np.vstack([train_predictions[i] for i in range(n_splits)]),
                                      mode='TRAIN')

            test_predictions = FindBestRecognizer.predict(test_requests)


            f1_test = FindBestRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                                np.vstack([test_predictions[i] for i in range(n_splits)]),
                                                mode='TEST')[0]
            mean_f1 = np.mean(f1_test)

            params_dict = FindBestRecognizer.all_params_to_dict()
            params_dict['mean_f1'] = mean_f1
            params_f1.append(params_dict)
            params_f1_dataframe = pd.DataFrame(params_f1)
            params_f1_dataframe.to_csv("/home/dilyara/data/outputs/intent_snips/depend_" +
                                       VERSION + '_' + str(train_size) + '.txt')

            if mean_f1 > best_mean_f1:
                FindBestRecognizer.save_models(fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_' +
                                                     VERSION + '_' + str(train_size))
                print('___BETTER PARAMETERS FOUND!___\n')
                print('___THESE PARAMETERS ARE:___', params_dict)
                best_mean_f1 = mean_f1

    if AVERAGE_FOR_PARAMS:
        params = [
            # 10
            [{'coef_reg_cnn': 0.0002240188358941768,
              'coef_reg_den': 0.00013254278511375586,
              'filters_cnn': 220,
              'dense_size': 80,
              'dropout_rate': 0.439508706178354},
             {'batch_size': 17,
              'lear_rate': 0.014911813954885302,
              'lear_rate_decay': 0.011552169958875022,
              'epochs': 22}],
            # 25
            [{'coef_reg_cnn': 0.00012373572818256555,
              'coef_reg_den': 0.00017171259810186691,
              'filters_cnn': 202,
              'dense_size': 67,
              'dropout_rate': 0.5603207356574003},
             {'batch_size': 26,
              'lear_rate': 0.054040612295756969,
              'lear_rate_decay': 0.084926115338805563,
              'epochs': 24}],
            # 50
            [{'coef_reg_cnn': 0.00015919311850687678,
              'coef_reg_den': 0.00016115679404622989,
              'filters_cnn': 290,
              'dense_size': 54,
              'dropout_rate': 0.5852312361349971},
             {'batch_size': 23,
              'lear_rate': 0.048151980276947157,
              'lear_rate_decay': 0.029064116214377402,
              'epochs': 33}],
            # 100
            [{'coef_reg_cnn': 0.00033168959552320646,
              'coef_reg_den': 0.00044867444269376276,
              'filters_cnn': 234,
              'dense_size': 95,
              'dropout_rate': 0.4171426478913063},
             {'batch_size': 32,
              'lear_rate': 0.034295802954288496,
              'lear_rate_decay': 0.067480368299883756,
              'epochs': 50}],
            # 200
            [{'coef_reg_cnn': 0.00020510867913527356,
              'coef_reg_den': 0.00030370411016572015,
              'filters_cnn': 277,
              'dense_size': 98,
              'dropout_rate': 0.4986233680859435},
             {'batch_size': 30,
              'lear_rate': 0.021880881947614603,
              'lear_rate_decay': 0.014620662267840959,
              'epochs': 23}],
            # 500
            [{'coef_reg_cnn': 0.00011826989851694623,
              'coef_reg_den': 0.00057033663916566111,
              'filters_cnn': 298,
              'dense_size': 71,
              'dropout_rate': 0.4026373274835373},
             {'batch_size': 21,
              'lear_rate': 0.025750585638000676,
              'lear_rate_decay': 0.023253677502792103,
              'epochs': 34}],
            # 1000
            [{'coef_reg_cnn': 0.00046255365614283103,
              'coef_reg_den': 0.0014098076556438696,
              'filters_cnn': 210,
              'dense_size': 59,
              'dropout_rate': 0.5557728960043049},
             {'batch_size': 28,
              'lear_rate': 0.024490853695736985,
              'lear_rate_decay': 0.028121698403082398,
              'epochs': 47}]]


        print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")
        f1_mean_scores = []
        for p in range(NUM_OF_CALCS):

            AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)
            AverageRecognizer.init_network_parameters([params[n_size][0]])
            AverageRecognizer.init_learning_parameters([params[n_size][1]])
            AverageRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

            AverageRecognizer.fit_model(train_requests_part, train_classes_part, to_use_kfold=False, verbose=True)

            train_predictions = AverageRecognizer.predict(train_requests_part)
            AverageRecognizer.report(np.vstack([train_classes_part[i] for i in range(n_splits)]),
                                     np.vstack([train_predictions[i] for i in range(n_splits)]),
                                     mode='TRAIN')

            test_predictions = AverageRecognizer.predict(test_requests)

            f1_scores = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                               np.vstack([test_predictions[i] for i in range(n_splits)]),
                                               mode='TEST')[0]
            f1_mean_scores.append(np.mean(f1_scores))

        f1_mean_per_size.append(np.mean(f1_mean_scores))
        f1_std_per_size.append(np.std(f1_mean_scores))

        print("___MEAN-STD___:\n size: %d\t f1-mean: %f\tf1-std: %f" % (
        train_size, f1_mean_per_size[n_size], f1_std_per_size[n_size]))

if AVERAGE_FOR_PARAMS:
    for n_size, train_size in enumerate(train_sizes):
        print("size: %d\t f1-mean: %f\tf1-std: %f" % (train_size, f1_mean_per_size[n_size], f1_std_per_size[n_size]))

