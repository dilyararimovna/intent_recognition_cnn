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


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 16
VERSION = '_findbest_byparts_paraphrases_2_nobpe'

path = '/home/dilyara/data/data_files/snips'

train_data = []

train_data.append(pd.read_csv(os.path.join(path, 'paraphrases/intent_train_data_with_paraphrases.csv')))

test_data = []

test_data.append(pd.read_csv(os.path.join(path, 'paraphrases/intent_test_data_with_paraphrases.csv')))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 1
kernel_sizes=[1,2,3]
train_sizes = [1000] # per intent
# train_sizes = [10, 25, 50] # per intent


intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

train_requests = [train_data[i].loc[:,'request'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:,intents].values for i in range(n_splits)]
test_requests = [test_data[i].loc[:2761, 'request'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:2761, intents].values for i in range(n_splits)]

for train_size in train_sizes:
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
        print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")

        AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)
        f1_scores_for_intents = []

        for p in range(NUM_OF_CALCS):
            AverageRecognizer.init_network_parameters([{'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                       'filters_cnn': 200, 'dense_size': 50, 'dropout_rate': 0.4},
                                                       {'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                       'filters_cnn': 200, 'dense_size': 50, 'dropout_rate': 0.4},
                                                       {'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                       'filters_cnn': 200, 'dense_size': 50, 'dropout_rate': 0.4}])
            AverageRecognizer.init_learning_parameters([{'batch_size': 16, 'lear_rate':0.01,
                                                         'lear_rate_decay': 0.01, 'epochs': 20},
                                                        {'batch_size': 16, 'lear_rate':0.01,
                                                         'lear_rate_decay': 0.01, 'epochs': 20},
                                                        {'batch_size': 16, 'lear_rate':0.01,
                                                         'lear_rate_decay': 0.01, 'epochs': 20}])
            AverageRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

            AverageRecognizer.fit_model(train_requests_part, train_classes_part, to_use_kfold=False, verbose=True)

            train_predictions = AverageRecognizer.predict(train_requests_part)
            AverageRecognizer.report(np.vstack([train_classes_part[i] for i in range(n_splits)]),
                                     np.vstack([train_predictions[i] for i in range(n_splits)]),
                                     mode='TRAIN')

            test_predictions = AverageRecognizer.predict(test_requests)

            f1_test = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                               np.vstack([test_predictions[i] for i in range(n_splits)]),
                                               mode='TEST')[0]
            f1_scores_for_intents.append(f1_test)
        f1_scores_for_intents = np.asarray(f1_scores_for_intents)
        for intent_id in range(len(intents)):
            f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
            f1_std = np.std(f1_scores_for_intents[:,intent_id])
            print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))

