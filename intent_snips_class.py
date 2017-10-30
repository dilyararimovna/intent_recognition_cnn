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
VERSION = 0

train_data = []

train_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_train_0"))
train_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_train_1"))
train_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_train_2"))

test_data = []

test_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_test_0"))
test_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_test_1"))
test_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_test_2"))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 5
kernel_sizes=[1,2,3]

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

if FIND_BEST_PARAMS:
    print("___TO FIND APPROPRIATE PARAMETERS____")
    train_requests = [train_data[i].loc[:,'request'].values for i in range(3)]
    train_classes = [train_data[i].loc[:,intents].values for i in range(3)]
    test_requests = [test_data[i].loc[:, 'request'].values for i in range(3)]
    test_classes = [test_data[i].loc[:, intents].values for i in range(3)]

    FindBestRecognizer = IntentRecognizer(intents, train_requests, train_classes, cnn_word_model,
                                          fasttext_embedding_model=fasttext_model,
                                          to_use_kfold=False, n_splits=None)

    best_mean_f1 = 0.
    best_network_params = dict()
    best_learning_params = dict()
    params_f1 = []

    while 1:
        FindBestRecognizer.gener_network_parameters(coef_reg_cnn={'range': [0.0001,0.01], 'scale': 'log'},
                                                    coef_reg_den={'range': [0.0001,0.01], 'scale': 'log'},
                                                    filters_cnn={'range': [200,300], 'discrete': True},
                                                    dense_size={'range': [50,100], 'discrete': True},
                                                    dropout_rate={'range': [0.4,0.6]})
        FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                     lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                                     lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                                     epochs={'range': [20,50], 'discrete': True, 'scale': 'log'})
        FindBestRecognizer.fit_model(text_size, embedding_size, kernel_sizes, verbose=True)

        train_predictions = FindBestRecognizer.predict(train_requests)
        FindBestRecognizer.report(np.vstack([train_classes[i] for i in range(3)]),
                                  np.vstack([train_predictions[i] for i in range(3)]),
                                  mode='TRAIN')

        test_predictions = FindBestRecognizer.predict(test_requests)


        f1_test = FindBestRecognizer.report(np.vstack([test_classes[i] for i in range(3)]),
                                            np.vstack([test_predictions[i] for i in range(3)]),
                                            mode='TEST')
        mean_f1 = np.mean(f1_test)


        params_dict = FindBestRecognizer.all_params_to_dict()
        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)
        print(params_f1)
        params_f1_dataframe = pd.DataFrame(params_f1)
        params_f1_dataframe.to_csv("/home/dilyara/data/outputs/intent_snips/depend_" + str(VERSION) + '.txt')

        if mean_f1 > best_mean_f1:
            FindBestRecognizer.save_models(fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_' + str(VERSION))
            print('___BETTER PARAMETERS FOUND!___\n')
            print('___THESE PARAMETERS ARE:___', params_dict)
            best_mean_f1 = mean_f1


if AVERAGE_FOR_PARAMS:
    print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")
    train_requests = [train_data[i].loc[:, 'request'].values for i in range(3)]
    train_classes = [train_data[i].loc[:, intents].values for i in range(3)]
    test_requests = [test_data[i].loc[:, 'request'].values for i in range(3)]
    test_classes = [test_data[i].loc[:, intents].values for i in range(3)]

    AverageRecognizer = IntentRecognizer(intents, train_requests, train_classes, cnn_word_model,
                                          fasttext_embedding_model=fasttext_model,
                                          to_use_kfold=False, n_splits=None)
    f1_scores_for_intents = []

    for p in range(NUM_OF_CALCS):
        AverageRecognizer.init_network_parameters([coef_reg_cnn = 0.0001, coef_reg_den = 0.0001,
                                                   filters_cnn = 200, dense_size = 50, dropout_rate = 0.4],
                                                  [coef_reg_cnn = 0.0001, coef_reg_den = 0.0001,
                                                   filters_cnn = 200, dense_size = 50, dropout_rate = 0.4],
                                                  [coef_reg_cnn = 0.0001, coef_reg_den = 0.0001,
                                                   filters_cnn = 200, dense_size = 50, dropout_rate = 0.4])
        AverageRecognizer.init_learning_parameters([batch_size=16,
                                                   lear_rate=0.01,
                                                   lear_rate_decay=0.01,
                                                   epochs=20)
        AverageRecognizer.fit_model(text_size, embedding_size, kernel_sizes, verbose=True)

        train_predictions = AverageRecognizer.predict(train_requests)
        AverageRecognizer.report(np.vstack([train_classes[i] for i in range(3)]),
                                  np.vstack([train_predictions[i] for i in range(3)]),
                                  mode='TRAIN')

        test_predictions = AverageRecognizer.predict(test_requests)

        f1_test = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(3)]),
                                            np.vstack([test_predictions[i] for i in range(3)]),
                                            mode='TEST')
        mean_f1 = np.mean(f1_test)

        params_dict = AverageRecognizer.all_params_to_dict()
        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)
        print(params_f1)
        params_f1_dataframe = pd.DataFrame(params_f1)
        params_f1_dataframe.to_csv("/home/dilyara/data/outputs/intent_snips/depend_" + str(VERSION) + '.txt')

        if mean_f1 > best_mean_f1:
            AverageRecognizer.save_models(
                fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_' + str(VERSION))
            print('___BETTER PARAMETERS FOUND!___\n')
            print('___THESE PARAMETERS ARE:___', params_dict)
            best_mean_f1 = mean_f1