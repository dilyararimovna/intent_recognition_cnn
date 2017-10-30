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

from intent_models import cnn_word_model,cnn_word_model_ner
from intent_recognizer_class import IntentRecognizer
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 2
VERSION = '_softmax_ner_findbest_0'

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
n_splits = 3
kernel_sizes=[1,2,3]
tag_size = 40

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------
train_requests = [train_data[i].loc[:,'request'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:,intents].values for i in range(n_splits)]
test_requests = [test_data[i].loc[:, 'request'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:, intents].values for i in range(n_splits)]
train_ner = [train_data[i].loc[:, 'ner_tag'].values for i in range(n_splits)]
test_ner = [test_data[i].loc[:, 'ner_tag'].values for i in range(n_splits)]

if FIND_BEST_PARAMS:
    print("___TO FIND APPROPRIATE PARAMETERS____")

    FindBestRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)

    best_mean_f1 = 0.
    best_network_params = dict()
    best_learning_params = dict()
    params_f1 = []

    while 1:
        FindBestRecognizer.gener_network_parameters(coef_reg_cnn_tag={'range': [0.0001,0.01], 'scale': 'log'},
                                                    coef_reg_cnn_emb={'range': [0.0001, 0.01],  'scale': 'log'},
                                                    coef_reg_den={'range': [0.0001,0.01], 'scale': 'log'},
                                                    filters_cnn_emb={'range': [200,300], 'discrete': True},
                                                    filters_cnn_tag={'range': [100,200], 'discrete': True},
                                                    dense_size={'range': [50,150], 'discrete': True},
                                                    dropout_rate={'range': [0.4,0.6]})
        FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                     lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                                     lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                                     epochs={'range': [20,50], 'discrete': True, 'scale': 'log'})
        FindBestRecognizer.init_model(cnn_word_model_ner, text_size, embedding_size, kernel_sizes,
                                      add_network_params={'tag_size': tag_size})

        list_of_tag_tables = FindBestRecognizer.get_tag_table(ner_data=train_ner, tag_size=tag_size)
        train_tags = [pad_sequences(list_of_tag_tables[i], maxlen=text_size,
                                    padding='pre') for i in range(n_splits)]

        FindBestRecognizer.fit_model(train_requests, train_classes, verbose=True, to_use_kfold=False,
                                     add_inputs=train_tags)

        train_predictions = FindBestRecognizer.predict(train_requests, add_inputs=train_tags)
        FindBestRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                  np.vstack([train_predictions[i] for i in range(n_splits)]),
                                  mode='TRAIN')

        list_of_tag_tables = FindBestRecognizer.get_tag_table(ner_data=test_ner, tag_size=tag_size)
        test_tags = [pad_sequences(list_of_tag_tables[i], maxlen=text_size,
                                    padding='pre') for i in range(n_splits)]
        test_predictions = FindBestRecognizer.predict(test_requests, add_inputs=test_tags)


        f1_test = FindBestRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                            np.vstack([test_predictions[i] for i in range(n_splits)]),
                                            mode='TEST')
        mean_f1 = np.mean(f1_test)


        params_dict = FindBestRecognizer.all_params_to_dict()
        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)
        print(params_f1)
        params_f1_dataframe = pd.DataFrame(params_f1)
        params_f1_dataframe.to_csv("/home/dilyara/data/outputs/intent_snips/depend_" + VERSION + '.txt')

        if mean_f1 > best_mean_f1:
            FindBestRecognizer.save_models(fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_' + VERSION)
            print('___BETTER PARAMETERS FOUND!___\n')
            print('___THESE PARAMETERS ARE:___', params_dict)
            best_mean_f1 = mean_f1

if AVERAGE_FOR_PARAMS:
    print("___TO CALCULATE AVERAGE ACCURACY FOR PARAMETERS____")

    AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=n_splits)
    f1_scores_for_intents = []

    for p in range(NUM_OF_CALCS):
        AverageRecognizer.init_network_parameters([{'coef_reg_cnn_tag': 0.0001, 'coef_reg_cnn_emb':0.0001,
                                                    'coef_reg_den': 0.0001,
                                                   'filters_cnn_tag': 200, 'filters_cnn_emb': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4},
                                                   {'coef_reg_cnn_tag': 0.0001, 'coef_reg_cnn_emb': 0.0001,
                                                    'coef_reg_den': 0.0001,
                                                    'filters_cnn_tag': 200, 'filters_cnn_emb': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4},
                                                   {'coef_reg_cnn_tag': 0.0001, 'coef_reg_cnn_emb': 0.0001,
                                                    'coef_reg_den': 0.0001,
                                                    'filters_cnn_tag': 200, 'filters_cnn_emb': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4}])
        AverageRecognizer.init_learning_parameters([{'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20},
                                                    {'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20},
                                                    {'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20}])
        AverageRecognizer.init_model(cnn_word_model_ner, text_size, embedding_size, kernel_sizes,
                                      add_network_params={'tag_size': tag_size})

        list_of_tag_tables = AverageRecognizer.get_tag_table(ner_data=train_ner, tag_size=tag_size)
        train_tags = [pad_sequences(list_of_tag_tables[i], maxlen=text_size,
                                    padding='pre') for i in range(n_splits)]

        AverageRecognizer.fit_model(train_requests, train_classes, verbose=True, to_use_kfold=False,
                                     add_inputs=train_tags)

        train_predictions = AverageRecognizer.predict(train_requests, add_inputs=train_tags)
        AverageRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                  np.vstack([train_predictions[i] for i in range(n_splits)]),
                                  mode='TRAIN')

        list_of_tag_tables = AverageRecognizer.get_tag_table(ner_data=test_ner, tag_size=tag_size)
        test_tags = [pad_sequences(list_of_tag_tables[i], maxlen=text_size,
                                    padding='pre') for i in range(n_splits)]
        test_predictions = AverageRecognizer.predict(test_requests, add_inputs=test_tags)

        f1_test = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                           np.vstack([test_predictions[i] for i in range(n_splits)]),
                                           mode='TEST')
        f1_scores_for_intents.append(f1_test)
    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))
