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

from intent_models import cnn_word_model, cnn_word_model_ner, cnn_word_model_with_sent_emb
from intent_recognizer_class import IntentRecognizer
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('/home/dilyara/Documents/InferSent/encoder')

import nltk
import cv2,torch

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 2
VERSION = '_softmax_infersent_findbest_0'

train_data = []

path_to_snips_data = "/home/dilyara/data/data_files/snips/"

# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_0"))
# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_1"))
# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_2"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_0.csv"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_1.csv"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_2.csv"))


test_data = []

test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_0.csv"))
test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_1.csv"))
test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_2.csv"))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------INFERSENT-----------------

infersent = torch.load('/home/dilyara/Documents/InferSent/encoder/infersent.allnli.pickle',map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
infersent.set_glove_path('/home/dilyara/Documents/InferSent/dataset/GloVe/glove.840B.300d.txt')
texts = []
for i in range(3):
    texts.extend(train_data[i].loc[:,'request'].values)
infersent.build_vocab(texts, tokenize=True)
del texts

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 3
kernel_sizes=[1,2,3]
sent_embedding_size = 4096

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------
train_requests = [train_data[i].loc[:,'request'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:,intents].values for i in range(n_splits)]
test_requests = [test_data[i].loc[:, 'request'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:, intents].values for i in range(n_splits)]

sent_emb_train = [infersent.encode(train_requests[model_ind], tokenize=True)
                  for model_ind in range(n_splits)]
sent_emb_test = [infersent.encode(test_requests[model_ind], tokenize=True)
                 for model_ind in range(n_splits)]


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
                                                    filters_cnn={'range': [200,300], 'discrete': True},
                                                    dense_size={'range': [50,150], 'discrete': True},
                                                    dropout_rate={'range': [0.4,0.6]})
        FindBestRecognizer.gener_learning_parameters(batch_size={'range': [16,64], 'discrete': True},
                                                     lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                                     lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                                     epochs={'range': [20,50], 'discrete': True, 'scale': 'log'})
        FindBestRecognizer.init_model(cnn_word_model_with_sent_emb, text_size, embedding_size, kernel_sizes,
                                      add_network_params={'sent_embedding_size': sent_embedding_size})

        FindBestRecognizer.fit_model(train_requests, train_classes, verbose=True, to_use_kfold=False,
                                     add_inputs=sent_emb_train)

        train_predictions = FindBestRecognizer.predict(train_requests, add_inputs=sent_emb_train)
        FindBestRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                  np.vstack([train_predictions[i] for i in range(n_splits)]),
                                  mode='TRAIN')

        test_predictions = FindBestRecognizer.predict(test_requests, add_inputs=sent_emb_test)


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
        AverageRecognizer.init_network_parameters([{'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                   'filters_cnn': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4},
                                                   {'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                    'filters_cnn': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4},
                                                   {'coef_reg_cnn': 0.0001, 'coef_reg_den': 0.0001,
                                                    'filters_cnn': 200,
                                                    'dense_size': 50, 'dropout_rate': 0.4}])
        AverageRecognizer.init_learning_parameters([{'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20},
                                                    {'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20},
                                                    {'batch_size': 16, 'lear_rate':0.01,
                                                     'lear_rate_decay': 0.01, 'epochs': 20}])
        AverageRecognizer.init_model(cnn_word_model_with_sent_emb, text_size, embedding_size, kernel_sizes,
                                      add_network_params={'sent_embedding_size': sent_embedding_size})

        AverageRecognizer.fit_model(train_requests, train_classes, verbose=True, to_use_kfold=False,
                                     add_inputs=sent_emb_train)

        train_predictions = AverageRecognizer.predict(train_requests, add_inputs=sent_emb_train)
        AverageRecognizer.report(np.vstack([train_classes[i] for i in range(n_splits)]),
                                  np.vstack([train_predictions[i] for i in range(n_splits)]),
                                  mode='TRAIN')

        test_predictions = AverageRecognizer.predict(test_requests, add_inputs=sent_emb_test)

        f1_test = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(n_splits)]),
                                           np.vstack([test_predictions[i] for i in range(n_splits)]),
                                           mode='TEST')
        f1_scores_for_intents.append(f1_test)
    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))

