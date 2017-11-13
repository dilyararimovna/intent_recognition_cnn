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

from intent_models import cnn_word_model, cnn_word_model_ner, cnn_word_model_ner_2
from intent_recognizer_class import IntentRecognizer
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)


FIND_BEST_PARAMS = True
AVERAGE_FOR_PARAMS = False
NUM_OF_CALCS = 16
VERSION = '_softmax_ner_crf_findbest_0'

train_data = []

path_to_snips_data = "/home/dilyara/data/data_files/snips/"

# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_0"))
# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_1"))
# train_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_train_2"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_0.csv"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_1.csv"))
train_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_train_2.csv"))


test_data = []

# test_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_test_0"))
# test_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_test_1"))
# test_data.append(pd.read_csv(path_to_snips_data + "snips_ner_gold/snips_ner_gold_0/snips_test_2"))

test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_0.csv"))
test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_1.csv"))
test_data.append(pd.read_csv(path_to_snips_data + "snips_crf_with_idxs/snips_test_2.csv"))

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

    for p in range(20):
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
                                            mode='TEST')[0]
        mean_f1 = np.mean(f1_test)


        params_dict = FindBestRecognizer.all_params_to_dict()
        params_dict['mean_f1'] = mean_f1
        params_f1.append(params_dict)

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
        #for ner crf
        AverageRecognizer.init_network_parameters([{'coef_reg_cnn_tag': 0.00066350094692529645,
                                                    'coef_reg_cnn_emb': 0.00047323342362332759,
                                                    'coef_reg_den': 0.0034790278642858716,
                                                    'filters_cnn_emb': 226,
                                                    'filters_cnn_tag': 145,
                                                    'dense_size': 117,
                                                    'dropout_rate': 0.4821011382436993},
                                                   {'coef_reg_cnn_tag': 0.00011549422217361988,
                                                    'coef_reg_cnn_emb': 0.0056291264624271538,
                                                    'coef_reg_den': 0.00022686582264503204,
                                                    'filters_cnn_emb': 298,
                                                    'filters_cnn_tag': 135,
                                                    'dense_size': 108,
                                                    'dropout_rate': 0.41812048375665867},
                                                   {'coef_reg_cnn_tag': 0.0081031832570831533,
                                                    'coef_reg_cnn_emb': 0.00017798242982311515,
                                                    'coef_reg_den': 0.0034131887364826454,
                                                    'filters_cnn_emb': 288,
                                                    'filters_cnn_tag': 111,
                                                    'dense_size': 129,
                                                    'dropout_rate': 0.5843603954268347}])
        AverageRecognizer.init_learning_parameters([{'batch_size': 24,
                                                     'lear_rate': 0.01985462354329437,
                                                     'lear_rate_decay': 0.043865053959028635,
                                                     'epochs': 48},
                                                    {'batch_size': 23,
                                                     'lear_rate': 0.021651609596649066,
                                                     'lear_rate_decay': 0.038241618449973806,
                                                     'epochs': 28},
                                                    {'batch_size': 47,
                                                     'lear_rate': 0.031771987829375958,
                                                     'lear_rate_decay': 0.015396011710902565,
                                                     'epochs': 37}])
        #for truth ner
        # AverageRecognizer.init_network_parameters([{'coef_reg_cnn_tag': 0.0031907564045027003,
        #                                             'coef_reg_cnn_emb': 0.00057584937340040025,
        #                                             'coef_reg_den': 0.00024229326256001223,
        #                                             'filters_cnn_emb': 295,
        #                                             'filters_cnn_tag': 108,
        #                                             'dense_size': 119,
        #                                             'dropout_rate': 0.5829969163684273},
        #                                            {'coef_reg_cnn_tag': 0.00048274629929004467,
        #                                             'coef_reg_cnn_emb': 0.0074890849204680351,
        #                                             'coef_reg_den': 0.0014876632446491709,
        #                                             'filters_cnn_emb': 250,
        #                                             'filters_cnn_tag': 126,
        #                                             'dense_size': 118,
        #                                             'dropout_rate': 0.5414202399581685},
        #                                            {'coef_reg_cnn_tag': 0.0004143495041595739,
        #                                             'coef_reg_cnn_emb': 0.0044085152610540178,
        #                                             'coef_reg_den': 0.00077312581675672762,
        #                                             'filters_cnn_emb': 207,
        #                                             'filters_cnn_tag': 102,
        #                                             'dense_size': 57,
        #                                             'dropout_rate': 0.49322297307786034}])
        # AverageRecognizer.init_learning_parameters([{'batch_size': 30,
        #                                              'lear_rate': 0.084277601001224903,
        #                                              'lear_rate_decay': 0.028767270475603395,
        #                                              'epochs': 21},
        #                                             {'batch_size': 46,
        #                                              'lear_rate': 0.033319806657432588,
        #                                              'lear_rate_decay': 0.059555438350099951,
        #                                              'epochs': 28},
        #                                             {'batch_size': 33,
        #                                              'lear_rate': 0.016023687305964939,
        #                                              'lear_rate_decay': 0.045488750345919336,
        #                                              'epochs': 39}])
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
                                           mode='TEST')[0]
        f1_scores_for_intents.append(f1_test)
    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))

