import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np

import sklearn.model_selection

import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import json
from keras import backend as K
import fasttext

from metrics import fmeasure
from fasttext_embeddings import text2embeddings
from intent_models import cnn_word_model
from report_intent import report

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save
from save_predictions import save_predictions

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

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
filters_cnn = 128
kernel_sizes = [1,2,3]
coef_reg_cnn = 0.001
coef_reg_den = 0.01
dense_size = 100
dropout_rate = 0.5
lear_rate = 0.1
lear_rate_decay = 0.1
batch_size = 32
epochs = 50

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

# stratif_y = [np.nonzero(data.loc[j, intents].values)[0][0] for j in range(data.shape[0])]
#
# kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
# kf_split.get_n_splits(data['request'], stratif_y)

# for train_index, test_index in kf_split.split(data['request'], stratif_y):
# X_train, X_test = data.loc[train_index, 'request'], data.loc[test_index, 'request']
# y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

best_mean_f1 = 0.
best_network_params = dict()
best_learning_params = dict()

f1_scores_for_intents = []

dependency_f1_on_params = dict()
dependency_f1_on_params['coer_reg_cnn'] = []
dependency_f1_on_params['coer_feg_den'] = []
dependency_f1_on_params['filters_cnn'] = []
dependency_f1_on_params['dense_size'] = []
dependency_f1_on_params['dropout_rate'] = []
dependency_f1_on_params['batch_size'] = []
dependency_f1_on_params['lear_rate'] = []
dependency_f1_on_params['lear_rate_decay'] = []
dependency_f1_on_params['epochs'] = []
dependency_f1_on_params['mean_f1_score'] = []


while 1:
    network_params = param_gen(coef_reg_cnn={'range': [0.0001,0.01], 'scale': 'log'},
                               coef_reg_den={'range': [0.0001,0.01], 'scale': 'log'},
                               filters_cnn={'range': [200,300], 'discrete': True},
                               dense_size={'range': [50,150], 'discrete': True},
                               dropout_rate={'range': [0.4,0.6]})

    learning_params = param_gen(batch_size={'range': [16,64], 'discrete': True},
                                lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                epochs={'range': [10,50], 'discrete': True, 'scale': 'log'})

    print('\n\nCONSIDERED PARAMETERS: ', network_params)
    print('\n\nCONSIDERED PARAMETERS: ', learning_params)
    train_preds = []
    train_true = []
    test_preds = []
    test_true = []
    ind = 0

    models = []

    for ind in range(3):
        X_train, X_test = train_data[ind].loc[:, 'request'].values, test_data[ind].loc[:, 'request'].values
        y_train, y_test = train_data[ind].loc[:, intents].values, test_data[ind].loc[:, intents].values
        ner_train, ner_test = train_data[ind].loc[:, 'ner_tag'].values, test_data[ind].loc[:, 'ner_tag'].values

        X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
        X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

        models.append(init_from_scratch(cnn_word_model, text_size=text_size,
                      embedding_size=embedding_size,
                      kernel_sizes=kernel_sizes, **network_params))

        # model = init_from_saved(cnn_word_model, fname='/home/dilyara/Documents/GitHub/intent_recognition_cnn/snips_models/best_model',
        #                         text_size=text_size,
        #                         embedding_size=embedding_size,
        #                         kernel_sizes=kernel_sizes, **network_params)

        optimizer = Adam(lr=learning_params['lear_rate'], decay=learning_params['lear_rate_decay'])
        models[ind].compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy',
                               fmeasure])
        history = models[ind].fit(X_train_embed, y_train.reshape(-1, 7),
                            batch_size=learning_params['batch_size'],
                            epochs=learning_params['epochs'],
                            validation_split=0.1,
                            verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0)])

        y_train_pred = models[ind].predict(X_train_embed).reshape(-1, 7)
        train_preds.extend(y_train_pred)
        train_true.extend(y_train)

        y_test_pred = models[ind].predict(X_test_embed).reshape(-1, 7)
        test_preds.extend(y_test_pred)
        test_true.extend(y_test)

        # print("Saving files for Mishanya NER\n\n")
        # save_predictions(train_data[ind], train_preds, intents, "/home/dilyara/data/data_files/snips/preds_by_cnn/train_preds_" + str(ind) + ".csv")
        # save_predictions(test_data[ind], test_preds, intents, "/home/dilyara/data/data_files/snips/preds_by_cnn/test_preds_" + str(ind) + ".csv")


    train_preds = np.asarray(train_preds)
    train_true = np.asarray(train_true)
    test_preds = np.asarray(test_preds)
    test_true = np.asarray(test_true)

    f1_scores = report(train_true, train_preds, test_true, test_preds, intents) #lengths - intents

    #TURN ON IF WANT TO FIND THE BEST PARAMETERS
    dependency_f1_on_params['coer_reg_cnn'].append(network_params['coef_reg_cnn'])
    dependency_f1_on_params['coer_feg_den'].append(network_params['coef_reg_den'])
    dependency_f1_on_params['filters_cnn'].append(network_params['filters_cnn'])
    dependency_f1_on_params['dense_size'].append(network_params['dense_size'])
    dependency_f1_on_params['dropout_rate'].append(network_params['dropout_rate'])
    dependency_f1_on_params['batch_size'].append(learning_params['batch_size'])
    dependency_f1_on_params['lear_rate'].append(learning_params['lear_rate'])
    dependency_f1_on_params['lear_rate_decay'].append(learning_params['lear_rate_decay'])
    dependency_f1_on_params['epochs'].append(learning_params['epochs'])
    dependency_f1_on_params['mean_f1_score'].append(np.mean(f1_scores))

    if np.mean(f1_scores) > best_mean_f1:
        best_network_params = network_params
        best_learning_params = learning_params
        save(models[0], fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_0')
        save(models[1], fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_1')
        save(models[2], fname='/home/dilyara/data/models/intent_models/snips_models_softmax/best_model_2')
        print('BETTER PARAMETERS FOUND!\n')
        print('PARAMETERS:', best_network_params, best_learning_params)
        best_mean_f1 = np.mean(f1_scores)

    depend = pd.DataFrame(dependency_f1_on_params, index=np.arange(len(dependency_f1_on_params['epochs'])))
    depend.to_csv("/home/dilyara/data/data_files/snips/Dependency_f1_on_params_0.csv", index=False)
    # TURN ON IF WANT AVERAGE FOR PARTICULAR PARAMETERS

#     f1_scores_for_intents.append(f1_scores)
#
# f1_scores_for_intents = np.asarray(f1_scores_for_intents)
# for intent_id in range(len(intents)):
#     f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
#     f1_std = np.std(f1_scores_for_intents[:,intent_id])
#     print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))
