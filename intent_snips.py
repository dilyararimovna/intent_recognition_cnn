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
sys.path.append('/home/dilyara.baymurzina')
from random_search_class import param_gen
from save_load_model import init_from_scratch, init_from_saved, save

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

data = pd.read_csv("./intent_data/all_intent_data.csv")
print(data.head())
#data = data.iloc[np.random.permutation(np.arange(data.shape[0])), :]

fasttext_model_file = '../data_preprocessing/reddit_fasttext_model.bin'
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

stratif_y = [np.nonzero(data.loc[j, intents].values)[0][0] for j in range(data.shape[0])]

kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
kf_split.get_n_splits(data['request'], stratif_y)

best_mean_f1 = 0.
best_network_params = dict()
best_learning_params = dict()

f1_scores_for_intents = []

for k in range(16):
    network_params = param_gen(coef_reg_cnn={'range': [0.000571,0.000571], 'scale': 'log'},
                               coef_reg_den={'range': [0.000377,0.000377], 'scale': 'log'},
                               filters_cnn={'range': [220,220], 'discrete': True},
                               dense_size={'range': [81,81], 'discrete': True},
                               dropout_rate={'range': [0.42,0.42]})

    learning_params = param_gen(batch_size={'range': [45,45], 'discrete': True},
                                lear_rate={'range': [0.024380,0.024380], 'scale': 'log'},
                                lear_rate_decay={'range': [0.063984,0.063984], 'scale': 'log'},
                                epochs={'range': [29,29], 'discrete': True, 'scale': 'log'})

    print('\n\nCONSIDERED PARAMETERS: ', network_params)
    print('\n\nCONSIDERED PARAMETERS: ', learning_params)
    train_preds = []
    train_true = []
    test_preds = []
    test_true = []
    ind = 0

    for train_index, test_index in kf_split.split(data['request'], stratif_y):
        #print("-----TRAIN-----", train_index[:10], "\n-----TEST-----", test_index[:10])
        #print("-----TRAIN-----", len(train_index), "\n-----TEST-----", len(test_index))
        X_train, X_test = data.loc[train_index, 'request'], data.loc[test_index, 'request']
        y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

        X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
        X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

        model = init_from_scratch(cnn_word_model, text_size=text_size, 
                                  embedding_size=embedding_size, 
                                  kernel_sizes=kernel_sizes, **network_params)

        #model = init_from_saved(cnn_char_model, fname='cnn_model_sber', text_size=text_size, 
        #                        embedding_size=embedding_size, 
        #                        kernel_sizes=kernel_sizes, **network_params)


        optimizer = Adam(lr=learning_params['lear_rate'], decay=learning_params['lear_rate_decay'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy',
                               fmeasure])
        history = model.fit(X_train_embed, y_train.reshape(-1, 7),
                            batch_size=learning_params['batch_size'],
                            epochs=learning_params['epochs'],
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0)])
        ind += 1
        y_train_pred = model.predict(X_train_embed).reshape(-1, 7)
        train_preds.extend(y_train_pred)
        train_true.extend(y_train)

        y_test_pred = model.predict(X_test_embed).reshape(-1, 7)
        test_preds.extend(y_test_pred)
        test_true.extend(y_test)
        if ind == 3:
            break

    train_preds = np.asarray(train_preds)
    train_true = np.asarray(train_true)
    test_preds = np.asarray(test_preds)
    test_true = np.asarray(test_true)

    f1_scores = report(train_true, train_preds, test_true, test_preds, intents) #lengths - intents
    # if np.mean(f1_scores) > best_mean_f1:
    #     best_network_params = network_params
    #     best_learning_params = learning_params
    #     save(model, fname='./snips_models/best_model')
    #     print('BETTER PARAMETERS FOUND!\n')
    #     print('PARAMETERS:', best_network_params, best_learning_params)
    #     best_mean_f1 = np.mean(f1_scores)
    f1_scores_for_intents.append(f1_scores)

f1_scores_for_intents = np.asarray(f1_scores_for_intents)
for intent_id in range(len(intents)):
    f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
    f1_std = np.std(f1_scores_for_intents[:,intent_id])
    print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))
