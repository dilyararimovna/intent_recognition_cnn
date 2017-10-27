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

test_data = []

test_data.append(pd.read_csv("/home/dilyara/data/data_files/snips/snips_ner_gold/snips_ner_gold_0/snips_test_0"))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 5
filters = [64, 64, 64, 128, 128, 256, 256]
kernel_sizes = [1,2,3]
coef_reg_cnn = 0.0001
coef_reg_den = 0.001
dense_sizes = [100, 100, 100, 100, 100, 100, 100]
dropout_rate = 0.5
lear_rate = 0.1
lear_rate_decay = 0.1
batch_sizes = [8, 8, 8, 16, 32, 32, 32]
epochs = 1000
#train_sizes = [10, 25, 50, 100, 200, 500, 1000] # per intent
train_sizes = [500,1000]

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

X_train, X_test = train_data[0].loc[:,'request'].values, test_data[0].loc[:, 'request'].values
y_train, y_test = train_data[0].loc[:,intents].values, test_data[0].loc[:, intents].values

for train_size in train_sizes:
    print("\n\n______NUMBER OF TRAIN SAMPLES PER INTENT = %d___________" % train_size)
    train_index_part = []
    for i, intent in enumerate(intents):
        samples_intent = np.nonzero(y_train[:, i])[0]
        train_index_part.extend(list(np.random.choice(samples_intent, size=train_size)))

    best_mean_f1 = 0.
    best_network_params = dict()
    best_learning_params = dict()

    f1_scores_for_intents = []
    for p in range(10):
        network_params = param_gen(coef_reg_cnn={'range': [0.0001,0.001], 'scale': 'log'},
                                   coef_reg_den={'range': [0.0001,0.001], 'scale': 'log'},
                                   filters_cnn={'range': [200,300], 'discrete': True},
                                   dense_size={'range': [50,100], 'discrete': True},
                                   dropout_rate={'range': [0.4, 0.6]})

        learning_params = param_gen(batch_size={'range': [30, 60], 'discrete': True},
                                    lear_rate={'range': [0.01,0.1], 'scale': 'log'},
                                    lear_rate_decay={'range': [0.01,0.1], 'scale': 'log'},
                                    epochs={'range': [50,100], 'discrete': True, 'scale': 'log'})

        print('\n\nCONSIDERED PARAMETERS: ', network_params)
        print('\n\nCONSIDERED PARAMETERS: ', learning_params)

        models = []

        X_train_embed = text2embeddings(X_train[train_index_part], fasttext_model, text_size, embedding_size)
        X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

        model = init_from_scratch(cnn_word_model, text_size=text_size,
                      embedding_size=embedding_size,
                      kernel_sizes=kernel_sizes, **network_params)

        optimizer = Adam(lr=learning_params['lear_rate'], decay=learning_params['lear_rate_decay'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy',
                               fmeasure])
        history = model.fit(X_train_embed, y_train[train_index_part,:].reshape(-1,7),
                            batch_size=learning_params['batch_size'],
                            epochs=learning_params['epochs'],
                            validation_split=0.1,
                            verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0)])

        y_train_pred = model.predict(X_train_embed).reshape(-1, 7)
        y_test_pred = model.predict(X_test_embed).reshape(-1, 7)

        # print("Saving files for Mishanya NER\n\n")
        # save_predictions(train_data[ind], train_preds, intents, "/home/dilyara/data/data_files/snips/preds_by_cnn/train_preds_" + str(ind) + ".csv")
        # save_predictions(test_data[ind], test_preds, intents, "/home/dilyara/data/data_files/snips/preds_by_cnn/test_preds_" + str(ind) + ".csv")

        f1_scores = report(y_train[train_index_part,:].reshape(-1,7), y_train_pred, y_test, y_test_pred, intents)
        if np.mean(f1_scores) > best_mean_f1:
            best_network_params = network_params
            best_learning_params = learning_params
            save(model, fname='/home/dilyara/data/models/intent_models/snips_models/best_model_byparts_' + str(train_size) + '_0')
            print('BETTER PARAMETERS FOUND!\n')
            print('PARAMETERS:', best_network_params, best_learning_params)
            best_mean_f1 = np.mean(f1_scores)

