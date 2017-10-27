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

from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import fasttext

from metrics import fmeasure
from fasttext_embeddings import text2embeddings
from intent_models import cnn_word_model_with_sent_emb
from report_intent import report
import cv2,torch
import numpy as np
import nltk

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from random_search_class import param_gen
from save_load_model import save, init_from_scratch, init_from_saved

sys.path.append('/home/dilyara/Documents/paraphraser')
from get_layer import Paraphraser


SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

data = pd.read_csv("/home/dilyara/Documents/GitHub/intent_recognition_cnn/intent_data/snips_intent_ner.csv")
print(data.head())
data = data.iloc[np.random.permutation(np.arange(data.shape[0])), :]


fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 5
filters_cnn = 256
kernel_sizes = [1,2,3]
coef_reg_cnn = 0.001
coef_reg_den = 0.01
dense_size = 100
dropout_rate = 0.5
lear_rate = 0.1
lear_rate_decay = 0.1
batch_size = 64
epochs = 500

sent_embedding_size = 300

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#-----------------------------------------------------------------

stratif_y = [np.nonzero(data.loc[j, intents].values)[0][0] for j in range(data.shape[0])]

kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
kf_split.get_n_splits(data['request'], stratif_y)

best_mean_f1 = 0.
best_network_params = dict()
best_learning_params = dict()

quora_emb = Paraphraser(data_path='/home/dilyara/Documents/paraphraser',
                          fasttext_model='moved_ft_0.8.3_nltk_yalen_sg_300.bin',
                          model_files='paraphraser')

print('Sentence embedding class INIT\n')


while 1:
    network_params = param_gen(coef_reg_cnn={'range': [1e-4,1e-2], 'scale': 'log'},
                               coef_reg_den={'range': [1e-4,1e-2], 'scale': 'log'},
                               filters_cnn={'range': [50, 300], 'discrete': True},
                               dense_size={'range': [50, 200], 'discrete': True},
                               dropout_rate={'range': [0.4, 0.6]})

    learning_params = param_gen(batch_size={'range': [2, 64], 'discrete': True},
                                lear_rate={'range': [1e-2, 1.], 'scale': 'log'},
                                lear_rate_decay={'range': [1e-3, 1e-1], 'scale': 'log'},
                                epochs={'range': [5, 50], 'discrete': True, 'scale': 'log'})
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
        X_train, X_test = data.loc[train_index, 'request'].values, data.loc[test_index, 'request'].values
        y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

        X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
        X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

        X_train_sent_emb = quora_emb.get_vectors(X_train)
        X_test_sent_emb = quora_emb.get_vectors(X_test)
        print("Sentence embeddings size: %d " % (X_train_embed.shape[1]))

        model = init_from_scratch(cnn_word_model_with_sent_emb, embedding_size=embedding_size,
                                             sent_embedding_size=sent_embedding_size, kernel_sizes=kernel_sizes,
                                             **network_params)
        optimizer = Adam(lr=learning_params['lear_rate'], decay=learning_params['lear_rate_decay'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[# 'categorical_accuracy',
                               fmeasure])
        history = model.fit([X_train_embed, X_train_sent_emb], y_train.reshape(-1, 7),
                            batch_size=learning_params['batch_size'],
                            epochs=learning_params['epochs'],
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0),
                                       #ModelCheckpoint(filepath="./keras_checkpoints/snips_" + str(ind)),
                                       #TensorBoard(log_dir='./keras_logs/keras_log_files_' + str(ind))
                                       ])
        ind += 1
        y_train_pred = model.predict([X_train_embed, X_train_sent_emb]).reshape(-1, 7)
        y_test_pred = model.predict([X_test_embed, X_test_sent_emb]).reshape(-1, 7)

        train_preds.extend(y_train_pred)
        train_true.extend(y_train)

        test_preds.extend(y_test_pred)
        test_true.extend(y_test)
        if ind == 3:
            break

    train_preds = np.asarray(train_preds)
    train_true = np.asarray(train_true)
    test_preds = np.asarray(test_preds)
    test_true = np.asarray(test_true)

    f1_scores = report(train_true, train_preds, test_true, test_preds, intents)
    if np.mean(f1_scores) > best_mean_f1:
        best_network_params = network_params
        best_learning_params = learning_params
        save(model, fname='./snips_quora_sent_emb_best_model')
        print('BETTER PARAMETERS FOUND!\n')
        print('PARAMETERS:', best_network_params, best_learning_params)
        best_mean_f1 = np.mean(f1_scores)

