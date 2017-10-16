import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
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

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

data = pd.read_csv("./intent_data/all_intent_data.csv")
print(data.head())
data = data.iloc[np.random.permutation(np.arange(data.shape[0])), :]

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

kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)
kf_split.get_n_splits(data['request'], stratif_y)

train_preds = []
train_true = []
test_preds = []
test_true = []
ind = 0

for train_index, test_index in kf_split.split(data['request'], stratif_y):
    print("-----TRAIN-----", train_index[:10], "\n-----TEST-----", test_index[:10])
    print("-----TRAIN-----", len(train_index), "\n-----TEST-----", len(test_index))
    X_train, X_test = data.loc[train_index, 'request'], data.loc[test_index, 'request']
    y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

    X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
    X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

    model = cnn_word_model(text_size, embedding_size=embedding_size, filters_cnn=filters_cnn, 
                           kernel_sizes=kernel_sizes, coef_reg_cnn=coef_reg_cnn, coef_reg_den=coef_reg_den, 
                           dropout_rate=dropout_rate, dense_size=dense_size)

    optimizer = Adam(lr=lear_rate, decay=lear_rate_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[# 'categorical_accuracy',
                           fmeasure])
    history = model.fit(X_train_embed, y_train.reshape(-1, 7),
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        verbose=0,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0),
                                   ModelCheckpoint(filepath="./keras_checkpoints/snips_" + str(n_splits)),
                                   TensorBoard(log_dir='./keras_logs/keras_log_files_' + str(ind))
                                   ])
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

report(train_true, train_preds, test_true, test_preds, intents)


