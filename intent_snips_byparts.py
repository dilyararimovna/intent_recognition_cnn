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
train_sizes = [10, 25, 50, 100, 200, 500, 1000] # per intent

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
#---------------------------------------

stratif_y = [np.nonzero(data.loc[j, intents].values)[0][0] for j in range(data.shape[0])]
# print(np.unique(stratif_y))

kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
kf_split.get_n_splits(data['request'], stratif_y)

train_preds = []
train_true = []
test_preds = []
test_true = []
ind = 0

for train_index, test_index in kf_split.split(data['request'], stratif_y):
    X_train, X_test = data.loc[train_index, 'request'].values, data.loc[test_index, 'request'].values
    y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

    for j, part in enumerate(train_sizes):
        print("\n\n______Number of samples=%d___________" % part)

        train_index_part = []
        for i, intent in enumerate(intents):
            samples_intent = np.nonzero(y_train[:, i])[0]
            print(samples_intent)
            train_index_part.extend(list(np.random.choice(samples_intent, size=part)))

        print('Size of train data %d' % len(train_index_part))
        X_train_part = X_train[train_index_part]
        y_train_part = y_train[train_index_part]

        X_train_embed = text2embeddings(X_train_part, fasttext_model, text_size, embedding_size)
        X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)

        model = cnn_word_model(text_size, embedding_size=embedding_size, filters_cnn=filters[j], 
                               kernel_sizes=kernel_sizes, coef_reg_cnn=coef_reg_cnn, coef_reg_den=coef_reg_den, 
                               dropout_rate=dropout_rate, dense_size=dense_sizes[j])

        optimizer = Adam(lr=lear_rate, decay=lear_rate_decay)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[  # 'categorical_accuracy',
                               fmeasure])
        history = model.fit(X_train_embed, y_train_part.reshape(-1, 7),
                            batch_size=batch_sizes[j],
                            epochs=epochs,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00),
                                       ModelCheckpoint(filepath="./keras_checkpoints_byparts/snips_" + str(n_splits)),
                                       TensorBoard(log_dir='./keras_log_files_byparts_' + str(ind))
                                       ])
        ind += 1
        y_train_pred = model.predict(X_train_embed).reshape(-1, 7)
        train_preds = y_train_pred
        train_true = y_train_part

        y_test_pred = model.predict(X_test_embed).reshape(-1, 7)
        test_preds = y_test_pred
        test_true = y_test

        train_preds = np.round(train_preds)
        test_preds = np.round(test_preds)

        f1_scores = report(train_true, train_preds, test_true, test_preds, intents)

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        print('F1 SCORE ON TEST : %f +- %f ' % (f1_mean, f1_std))

    break # doing only once

