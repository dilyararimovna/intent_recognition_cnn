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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_fscore_support
import fasttext
import keras
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam, SGD
from keras.layers.core import Dropout, Reshape, Flatten, Permute, Lambda, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers.merge import Multiply

import json
from metrics import fmeasure
from keras import backend as K

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

data = pd.read_csv("./intent_data/all_intent_data.csv")
print(data.head())
data = data.iloc[np.random.permutation(np.arange(data.shape[0])), :]


def text2embeddings(texts, fasttext_model, text_size, embedding_size):
    all_embeddings = []
    for text in texts:
        embeddings = []  # embedded text
        tokens = text.split(' ')
        if len(tokens) > text_size:
            tokens = tokens[-text_size:]
        for token in tokens:
            embeddings.append(fasttext_model[token])
        if len(tokens) < text_size:
            pads = [np.zeros(embedding_size) for _ in range(text_size - len(tokens))]
            embeddings = pads + embeddings
        embeddings = np.asarray(embeddings)
        all_embeddings.append(embeddings)

    all_embeddings = np.asarray(all_embeddings)
    return all_embeddings


def cnn_word_model(text_size, embedding_size, filters_cnn, kernel_sizes):
    inp = Input(shape=(text_size, embedding_size))

    outputs = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(0.001), padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    print('Concatenate shape:', output.shape)

    output = Dropout(rate=0.5)(output)
    output = Dense(100, activation=None,
                   kernel_regularizer=l2(0.01))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=0.5)(output)
    output = Dense(7, activation=None, kernel_regularizer=l2(0.01))(output)
    output = BatchNormalization()(output)
    act_output = Activation('sigmoid')(output)
    model = Model(inputs=inp, outputs=act_output)
    return model


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, text_size))(a)
    a = Dense(text_size, activation='relu')(a)
    a = Dense(text_size, activation='relu')(a)
    a = Dense(text_size, activation='relu')(a)
    a = Dense(text_size, activation='softmax')(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def lstm_word_model(text_size, embedding_size, units_lstm):
    embed_input = Input(shape=(text_size, embedding_size))
    output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                kernel_regularizer=l2(0.0001),
                                return_sequences=True,
                                dropout=0.5))(embed_input)
    output = attention_3d_block(output)
    output = Flatten()(output)
    output = Dropout(rate=0.5)(output)
    output = Dense(100, activation=None,
                   kernel_regularizer=l2(0.001))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=0.5)(output)
    output = Dense(7, activation=None,
                   kernel_regularizer=l2(0.001))(output)
    output = BatchNormalization()(output)
    act_output = Activation('sigmoid')(output)
    model = Model(inputs=embed_input, outputs=act_output)
    return model


fasttext_model_file = '../data_preprocessing/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

text_size = 25
embedding_size = 100

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']
stratif_y = [np.nonzero(data.loc[j, intents].values)[0][0] for j in range(data.shape[0])]
# print(np.unique(stratif_y))
n_splits = 5
kf_split = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
kf_split.get_n_splits(data['request'], stratif_y)

train_preds = []
train_true = []
test_preds = []
test_true = []
filters = [128, 128, 128, 128, 128, 128, 128, 128]
ind = 0

for train_index, test_index in kf_split.split(data['request'], stratif_y):
    # print("TRAIN:", train_index[:10], "\nTEST:", test_index[:10])
    # print("TRAIN:", len(train_index), "\nTEST:", len(test_index))
    X_train, X_test = data.loc[train_index, 'request'].values, data.loc[test_index, 'request'].values
    y_train, y_test = data.loc[train_index, intents].values, data.loc[test_index, intents].values

    for j, part in enumerate([10, 25, 50, 100, 200, 500, 1000]):
        print("______Number of samples=%d___________" % part)

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

        model = cnn_word_model(text_size=text_size, embedding_size=100,
                               filters_cnn=filters[j], kernel_sizes=[1, 2, 3])
        # model = lstm_word_model(text_size=text_size, embedding_size=100,units_lstm=64)
        optimizer = Adam(lr=0.1, decay=0.1)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[  # 'categorical_accuracy',
                          fmeasure])
        history = model.fit(X_train_embed, y_train_part.reshape(-1, 7),
                            batch_size=32,
                            epochs=100,
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
        #        print('Train AUC-ROC:', roc_auc_score(train_true, train_preds))
        #        print('Test AUC-ROC:', roc_auc_score(test_true, test_preds))
        train_preds = np.round(train_preds)
        test_preds = np.round(test_preds)

        print('TRAIN: (with repeats)')
        print("     type \t precision \t recall \t f1-score \t support")

        for i, intent in enumerate(intents):
            scores = np.asarray(precision_recall_fscore_support(train_true[:, i], train_preds[:, i]))[:, 1]
            print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))

        print('TEST:')
        print("     type \t precision \t recall \t f1-score \t support")
        f1_scores = []
        for i, intent in enumerate(intents):
            scores = np.asarray(precision_recall_fscore_support(test_true[:, i], test_preds[:, i]))[:, 1]
            print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))
            f1_scores.append(scores[2])

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        print('F1 SCORE ON TEST : %f +- %f ' % (f1_mean, f1_std))

    break

