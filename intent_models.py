import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import tensorflow as tf
SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

def cnn_word_model(text_size, n_classes, embedding_size, filters_cnn,
                   kernel_sizes, coef_reg_cnn, coef_reg_den, dropout_rate, dense_size):

    inp = Input(shape=(text_size, embedding_size))

    outputs = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(coef_reg_cnn), padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    print('Concatenate shape:', output.shape)

    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation=None, kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation('softmax')(output)
    model = Model(inputs=inp, outputs=act_output)
    return model

def cnn_word_model_ner(text_size, n_classes, tag_size, embedding_size, filters_cnn_emb, filters_cnn_tag,
                       kernel_sizes, coef_reg_cnn_emb, coef_reg_cnn_tag, coef_reg_den, 
                       dropout_rate, dense_size):

    inp_emb = Input(shape=(text_size, embedding_size))
    inp_tag = Input(shape=(text_size, tag_size))

    outputs_emb = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn_emb, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(coef_reg_cnn_emb), padding='same')(inp_emb)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs_emb.append(output_i)

    outputs_tag = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn_tag, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(coef_reg_cnn_tag), padding='same')(inp_tag)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs_tag.append(output_i)

    output_emb = concatenate(outputs_emb, axis=1)
    print('Concatenate emb shape:', output_emb.shape)
    output_tag = concatenate(outputs_tag, axis=1)
    print('Concatenate tag shape:', output_tag.shape)
    output = concatenate([output_emb, output_tag], axis=1)
    print('Concatenate shape:', output.shape)
    
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation=None, kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation('softmax')(output)
    model = Model(inputs=[inp_emb, inp_tag], outputs=act_output)
    return model


def cnn_word_model_with_sent_emb(text_size, n_classes, embedding_size, sent_embedding_size, filters_cnn,
	                             kernel_sizes, coef_reg_cnn, coef_reg_den, dropout_rate, dense_size):
    inp = Input(shape=(text_size, embedding_size))
    sent_emb = Input(shape=(sent_embedding_size,))

    #sent_emb_resh = Reshape(target_shape=(1,sent_embedding_size))(sent_emb)

    outputs = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(coef_reg_cnn), padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    print('Concatenate shape:', output.shape)
    output = concatenate([output, sent_emb], axis=1)
    print('Concatenate with sent emb shape:', output.shape)

    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation=None, kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation('sigmoid')(output)
    model = Model(inputs=[inp, sent_emb], outputs=act_output)
    return model

def cnn_word_model_glove(text_size, n_classes, embedding_size, filters_cnn,
                         kernel_sizes, coef_reg_cnn, coef_reg_den, dropout_rate, dense_size):

    inp = Input(shape=(text_size, embedding_size))

    outputs = []
    for i in range(len(kernel_sizes)):
        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes[i], activation=None,
                          kernel_regularizer=l2(coef_reg_cnn), padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    print('Concatenate shape:', output.shape)

    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation=None, kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation('sigmoid')(output)
    model = Model(inputs=inp, outputs=act_output)
    return model
