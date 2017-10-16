import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def cnn_word_model(text_size, embedding_size, filters_cnn, 
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
    output = Dense(7, activation=None, kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation('sigmoid')(output)
    model = Model(inputs=inp, outputs=act_output)
    return model

