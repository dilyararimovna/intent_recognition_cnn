from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
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

from metrics import fmeasure
from fasttext_embeddings import text2embeddings
from intent_models import cnn_word_model
from report_intent import report

SEED = 23
np.random.seed(SEED)
tf.set_random_seed(SEED)

train_data = []

train_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_train_0.csv"))
train_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_train_1.csv"))
train_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_train_2.csv"))

test_data = []

test_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_test_0.csv"))
test_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_test_1.csv"))
test_data.append(pd.read_csv("./intent_data/snips_ner_2017_ground_truth/snips_test_2.csv"))

filename = '/home/dilyara.baymurzina/data_preprocessing/glove.6B/glove.6B.50d.txt'

def loadGloVe(filename):
    vocab = []
    embd = []
    file = io.open(filename,'r' , encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd


vocabulary, embedding_ = loadGloVe(filename)
vocabulary_size = len(vocabulary)
embedding_size = len(embedding_[0])
embedding = np.asarray(embedding_)

print ('Vocabulary size:', vocabulary_size)
print ('Embedding size: ', embedding_size)
#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 5
filters_cnn = 256
kernel_sizes = [1,2,3]
coef_reg_cnn = 0.001
coef_reg_den = 0.001
dense_size = 100
dropout_rate = 0.5
lear_rate = 0.1
lear_rate_decay = 0.1
batch_size = 64
epochs = 500
num_of_tags = 40

intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']

#---------------------------------------
# tag2id = {'O': 0, 'album': 38, 'artist': 25, 'best_rating': 31, 'city': 32, 'condition_description': 22,
#           'condition_temperature': 30, 'country': 27, 'cuisine': 18, 'current_location': 16, 'entity_name': 20,
#           'facility': 9, 'genre': 24, 'geographic_poi': 7, 'location_name': 33, 'movie_name': 21, 'movie_type': 37,
#           'music_item': 29, 'object_location_type': 19, 'object_name': 13, 'object_part_of_series_type': 17,
#           'object_select': 6, 'object_type': 10, 'party_size_description': 26, 'party_size_number': 4, 'playlist': 14,
#           'playlist_owner': 1, 'poi': 36, 'rating_unit': 2, 'rating_value': 35, 'restaurant_name': 28, 'restaurant_type': 8,
#           'served_dish': 15, 'service': 34, 'sort': 3, 'spatial_relation': 11, 'state': 23, 'timeRange': 5, 'track': 39, 'year': 12}

# id2tag = dict()
# for tag_ind in tag2id:
#     id2tag[tag2id[tag_ind]] = tag_ind


# for intent in intents:
#     print('\n---------------Intent: %s-----------------\n' % intent)
#     data_tags = data.loc[data[intent] == 1,'ner_tag'].values
#     tags_for_intent = []
#     for request_tags in data_tags:
#         request_int_tags = [int(tag) for tag in request_tags.split(' ')]
#         tags_for_intent.extend(request_int_tags)
#         tags_for_intent = list(set(tags_for_intent))
#     print('Tags for that intent:', tags_for_intent)

#__________________________________________________________________________________

train_preds = []
train_true = []
test_preds = []
test_true = []

for ind in range(3):
    print("-----TRAIN-----", train_data[ind].head(), "\n-----TEST-----", test_data[ind].head())
    print("-----TRAIN-----", train_data[ind].shape[0], "\n-----TEST-----", test_data[ind].shape[0])
    X_train, X_test = train_data[ind].loc[:,'request'].values, test_data[ind].loc[:,'request'].values
    y_train, y_test = train_data[ind].loc[:,intents].values, test_data[ind].loc[:,intents].values
    ner_train, ner_test = train_data[ind].loc[:,'ner_tag'].values, test_data[ind].loc[:,'ner_tag'].values

    # X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
    # X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)
    X_train_embed = X_train
    X_test_embed = X_test

    train_tags_table = []
    for k in range(train_data[ind].shape[0]):
        tags = [int(tag) for tag in ner_train[k].split(' ')]
        request_tags = []
        for i_word, tag in enumerate(tags):
            request_tags.append([(1 * (tag == m)) for m in range(num_of_tags)])
        train_tags_table.append(request_tags)

    train_tags_table = keras.preprocessing.sequence.pad_sequences(train_tags_table, maxlen=text_size, 
                                                                  padding='pre')

    X_train_embed = np.dstack((X_train_embed, train_tags_table))

    test_tags_table = []
    for k in range(test_data[ind].shape[0]):
        tags = [int(tag) for tag in ner_test[k].split(' ')]
        request_tags = []
        for i_word, tag in enumerate(tags):
            request_tags.append([(1 * (tag == m)) for m in range(num_of_tags)])
        test_tags_table.append(request_tags)

    test_tags_table = keras.preprocessing.sequence.pad_sequences(test_tags_table, maxlen=text_size, 
                                                                 padding='pre')

    X_test_embed = np.dstack((X_test_embed, test_tags_table))

    model = cnn_word_model(text_size, embedding_size=embedding_size+num_of_tags, filters_cnn=filters_cnn, 
                           kernel_sizes=kernel_sizes, coef_reg_cnn=coef_reg_cnn, coef_reg_den=coef_reg_den, 
                           dropout_rate=dropout_rate, dense_size=dense_size)

    optimizer = Adam(lr=lear_rate, decay=lear_rate_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy',
                           fmeasure])
    history = model.fit(X_train_embed, y_train.reshape(-1, 7),
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0),
                                   #ModelCheckpoint(filepath="./keras_checkpoints/snips_" + str(n_splits)),
                                   #TensorBoard(log_dir='./keras_logs/keras_log_files_' + str(ind))
                                   ])

    y_train_pred = model.predict(X_train_embed).reshape(-1, 7)
    train_preds.extend(y_train_pred)
    train_true.extend(y_train)

    y_test_pred = model.predict(X_test_embed).reshape(-1, 7)
    test_preds.extend(y_test_pred)
    test_true.extend(y_test)


train_preds = np.asarray(train_preds)
train_true = np.asarray(train_true)
test_preds = np.asarray(test_preds)
test_true = np.asarray(test_true)

report(train_true, train_preds, test_true, test_preds, intents)


