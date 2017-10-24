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

train_data = []

train_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_train_0"))
train_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_train_1"))
train_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_train_2"))

test_data = []

test_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_test_0"))
test_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_test_1"))
test_data.append(pd.read_csv("./intent_data/snips_ner_gold/snips_ner_gold_0/snips_test_2"))

fasttext_model_file = '../data_preprocessing/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

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
tag2id = {'O': 0, 'album': 38, 'artist': 25, 'best_rating': 31, 'city': 32, 'condition_description': 22,
           'condition_temperature': 30, 'country': 27, 'cuisine': 18, 'current_location': 16, 'entity_name': 20,
           'facility': 9, 'genre': 24, 'geographic_poi': 7, 'location_name': 33, 'movie_name': 21,
           'movie_type': 37, 'music_item': 29, 'object_location_type': 19, 'object_name': 13,
           'object_part_of_series_type': 17, 'object_select': 6, 'object_type': 10, 'party_size_description': 26,
           'party_size_number': 4, 'playlist': 14, 'playlist_owner': 1, 'poi': 36, 'rating_unit': 2,
           'rating_value': 35, 'restaurant_name': 28, 'restaurant_type': 8, 'served_dish': 15,
           'service': 34, 'sort': 3, 'spatial_relation': 11, 'state': 23, 'timeRange': 5, 'track': 39, 'year': 12}

id2tag = dict()
for tag_ind in tag2id:
    id2tag[tag2id[tag_ind]] = tag_ind


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
tags_acc_table = pd.DataFrame()
tags_acc_table['tag'] = [id2tag[i] for i in range(num_of_tags)] 
# table structure:
# tag | intent_0_mean | intent_0_std | intent_1_mean | intent_1_std | ... | 

for curr_tag in range(num_of_tags):
    print("\n\n-------------------CURRENT TAG: %s-----------------------------\n\n" % id2tag[curr_tag])

    f1_scores_for_intents = []

    for iter_ in range(16):
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
        for ind in range(3):
            #print("-----TRAIN-----", train_data[ind].head(), "\n-----TEST-----", test_data[ind].head())
            #print("-----TRAIN-----", train_data[ind].shape[0], "\n-----TEST-----", test_data[ind].shape[0])
            X_train, X_test = train_data[ind].loc[:,'request'].values, test_data[ind].loc[:,'request'].values
            y_train, y_test = train_data[ind].loc[:,intents].values, test_data[ind].loc[:,intents].values
            ner_train, ner_test = train_data[ind].loc[:,'ner_tag'].values, test_data[ind].loc[:,'ner_tag'].values

            X_train_embed = text2embeddings(X_train, fasttext_model, text_size, embedding_size)
            X_test_embed = text2embeddings(X_test, fasttext_model, text_size, embedding_size)
            
            train_tags_table = []
            for k in range(train_data[ind].shape[0]):
                tags = [int(tag) for tag in ner_train[k].split(' ')]
                request_tags = []
                for i_word, tag in enumerate(tags):
                    #request_tags.append([(1 * (tag == m)) for m in range(num_of_tags)])
                    request_tags.append([(1 * (tag == curr_tag)) for m in range(num_of_tags)])
                train_tags_table.append(request_tags)

            train_tags_table = keras.preprocessing.sequence.pad_sequences(train_tags_table, maxlen=text_size, 
                                                                          padding='pre')

            X_train_embed = np.dstack((X_train_embed, train_tags_table))

            test_tags_table = []
            for k in range(test_data[ind].shape[0]):
                tags = [int(tag) for tag in ner_test[k].split(' ')]
                request_tags = []
                for i_word, tag in enumerate(tags):
                    #request_tags.append([(1 * (tag == m)) for m in range(num_of_tags)])
                    request_tags.append([(1 * (tag == curr_tag)) for m in range(num_of_tags)])
                test_tags_table.append(request_tags)

            test_tags_table = keras.preprocessing.sequence.pad_sequences(test_tags_table, maxlen=text_size, 
                                                                         padding='pre')

            X_test_embed = np.dstack((X_test_embed, test_tags_table))
            model = init_from_scratch(cnn_word_model, text_size=text_size, 
                                  embedding_size=embedding_size+num_of_tags, 
                                  kernel_sizes=kernel_sizes, **network_params)
            #model = init_from_saved(cnn_word_model, fname='cnn_model_sber', text_size=text_size, 
            #                        embedding_size=embedding_size+num_of_tags, 
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

        f1_scores = report(train_true, train_preds, test_true, test_preds, intents)
        f1_scores_for_intents.append(f1_scores) # for particular tag!

    f1_scores_for_intents = np.asarray(f1_scores_for_intents)
    for intent_id in range(len(intents)):
        f1_mean = np.mean(f1_scores_for_intents[:,intent_id])
        f1_std = np.std(f1_scores_for_intents[:,intent_id])
        print("Intent: %s \t F1: %f +- %f" % (intents[intent_id], f1_mean, f1_std))
        tags_acc_table.loc[curr_tag,intent+"_mean"] = f1_mean
        tags_acc_table.loc[curr_tag,intent+"_std"] = f1_std


tags_acc_table.to_csv('./tags_intent_f1scores.csv', index=False)