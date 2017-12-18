import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
import fasttext
from metrics import fmeasure

import sys
sys.path.append('/home/dilyara/Documents/GitHub/general_scripts')
from intent_models import cnn_word_model
from intent_recognizer_class import IntentRecognizer
import os

SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

path = '/home/dilyara/data/data_files/snips'

train_data = []
train_data.append(pd.read_csv(os.path.join(path, 'paraphrases/intent_train_data_with_paraphrases.csv')))

test_data = []
test_data.append(pd.read_csv(os.path.join(path, 'paraphrases/intent_test_data_with_paraphrases.csv')))

fasttext_model_file = '/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin'
fasttext_model = fasttext.load_model(fasttext_model_file)

#-------------PARAMETERS----------------
text_size = 25
embedding_size = 100
n_splits = 1
kernel_sizes = [1, 2, 3]
# train_sizes = [500, 1000] # per intent
# train_sizes = [10, 25, 50, 100, 200]
train_sizes = [25]
intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather',
           'PlayMusic', 'RateBook', 'SearchCreativeWork',
           'SearchScreeningEvent']

# -----------------------------PARAMETERS--------------------------------------
nobpe_params = [
    # 10
    [{'coef_reg_cnn': 0.00023374473715304027,
      'coef_reg_den': 0.0001515911756700429,
      'filters_cnn': 164,
      'dense_size': 75,
      'dropout_rate': 0.4136499410708183},
     {'batch_size': 26,
      'lear_rate': 0.019439887028475728,
      'lear_rate_decay': 0.022681549302200486,
      'epochs': 82}],
    # 25
    [{'coef_reg_cnn': 0.0001727114892262775,
      'coef_reg_den': 0.0002206749436046956,
      'filters_cnn': 178,
      'dense_size': 70,
      'dropout_rate': 0.5719290690705213},
     {'batch_size': 59,
      'lear_rate': 0.027546794873469779,
      'lear_rate_decay': 0.030398438427729761,
      'epochs': 58}],
    # 50
    [{'coef_reg_cnn': 0.0019754865502769258,
      'coef_reg_den': 0.0026554858066638177,
      'filters_cnn': 167,
      'dense_size': 197,
      'dropout_rate': 0.46320094457327876},
     {'batch_size': 32,
      'lear_rate': 0.018224951554943532,
      'lear_rate_decay': 0.071342199989434155,
      'epochs': 91}],
    # 100
    [{'coef_reg_cnn': 0.00013840101560216297,
      'coef_reg_den': 0.00066864868323891779,
      'filters_cnn': 108,
      'dense_size': 163,
      'dropout_rate': 0.46100534877757077},
     {'batch_size': 40,
      'lear_rate': 0.037959864853053367,
      'lear_rate_decay': 0.073924000819309235,
      'epochs': 87}],
    # 200
    [{'coef_reg_cnn': 0.00012624978090688168,
      'coef_reg_den': 0.00017141355945092038,
      'filters_cnn': 197,
      'dense_size': 113,
      'dropout_rate': 0.4958638957918347},
     {'batch_size': 17,
      'lear_rate': 0.037735283746489802,
      'lear_rate_decay': 0.021084734592690852,
      'epochs': 57}],
    # 500
    [{'coef_reg_cnn': 0.00028700893261581471,
      'coef_reg_den': 0.00013723898248435492,
      'filters_cnn': 196,
      'dense_size': 81,
      'dropout_rate': 0.430525695641187},
     {'batch_size': 31,
      'lear_rate': 0.046812381358125064,
      'lear_rate_decay': 0.039643826749619838,
      'epochs': 60}],
    # 1000
    [{'coef_reg_cnn': 0.0071295722198561817,
      'coef_reg_den': 0.00039016164266169657,
      'filters_cnn': 74,
      'dense_size': 176,
      'dropout_rate': 0.5714650163714523},
     {'batch_size': 55,
      'lear_rate': 0.013671621598471883,
      'lear_rate_decay': 0.031964283578299571,
      'epochs': 61}]]

bpe_params = [
    # 10
    [{'coef_reg_cnn': 0.00023374473715304027,
      'coef_reg_den': 0.0001515911756700429,
      'filters_cnn': 164,
      'dense_size': 75,
      'dropout_rate': 0.4136499410708183},
     {'batch_size': 26,
      'lear_rate': 0.019439887028475728,
      'lear_rate_decay': 0.022681549302200486,
      'epochs': 82}],
    # 25
    [{'coef_reg_cnn': 0.0001727114892262775,
      'coef_reg_den': 0.0002206749436046956,
      'filters_cnn': 178,
      'dense_size': 70,
      'dropout_rate': 0.5719290690705213},
     {'batch_size': 59,
      'lear_rate': 0.027546794873469779,
      'lear_rate_decay': 0.030398438427729761,
      'epochs': 58}],
    # 50
    [{'coef_reg_cnn': 0.0001575817566046776,
      'coef_reg_den': 0.00013777921143776498,
      'filters_cnn': 82,
      'dense_size': 149,
      'dropout_rate': 0.5528800222778611},
     {'batch_size': 61,
      'lear_rate': 0.034865344780833266,
      'lear_rate_decay': 0.016680428655624448,
      'epochs': 55}],
    # 100
    [{'coef_reg_cnn': 0.00086018225135340339,
      'coef_reg_den': 0.00056675390165778373,
      'filters_cnn': 83,
      'dense_size': 91,
      'dropout_rate': 0.4910735408930962},
     {'batch_size': 32,
      'lear_rate': 0.043120681334714202,
      'lear_rate_decay': 0.090639680498334457,
      'epochs': 79}],
    # 200
    [{'coef_reg_cnn': 0.0057884273666188608,
      'coef_reg_den': 0.00064855237869837436,
      'filters_cnn': 167,
      'dense_size': 98,
      'dropout_rate': 0.5650273136217192},
     {'batch_size': 17,
      'lear_rate': 0.088488228735083843,
      'lear_rate_decay': 0.094810650343581299,
      'epochs': 58}],
    # 500
    [{'coef_reg_cnn': 0.00013691180821036192,
      'coef_reg_den': 0.00021350889614848756,
      'filters_cnn': 180,
      'dense_size': 131,
      'dropout_rate': 0.4497705652336749},
     {'batch_size': 39,
      'lear_rate': 0.032290366863280281,
      'lear_rate_decay': 0.026813191023760292,
      'epochs': 53}],
    # 1000
    [{'coef_reg_cnn': 0.00020307712345035391,
      'coef_reg_den': 0.0013979256142396732,
      'filters_cnn': 94,
      'dense_size': 73,
      'dropout_rate': 0.5985072605231265},
     {'batch_size': 63,
      'lear_rate': 0.012186531560033385,
      'lear_rate_decay': 0.018144339691863486,
      'epochs': 76}]]

train_requests = [train_data[i].loc[:, 'request'].values for i in range(n_splits)]
train_bpe_paraphrases =[train_data[i].loc[:, 'bpe_paraphrases'].values for i in range(n_splits)]
train_nobpe_paraphrases =[train_data[i].loc[:, 'nobpe_paraphrases'].values for i in range(n_splits)]
train_classes = [train_data[i].loc[:, intents].values for i in range(n_splits)]

test_requests = [test_data[i].loc[:, 'request'].values for i in range(n_splits)]
test_bpe_paraphrases = [test_data[i].loc[:, 'bpe_paraphrases'].values for i in range(n_splits)]
test_nobpe_paraphrases = [test_data[i].loc[:, 'nobpe_paraphrases'].values for i in range(n_splits)]
test_classes = [test_data[i].loc[:, intents].values for i in range(n_splits)]

# ------------------------ which paraphrases to use-------------------
train_paraphrases = train_nobpe_paraphrases
test_paraphrases = test_nobpe_paraphrases
params = nobpe_params

f1_mean_per_size = []
f1_std_per_size = []

for n_size, train_size in enumerate(train_sizes):
    print("\n\n______NUMBER OF TRAIN SAMPLES PER INTENT = %d___________" % train_size)
    train_index_parts = []
    for model_ind in range(n_splits):
        train_part = []
        for i, intent in enumerate(intents):
            samples_intent = np.nonzero(train_classes[model_ind][:, i])[0]
            train_part.extend(list(np.random.choice(samples_intent, size=train_size)))
        train_index_parts.append(train_part)

    train_requests_part = []
    train_classes_part = []
    for model_ind in range(n_splits):
        requests = np.hstack((train_requests[model_ind][train_index_parts[model_ind]],
                              train_paraphrases[model_ind][train_index_parts[model_ind]]))
        train_requests_part.append(requests)
        classes = np.vstack((train_classes[model_ind][train_index_parts[model_ind]],
                             train_classes[model_ind][train_index_parts[model_ind]]))
        train_classes_part.append(classes)

    f1_mean_scores = []

    for p in range(16):
        AverageRecognizer = IntentRecognizer(intents, fasttext_embedding_model=fasttext_model, n_splits=1)
        AverageRecognizer.init_network_parameters([params[n_size][0]])
        AverageRecognizer.init_learning_parameters([params[n_size][1]])
        AverageRecognizer.init_model(cnn_word_model, text_size, embedding_size, kernel_sizes, add_network_params=None)

        AverageRecognizer.fit_model(train_requests_part, train_classes_part, to_use_kfold=False, verbose=True)

        train_predictions = AverageRecognizer.predict(train_requests_part)
        AverageRecognizer.report(np.vstack([train_classes_part[i] for i in range(1)]),
                                 np.vstack([train_predictions[i] for i in range(1)]),
                                 mode='TRAIN')

        test_predictions = AverageRecognizer.predict(test_requests)

        f1_scores = AverageRecognizer.report(np.vstack([test_classes[i] for i in range(1)]),
                                             np.vstack([test_predictions[i] for i in range(1)]),
                                             mode='TEST')[0]
        f1_mean_scores.append(np.mean(f1_scores))

    f1_mean_per_size.append(np.mean(f1_mean_scores))
    f1_std_per_size.append(np.std(f1_mean_scores))

    print("___MEAN-STD___:\n size: %d\t f1-mean: %f\tf1-std: %f" % (train_size, f1_mean_per_size[n_size], f1_std_per_size[n_size]))

for n_size, train_size in enumerate(train_sizes):
    print("size: %d\t f1-mean: %f\tf1-std: %f"%(train_size, f1_mean_per_size[n_size], f1_std_per_size[n_size]))


