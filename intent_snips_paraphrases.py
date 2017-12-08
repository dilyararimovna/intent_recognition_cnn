import pandas as pd
import os
import numpy as np

# path = '/home/dilyara/data/data_files/snips'
# bpe_nmt_data = pd.read_csv(os.path.join(path,'paraphrases/BPE_paraphrases_nmt_endeen_with_labels_pandas.csv'),
#                        sep=',', names=['AddToPlaylist', 'BookRestaurant', 'GetWeather',
#                                        'PlayMusic', 'RateBook', 'SearchCreativeWork',
#                                        'SearchScreeningEvent', 'request'], header=0)
# print(bpe_nmt_data.head())
#
# nmt_data = pd.read_csv(os.path.join(path,'paraphrases/paraphrases_nmt_endeen_with_labels_pandas.csv'),
#                        sep=',', names=['AddToPlaylist', 'BookRestaurant', 'GetWeather',
#                                        'PlayMusic', 'RateBook', 'SearchCreativeWork',
#                                        'SearchScreeningEvent', 'request'], header=0)
# print(nmt_data.head())
#
# all_intent_data = pd.read_csv(os.path.join(path,'paraphrases/all_intent_data.csv'),
#                        sep=',')
# print(all_intent_data.head())
#
# data = pd.concat([all_intent_data, bpe_nmt_data, nmt_data])
# data.to_csv(os.path.join(path, 'paraphrases/intent_data_with_paraphrases.csv'), index=False)
#
# train_data = pd.read_csv(os.path.join(path, "snips_ner_gold/snips_ner_gold_0/snips_train_0"))
# test_data = pd.read_csv(os.path.join(path, "snips_ner_gold/snips_ner_gold_0/snips_test_0"))
#
# test_ids = []
# for i in range(test_data.shape[0]):
#     for j in range(all_intent_data.shape[0]):
#         if all_intent_data.loc[j, 'request'] == test_data.loc[i, 'request']:
#             test_ids.append(j)
#
# train_ids = list(pd.Int64Index(np.arange(all_intent_data.shape[0])).difference(test_ids))
#
# train_bpe_nmt_data = bpe_nmt_data.loc[train_ids, :]
# train_nmt_data = nmt_data.loc[train_ids, :]
# test_bpe_nmt_data = bpe_nmt_data.loc[test_ids, :]
# test_nmt_data = nmt_data.loc[test_ids, :]
#
# train_data_with_paraphrases = pd.concat([train_data, train_bpe_nmt_data])
# train_data_with_paraphrases.to_csv(os.path.join(path,
#                                                 'paraphrases/intent_train_data_with_bpe_paraphrases.csv'), index=False)
#
# test_data_with_paraphrases = pd.concat([test_data, test_bpe_nmt_data])
# test_data_with_paraphrases.to_csv(os.path.join(path,
#                                                'paraphrases/intent_test_data_with_bpe_paraphrases.csv'), index=False)
#
# train_data_with_paraphrases = pd.concat([train_data, train_nmt_data])
# train_data_with_paraphrases.to_csv(os.path.join(path,
#                                                 'paraphrases/intent_train_data_with_paraphrases.csv'), index=False)
#
# test_data_with_paraphrases = pd.concat([test_data, test_nmt_data])
# test_data_with_paraphrases.to_csv(os.path.join(path,
#                                                'paraphrases/intent_test_data_with_paraphrases.csv'), index=False)

# train_data_with_paraphrases = pd.read_csv(os.path.join(path, 'paraphrases/intent_train_data_with_paraphrases.csv'),
#                                           sep=',')
# test_data_with_paraphrases = pd.read_csv(os.path.join(path, 'paraphrases/intent_test_data_with_paraphrases.csv'),
#                                          sep=',')





