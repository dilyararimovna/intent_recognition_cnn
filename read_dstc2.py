
import json
import os
import pandas as pd

datapath = '/home/dilyara/data/data_files'
dpath = os.path.join(datapath, 'dstc2')
version = None

train_data = []
valid_data = []
test_data = []

with open(os.path.join(dpath, 'dstc2-trn.jsonlist')) as read:
    for line in read:
        line = line.strip()
        # if empty line - it is the end of dialog
        if not line:
            continue

        replica = json.loads(line)
        if 'goals' not in replica.keys():
            # bot reply
            continue
        curr_intents = []
        if replica['dialog-acts']:
            for act in replica['dialog-acts']:
                for slot in act['slots']:
                    if slot[0] == 'slot':
                        curr_intents.append(act['act'] + '_' + slot[1])
                    else:
                        curr_intents.append(act['act'] + '_' + slot[0])
                if len(act['slots']) == 0:
                    curr_intents.append(act['act'])
        else:
            if replica['text']:
                curr_intents.append('unknown')
            else:
                continue
        train_data.append({'text': replica['text'],
                           'intents': ' '.join(curr_intents)})

with open(os.path.join(dpath, 'dstc2-val.jsonlist')) as read:
    for line in read:
        line = line.strip()
        # if empty line - it is the end of dialog
        if not line:
            continue

        replica = json.loads(line)
        if 'goals' not in replica.keys():
            # bot reply
            continue
        curr_intents = []
        if replica['dialog-acts']:
            for act in replica['dialog-acts']:
                for slot in act['slots']:
                    if slot[0] == 'slot':
                        curr_intents.append(act['act'] + '_' + slot[1])
                    else:
                        curr_intents.append(act['act'] + '_' + slot[0])
                if len(act['slots']) == 0:
                    curr_intents.append(act['act'])
        else:
            if replica['text']:
                curr_intents.append('unknown')
            else:
                continue
        valid_data.append({'text': replica['text'],
                           'intents': ' '.join(curr_intents)})

with open(os.path.join(dpath, 'dstc2-tst.jsonlist')) as read:
    for line in read:
        line = line.strip()
        # if empty line - it is the end of dialog
        if not line:
            continue

        replica = json.loads(line)
        if 'goals' not in replica.keys():
            # bot reply
            continue
        curr_intents = []
        if replica['dialog-acts']:
            for act in replica['dialog-acts']:
                for slot in act['slots']:
                    if slot[0] == 'slot':
                        curr_intents.append(act['act'] + '_' + slot[1])
                    else:
                        curr_intents.append(act['act'] + '_' + slot[0])
                if len(act['slots']) == 0:
                    curr_intents.append(act['act'])
        else:
            if replica['text']:
                curr_intents.append('unknown')
            else:
                continue
        test_data.append({'text': replica['text'],
                           'intents': ' '.join(curr_intents)})


train_data = pd.DataFrame(train_data)
train_data.to_csv(os.path.join(dpath, 'dstc2_train.csv'), index=False)
valid_data = pd.DataFrame(valid_data)
valid_data.to_csv(os.path.join(dpath, 'dstc2_valid.csv'), index=False)
test_data = pd.DataFrame(test_data)
test_data.to_csv(os.path.join(dpath, 'dstc2_test.csv'), index=False)
