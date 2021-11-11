# Data prepping for coicop project

import pandas as pd
import random
from math import floor, ceil
import sys

df = pd.read_csv("testDataCOICOP.csv", sep=",")

just_coicop = df["coicop2018"]

categories = []
for cat in just_coicop:
    if cat not in categories:
        categories.append(cat)

# now to make the 90 different dfs, in a dict of dataFrames
cat_dict = {}
for cat in categories:
    cat_dict[cat] = df[df['coicop2018']==cat]

rando_cat_dict = {}
for cat in categories:
    rando_cat_dict[cat] = cat_dict[cat].sample(frac=1)

"""
 this splits into dev, test, and train dicts of
 dataFrames, having approximately 10%, 10%, and 80%
 respectively
""" 
test_cat_dict = {}
dev_cat_dict = {}
train_cat_dict = {}
for cat in categories:
    cat_size = len(rando_cat_dict[cat])
    train_cat_dict[cat] = rando_cat_dict[cat][:floor(cat_size * 0.8)]
    test_and_dev = rando_cat_dict[cat][:floor(cat_size * - 0.2)]
    test_cat_dict[cat] = test_and_dev[:len(test_and_dev)//2]
    dev_cat_dict[cat] = test_and_dev[len(test_and_dev)//2:]


all_train_dataFrame = pd.concat(train_cat_dict.values(), ignore_index=True)
all_test_dataFrame = pd.concat(test_cat_dict.values(), ignore_index=True)
all_dev_dataFrame = pd.concat(dev_cat_dict.values(), ignore_index=True)

# exporting to csv --- works
"""
all_train_dataFrame.to_csv('train_coicop.csv', index=False)
all_test_dataFrame.to_csv('test_coicop.csv', index=False)
all_dev_dataFrame.to_csv('dev_coicop.csv', index=False)
"""

# json export tests
# all_train_dataFrame.to_json('coicop_train.json') ekki malid

json_dev = all_dev_dataFrame.drop(columns=['Heading'])

# todo fix, check no quotes problem ..
with open('coicop_dev.json', 'w') as f:
    sys.stdout = f
    for index, row in json_dev.iterrows():
        print('{"text": "' + row['vorulysing'] + '", "label": "' + row['coicop2018'] + '"}')
