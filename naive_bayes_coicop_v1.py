# Data prepping for coicop project

import pandas as pd
import random
from math import floor, ceil

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

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

print('Shape of training data: ', all_train_dataFrame.shape)
print('Shape of testing data: ', all_test_dataFrame.shape)
train_x_data = all_train_dataFrame.drop(columns=['Heading','coicop2018'])
train_y_data = all_train_dataFrame.drop(columns=['Heading','vorulysing'])

test_x_data = all_test_dataFrame.drop(columns=['Heading','coicop2018'])
test_y_data = all_test_dataFrame.drop(columns=['Heading','vorulysing'])

train_x = train_x_data['vorulysing']
train_y = train_y_data['coicop2018']

test_x = test_x_data['vorulysing']
test_y = test_y_data['coicop2018']

less_train_x = train_x.tolist()
less_train_y = train_y.tolist()

less_test_x = test_x.tolist()
less_test_y = test_y.tolist()

vectorizer = CountVectorizer()

model = MultinomialNB()

less_x_counts = vectorizer.fit_transform(less_train_x)
less_x_test_counts = vectorizer.transform(less_test_x)

model.fit(less_x_counts, less_train_y)

predict_train = model.predict(less_x_counts)
print('Target on train data', predict_train)

accuracy_train = accuracy_score(less_train_y, predict_train)
print('accuracy_score on traindataset :', accuracy_train)

predict_test = model.predict(less_x_test_counts)
print('Target on test data', predict_test)

accuracy_test = accuracy_score(less_test_y, predict_test)
print('accuracy_score on test dataset : ', accuracy_test)

# vectorizer.get_feature_names() # gives long list of vocab words
"""
# example
milk = vectorizer.transform(['Mjólk'])
milk.toarray()[0]
model.predict(milk)
"""

value = "start"
while(value != ""):
    print("----" * 20)
    value = input("Please enter a product description:\n")
    description_arr = value.split(" ")
    desc_vect = vectorizer.transform(description_arr)
    prediction = model.predict(desc_vect)
    predicted_heading = cat_dict[prediction[0]]['Heading'].tolist()[0]
    print("predicted COICOP category: ",prediction[0])
    print("With heading: ",predicted_heading)
    print("----" * 20)
    print()


