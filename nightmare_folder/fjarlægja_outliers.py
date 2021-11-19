import numpy as np 
import pandas as pd
from collections import defaultdict
from pandas.core.indexes.base import Index
import tqdm
lines =  np.genfromtxt('allar_vorur.txt',dtype='str', delimiter="\n")

counter = 1
for i in lines:
    #print (i)
    if(counter == 100):
        break
    counter+=1

#print(len(i))

df = pd.read_csv("testDataCOICOP.csv", sep=",")

df = df.drop("Heading", 1)

bara_labels = df["coicop2018"]

def get_label_counter(label_list):
    label_count = defaultdict(int)
    for word in tqdm.tqdm(label_list):
        label_count[word] += 1 

    return label_count

print(len(df), " : þetta er len df fyrir cutt") 

label_count_listi = get_label_counter(bara_labels)

label_count_listi_sorted_asc = sorted(label_count_listi.items(), key=lambda Heading: Heading[1])

outlier_labels = []
label_perc = 0 
total_counter = len(df["vorulysing"])
for key,value in label_count_listi_sorted_asc:

    if(label_perc <= (0.02*total_counter)):
        outlier_labels.append(key)
        label_perc += value

print(label_perc)
print(0.01*total_counter)
print(outlier_labels)

tester = len(df)



"""Taka út öll tilvik af outlier flokkunum."""
"""Googla pop í dataframe, hugmyndin er að eyða út öllum outlier flokkum
áður en við trainum á þetta. """


counter = 0



print(len(df), " : þetta er len df fyrir cutt") 

stopper = 0 
bara_labels = df["coicop2018"]
#print(df["coicop2018"])
for key in range (len(df)):
    if(bara_labels[key] in outlier_labels):
        df = df.drop([key])
        stopper += 1

print(len(df), " : þetta er len df eftir cutt")
#print(df.keys())
df.to_csv("dataset_no_outliers.csv", sep = ",", index = False)