import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
from numpy import loadtxt
import numpy as np 


df = pd.read_csv("dataset_no_outliers.csv", sep=",")


#df = df.drop("Heading", 1)

#  outlier_labels = {}
#  label_perc = 0 
#  for key,value in label_count_listi_sorted_asc:
#      #print(key,value)
#      if(label_perc < (0.01*total_counter)):
#          outlier_labels[key] = value
#          label_perc += value


training_data = df.sample(frac=0.8, random_state=10)
testing_data = df.drop(training_data.index)



create_text_file = open("geymsla_f_outputs/training_dataset_no_outliers.txt", "w+")

for key,value in training_data.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")


create_text_file = open("geymsla_f_outputs/testing_data_no_outliers.txt", "w+")

for key,value in testing_data.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")



model = fasttext.train_supervised(input="geymsla_f_outputs/training_dataset_no_outliers.txt")

print(model.test("geymsla_f_outputs/testing_data.txt", k = 1))

output_data = testing_data

def predict(row):
    return model.predict(row['vorulysing'])
output_data['coicop2018'] = output_data.apply(predict,axis=1)

#print(output_data)

create_text_file = open("geymsla_f_outputs/output_data_fasttext_no_outliers.txt", "w+")

for key,value in output_data.values:
    create_text_file.write(str(key))
    #create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write("\n")



model = fasttext.train_supervised(input="geymsla_f_outputs/training_dataset_no_outliers.txt")

print(model.test("geymsla_f_outputs/testing_data.txt", k = 1))

output_data = testing_data

def predict(row):
    return model.predict(row['vorulysing'])
output_data['coicop2018'] = output_data.apply(predict,axis=1)


create_text_file = open("geymsla_f_outputs/output_data_fasttext_no_outliers.txt", "w+")

for key,value in output_data.values:
    create_text_file.write(str(key))
    #create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write("\n")