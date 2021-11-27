"""Fasttext tester for COICOP project"""
import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
from numpy import loadtxt
import numpy as np 


df = pd.read_csv("testDataCOICOP.csv", sep=",")

#lines =  np.genfromtxt('allar_vorur.txt',dtype='str', delimiter="\n")
#df.drop(["vorulysing"],1)
#df["vorulysing"] = lines

df = df.drop("Heading", 1)

# new_col = []
# new_string = ""
# for label in df["Heading"]:
#     for letter in range(len(label)):  
#         if(label[letter] == " "):
#             new_string += "-"
#         else:
#             new_string += label[letter]    
#     new_col.append(new_string)
#     new_string = ""
    


#print(new_col[100])

#df.drop("Heading",1)
#df["Heading"] = new_col

#print(df["Heading"])




training_data = df.sample(frac=0.8, random_state=20)
testing_data = df.drop(training_data.index)


#print((df["Heading"].sample(10)))


create_text_file = open("geymsla_f_outputs/dataframe_dataset.txt", "w+")

for key, value in training_data.values:
    create_text_file.write(str(key))
    create_text_file.write(" ")
    create_text_file.write(str(value))
    create_text_file.write("\n")


#print(f"No. of training examples: {training_data.shape[0]}")
#print(f"No. of testing examples: {testing_data.shape[0]}")


create_text_file = open("geymsla_f_outputs/training_dataset.txt", "w+")

for key,value in training_data.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")


create_text_file = open("geymsla_f_outputs/testing_data.txt", "w+")

for key,value in testing_data.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")



model = fasttext.train_supervised(input="geymsla_f_outputs/training_dataset.txt")

print(model.test("geymsla_f_outputs/testing_data.txt", k = 1))

output_data = testing_data

def predict(row):
    return model.predict(row['vorulysing'])
output_data['coicop2018'] = output_data.apply(predict,axis=1)

#print(output_data)

create_text_file = open("geymsla_f_outputs/output_data_fasttext.txt", "w+")

for key,value in output_data.values:
    create_text_file.write(str(key))
    #create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write("\n")

