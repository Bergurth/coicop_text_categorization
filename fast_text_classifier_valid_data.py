import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
from numpy import loadtxt
import numpy as np 


df_train = pd.read_csv("train_coicop.csv", sep=",")
df_test = pd.read_csv("test_coicop.csv", sep=",")
df_valid = pd.read_csv("dev_coicop.csv", sep=",")


#df_train = df_train.drop["Heading"]
#df_test = df_test.drop["Heading"]
#df_valid = df_valid.drop["Heading"]




#df = df.drop("Heading", 1)

#  outlier_labels = {}
#  label_perc = 0 
#  for key,value in label_count_listi_sorted_asc:
#      #print(key,value)
#      if(label_perc < (0.01*total_counter)):
#          outlier_labels[key] = value
#          label_perc += value


#training_data = df.sample(frac=0.8, random_state=15)
#testing_data = df.drop(training_data.index)
#valid_data = testing_data[0:(len(testing_data)//2)]
#valid_data = testing_data[(len(testing_data)//2):]




create_text_file = open("geymsla_f_outputs/training_dataset_valid.txt", "w+")

for key,value in df_train.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")

create_text_file = open("geymsla_f_outputs/testing_data_valid.txt", "w+")
for key,value in df_test.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")


create_text_file = open("geymsla_f_outputs/validation_data.txt", "w+")
for key,value in df_valid.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")



#TODO: autotuneValidationFile tekur 5 min að keyra, taktu hana út og skiptu fyrir epoch = 30
model = fasttext.train_supervised(input="geymsla_f_outputs/training_dataset_valid.txt", autotuneValidationFile="geymsla_f_outputs/testing_data_valid.txt")

print(model.test("geymsla_f_outputs/validation_data.txt"))

output_data = df_test
counter = 0 
def predict(row):
    return model.predict(row['vorulysing'])
output_data['coicop2018'] = output_data.apply(predict,axis=1)

print(len(output_data["coicop2018"]), " : þetta er len af output data")


create_text_file = open("geymsla_f_outputs/output_data_fasttext_valid.txt", "w+")


print(len(output_data.values), " : þetta er len af output data values")


for key,value in output_data.values:
    create_text_file.write(str(key))
    #create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write("\n")


model.save_model("fastext_model.bin")



#print(model.test("geymsla_f_outputs/training_dataset_valid.txt",k=1))

while(value != ""):
    print("----" * 20)
    value = input("Please enter a product description:\n")
    description_arr = value.split(" ")

    #desc_vect = vectorizer.transform(description_arr)
    prediction = model.predict(value)
    #predicted_heading = cat_dict[prediction[0]]['coicop2018'].tolist()[0]
    print("predicted COICOP category: ",prediction[0])
    #print("With heading: ",predicted_heading)
    print("----" * 20)
    print()

