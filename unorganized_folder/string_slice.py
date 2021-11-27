"""Fasttext tester for COICOP project"""
import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
import re



df = pd.read_csv("testDataCOICOP.csv", sep=",")

df = df.drop("coicop2018", 1)

new_col = []
new_string = ""
for label in df["Heading"]:
    for letter in range(len(label)):  
        if(label[letter] == " "):
            new_string += "-"
        else:
            new_string += label[letter]    
    new_col.append(new_string)
    new_string = ""
    


print(new_col[100])

df.drop("Heading",1)
df["Heading"] = new_col

print(df["Heading"])







