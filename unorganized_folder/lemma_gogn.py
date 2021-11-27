"""lemma öll gögn áður en við keyrum þau í gegn með Naive bayes"""
import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
from numpy import loadtxt
import numpy as np 
import tokenizer
import islenska 
from islenska import Bin

"""
breytunöfn standa fyrir
wtos = word to sentence
ltos = list to sentance


"""

b = Bin()

df = pd.read_csv("testDataCOICOP.csv", sep=",")

df = df.drop("Heading",1)

df_vorur = df["vorulysing"]

CC_lines = [None] * len(df_vorur)
#wtos breytan er notuð til að halda utan um hvar setningin er 
# í gönunum(df)
#
wtos = 0
jlist = []

def naive_bin_lemma(token):
    candidates = b.lookup_lemmas_and_cats(token)
    if not candidates:
        return token
    return candidates.pop()[0]

lemma_list = []
for line in df_vorur:
    line = line.split()
    for word in line:
        word = naive_bin_lemma(word)
        jlist.append(word)
  
    ltos = (" ".join(jlist))
    lemma_list.append(ltos)
    jlist = []    

#print(len(lemma_list))

"""Frá verkefni 1 """

df = df.drop("vorulysing",1)

df["vorulysing"] = lemma_list

df.to_csv("dataset_with_lemmas.csv", sep = ",", index = False)





