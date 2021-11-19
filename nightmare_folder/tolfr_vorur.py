
""""
Markmið að skoða gögnin og ná í tölfræðilegar upplýsingar.
Tókun, lemmun og vinnsla gagna kemur í öðru skjali. 

Tölfræðilegar upplýsingar um vörurnar eru hér.


"""

import pandas as pd

df = pd.read_csv("testDataCOICOP.csv", sep=",")
#print(df.head())


bara_vorur = df["vorulysing"]


counter = 0 

#TODO: skrá niður all flokka sem eru í notkun í flokkunarkerfinu. 
#Enga tvisvar. 
vara_single_word = []
for vara in bara_vorur : 
    #print(i)
    if(" " not in vara):
        vara_single_word.append(vara)


print(len(vara_single_word))

# create_text_file = open("single_vorur.txt", "w+")
# for vara in vara_single_word:
#     create_text_file.write(str(vara))
#     create_text_file.write("\n")


allar_vorur = []


create_text_file = open("allar_vorur.txt", "w+")
for vara in bara_vorur:
    create_text_file.write(str(vara))
    create_text_file.write("\n")

