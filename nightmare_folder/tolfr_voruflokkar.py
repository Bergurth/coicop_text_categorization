import pandas as pd

""""
Markmið að skoða gögnin og ná í tölfræðilegar upplýsingar.
Tókun, lemmun og vinnsla gagna kemur seinna. 


Tölfræðilegar upplýsingar um vöruflokkana koma hér. 


"""

df = pd.read_csv("testDataCOICOP.csv", sep=",")
#print(df.head())


bara_labels = df["Heading"]
bara_vorunumer = df["coicop2018"]

#print(bara_labels)


counter = 0 

#TODO: skrá niður all flokka sem eru í notkun í flokkunarkerfinu. 
#Enga tvisvar. 
label_geymsla = []
for vorulysing in bara_labels : 
    #print(i)
    if(vorulysing not in label_geymsla):
        label_geymsla.append(vorulysing)


#print(len(label_geymsla), " : Hversu margir ólíkir flokkar eru í COICOP")
#print(len(bara_labels), " : Hversu margar vörur eru til yfir höfuð")


#Ókláraður histogram kóði.

from collections import defaultdict
import tqdm
def get_label_counter(label_list):
    label_count = defaultdict(int)
    for word in tqdm.tqdm(label_list):
        label_count[word] += 1 

    return label_count


#print(get_label_counter(bara_labels), " :    Dictionary af öllum labels með tölu yfir hversu oft þeir koma fyrir.")

label_count_listi = get_label_counter(bara_labels)
label_vorulisti_count = get_label_counter(bara_vorunumer)




#Við erum með 90 labels í geymslunni. 

#create_text_file = open("geymsla_f_outputs/labels.txt", "w+")
#
#for i in label_geymsla:
#    create_text_file.write(i + "\n")
#







label_count_listi_sorted = sorted(label_count_listi.items(), key=lambda Heading: -Heading[1])

label_count_listi_sorted_asc = sorted(label_count_listi.items(), key=lambda Heading: Heading[1])

label_vorulisti_count_sorted = sorted(label_vorulisti_count.items(), key=lambda Heading: Heading[1])


print(len(label_vorulisti_count_sorted))

#print(type(label_count_listi), "  :  þetta er label count listi type")

total_counter = len(df["vorulysing"])


#print(total_counter)
outlier_labels = {}
label_perc = 0 
for key,value in label_count_listi_sorted_asc:
    #print(key,value)
    if(label_perc < (0.01*total_counter)):
        outlier_labels[key] = value
        label_perc += value



#print(outlier_labels)
print(len(outlier_labels.values()))




#print(label_count_listi_sorted[0][0])

create_text_file = open("geymsla_f_outputs/label_counter.txt", "w+")

increment = 0 
for key,value in label_count_listi_sorted:
    create_text_file.write(str(key))
    create_text_file.write("   ")
    create_text_file.write(str(label_vorulisti_count_sorted[increment][0]))
    create_text_file.write("   ")
    create_text_file.write(str(value))
    create_text_file.write("\n")
    increment += 1 



import matplotlib.pyplot as plt
#plt.hist([v for v in label_count_listi.values() if v > 4], bins="auto")
#plt.show()

#plt.bar(label_count_listi.keys(), label_count_listi.values(), width = 1000, color='g')
#plt.show()

#plt.bar(list(label_count_listi.keys()), label_count_listi.values(), color='g')
#plt.show()




