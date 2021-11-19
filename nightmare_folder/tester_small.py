from matplotlib import colors
from matplotlib.pyplot import tight_layout, title, ylabel
import pandas as pd
import numpy as np
import random
from math import floor, ceil
import fasttext
from numpy import loadtxt
import numpy as np 
from collections import defaultdict
import tqdm
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("testDataCOICOP.csv", sep=",")
bara_labels = df["coicop2018"]


def get_label_counter(label_list):
    label_count = defaultdict(int)
    for word in tqdm.tqdm(label_list):
        label_count[word] += 1 

    return label_count

label_count_listi = get_label_counter(bara_labels)
label_vorulisti_count = get_label_counter(bara_labels)

label_count_listi_sorted = sorted(label_count_listi.items(), key=lambda Heading: -Heading[1])

label_count_listi_sorted_asc = sorted(label_count_listi.items(), key=lambda Heading: Heading[1])

outlier_labels = {}
label_perc = 0 
total_counter = len(bara_labels)
for key,value in label_count_listi_sorted_asc:
    #print(key,value)
    if(label_perc < (0.015*total_counter)):
        outlier_labels[key] = value
        label_perc += value


outlier_list_sorted_asc = sorted(outlier_labels.items(), key=lambda Heading: Heading[1]) 
print(outlier_labels)

# create_text_file = open("geymsla_f_outputs/outlier_voruflokkar.txt", "w+")
# for key,value in outlier_labels.items():
#     create_text_file.write("Flokkur númer : ")
#     create_text_file.write(key)
#     create_text_file.write("  inniheldur ->  ")
#     create_text_file.write(str(value))
#     create_text_file.write("\n")

plot_x,plot_y = zip(*outlier_list_sorted_asc)

type(plot_x)

replace_num = 1
new_x = [i for i in range (len(plot_x)) ]
#print(new_x)
#print(plot_x)

#fig = plt.figure(figsize=(8,4), tight_layout=True)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.set_title('Outlier data ')

plt.figure(figsize=(8,4), tight_layout=True)
plt.title("Outlier datasets")
plt.xlabel('Fjöldi vöruflokka')
plt.ylabel('Fjöldi vara í hverjum flokki')
colors = sns.color_palette("muted")
plt.bar(new_x,plot_y, color=colors[:3], width = 1.5)
plt.show()



# plot_x,plot_y = plt.subplots()
# for tick in plot_x.get_xticklabels():
#     tick.set_rotation(45)
