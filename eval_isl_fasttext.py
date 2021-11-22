# isl_fasttext_rmh_fb-finetuned-coicop.bin

import fasttext
import json
import pandas as pd

model = fasttext.load_model("isl_fasttext_rmh_fb-finetuned-coicop.bin")

value = "start"

# all for the heading
with open('cat_mapping.json') as json_file:
    cat_mapping_dict = json.load(json_file)
    inv_cat_map = {v: k for k, v in cat_mapping_dict.items()}

df = pd.read_csv("testDataCOICOP.csv", sep=",")

just_coicop = df["coicop2018"]

categories = []
for cat in just_coicop:
    if cat not in categories:
        categories.append(cat)

cat_dict = {}
for cat in categories:
    cat_dict[cat] = df[df['coicop2018']==cat]

df_valid = pd.read_csv("dev_coicop.csv", sep=",")



create_text_file = open("geymsla_f_outputs/validation_data.txt", "w+")
for key,value in df_valid.values:
    create_text_file.write("__label__")
    create_text_file.write(str(value))
    create_text_file.write(" ")
    create_text_file.write(str(key))
    create_text_file.write("\n")

output_data = df_valid
counter = 0 
def predict(row):
    return model.predict(row['vorulysing'])
thing = output_data.apply(predict,axis=1)


thing2 = thing.to_numpy()

predictions = [tp[0] for tp in thing2]



pred_y = [pred[0].split("__")[2] for pred in predictions]

# print(pred_y) # this is predictions

true_y = df_valid['coicop2018'].to_numpy()

from sklearn.metrics import classification_report

print(classification_report(true_y, pred_y))
