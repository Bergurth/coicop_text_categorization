import fasttext
import json
import pandas as pd

model = fasttext.load_model("fastext_model.bin")

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

#

# Please enter a product description:
# coca cola
# predicted COICOP category:  ('__label__01.2.6.0',)


while(value != ""):
    print("----" * 20)
    value = input("Please enter a product description:\n")
    description_arr = value.split(" ")
    prediction = model.predict(value)
    print("predicted COICOP category: ",prediction[0])
    predicted_heading = cat_dict[prediction[0][0].split("__")[2]]
    print("heading: " , predicted_heading)
    print("----" * 20)
    print()

