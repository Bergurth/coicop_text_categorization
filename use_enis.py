from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import torch
from transformers import AutoTokenizer
import os
import json
import pandas as pd



device = "cpu"
working_dir = os.getcwd()
print(working_dir)

# path to the dir above model dir containing config.json
model_path = os.path.join(working_dir, 'XLMR-ENIS-model/XLMR-ENIS-finetuned-coicop-classification-i')

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

tokenizer = AutoTokenizer.from_pretrained("vesteinn/XLMR-ENIS", use_fast=True)

os.chdir(model_path)
# now us the name of the model dir
model = AutoModelForSequenceClassification.from_pretrained('XLMR-ENIS-finetuned-coicop-classification').to(device)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(nlp("Mjólk"))

def get_prediction(product_description):
    predict_obj = nlp(product_description)[0]
    return_obj = predict_obj
    # mydict.keys()[mydict.values().index(16)]
    
    return_obj['label'] = inv_cat_map[int(predict_obj['label'].split("_")[1])]
    return return_obj


value = "start"
while(value != ""):
    print("----" * 20)
    value = input("Please enter a product description:\n")
    response = get_prediction(value)
    print(response)

    predicted_heading = cat_dict[response['label']]['Heading'].tolist()[0]
    print(predicted_heading)

