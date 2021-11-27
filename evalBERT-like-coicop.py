import transformers

from datasets import load_dataset, load_metric
# be in or navigate to the directory where the data is found
"""
e.g.
from google.colab import drive
drive.mount('/content/gdrive')
import os
print(os.getcwd())
os.chdir('/content/gdrive/MyDrive/COICOP-stuff')
print(os.getcwd())

in case of using drive

otherwise just:
os.chdir('/dir-where-data-is')
"""
train_dataset = load_dataset(".", data_files="train.json", field="data")
validation_dataset = load_dataset(".", data_files="validation.json", field="data")
test_dataset = load_dataset(".", data_files="test.json", field="data")

# getting models

from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import torch
from transformers import AutoTokenizer

device = "cpu"

"""
  current model candidates:
  IceBERT-COICOP-classifier,
  XLMR-ENIS-finetuned-coicop-classification-i,
  XLMR-ENIS-finetuned-COICOP-classifier
"""
# IceBERT
model_icebert_26 = AutoModelForSequenceClassification.from_pretrained('IceBERT-COICOP-classifier').to(device)

tokenizer_icebert_26 = AutoTokenizer.from_pretrained('IceBERT-COICOP-classifier', use_fast=True)

nlp_icebert_26 = pipeline("sentiment-analysis", model=model_icebert_26, tokenizer=tokenizer_icebert_26)
# print(nlp_icebert_26("Mjólk"))

# XLMR-ENIS 26 epochs
model_enis_26 = AutoModelForSequenceClassification.from_pretrained('XLMR-ENIS-finetuned-COICOP-classifier').to(device)
tokenizer_enis_26 = AutoTokenizer.from_pretrained('XLMR-ENIS-finetuned-COICOP-classifier', use_fast=True)
nlp_enis_26 = pipeline("sentiment-analysis", model=model_enis_26, tokenizer=tokenizer_enis_26)

# enis 9 epochs
model_enis_9 = AutoModelForSequenceClassification.from_pretrained('XLMR-ENIS-finetuned-coicop-classification-i').to(device)
tokenizer_enis_9 = AutoTokenizer.from_pretrained('XLMR-ENIS-finetuned-coicop-classification-i', use_fast=True)
nlp_enis_9 = pipeline("sentiment-analysis", model=model_enis_9, tokenizer=tokenizer_enis_9)

# getting the category mapping
import json

with open('cat_mapping.json') as json_file:
    cat_mapping_dict = json.load(json_file)
    inv_cat_map = {v: k for k, v in cat_mapping_dict.items()}

# get prediction
def get_prediction(product_description, pl):
    predict_obj = pl(product_description)[0]
    return_obj = predict_obj
    # mydict.keys()[mydict.values().index(16)]
    
    return_obj['label'] = inv_cat_map[int(predict_obj['label'].split("_")[1])]
    return return_obj

# predicting whole dataset
# validation_dataset['train']
for_pred = validation_dataset['train']['text'] # gives list
y_true = validation_dataset['train']['label'] # true list

target_names = [ str(n) for n in range(1, 91) ]
coicop_names = [inv_cat_map[int(i)] for i in target_names]

from sklearn.metrics import classification_report

# IceBERT
icebert_preds = nlp_icebert_26(for_pred)
icebert_pred = [int(pred['label'].split("_")[1]) for pred in icebert_preds]

print(classification_report(y_true, icebert_pred, target_names=coicop_names))

# ENIS - 26
enis_26_preds = nlp_enis_26(for_pred)
enis_26_pred = [int(pred['label'].split("_")[1]) for pred in enis_26_preds]

print(classification_report(y_true, enis_26_pred, target_names=coicop_names))

# ENIS - 9 plotting
enis_9_preds = nlp_enis_9(for_pred)
enis_9_pred = [int(pred['label'].split("_")[1]) for pred in enis_9_preds]

enis_9_pred_cat_names = [inv_cat_map[cat] for cat in icebert_pred]
