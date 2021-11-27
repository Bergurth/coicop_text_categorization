"""
  This is a  minimal script for training IceBERT or XLMR-ENIS
  models, to classify description into COICOP2018 classes. 
 
  of the data files:

 "train.json"
 "validation.json"
 "test.json"

  only train.json and validation.json are used during the training and
  evaluation as of now.

 additionally a file cat_mapping.json is needed to maintain the connection
 between the coicop classes actual name and the label given.

 example of contents of one of either train, validation or test .json

 {"data": [
 {"text": "some product description", "label": 1, "idx": 0}
 ,
 {"text": "another product description", "label": 2, "idx": 1}
 ,
 {"text": "yet another product description", "label": 2, "idx": 1}
 ,

 ...

 ...
 ,
 {"text": "final product description in json file", "label": 2, "idx": 1}
 ]}



"""
import transformers

print("transformers version: ", transformers.__version__)

# choose one of these
model_checkpoint = ["vesteinn/IceBERT", "vesteinn/XLMR-ENIS"][0]

batch_size = 16

#================================================================================
#                             Loading the datasets
#================================================================================

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


# selecting a metric

# Metric related
#--------------------------------------------------------------------------------

metric_names = ["f1", "accuracy"] # choose a metric to optimize for in the training
metric_name = metric_names[0]
metric = load_metric(metric_name)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels, average="micro")

#================================================================================
#                             Preprocessing the Data
#================================================================================

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

#================================================================================
#                             Fine-tuning the model
#================================================================================

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=91 # there ar 90 classes, this avoids off by one
    )

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-COICOP-classifier", # choose a descriptive model name
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20, # from 9 to 26 or more seems to be good 
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_validation_dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

task = "custom" # there needs to be a task

trainer.train()

trainer.evaluate()

trainer.save_model()
# trainer.push_to_hub() # if deemed advisable


#================================================================================
#                          continued
#================================================================================

"""
  In case of wanting to continue training from a specific epoch
 specify the number of total epochs in the TrainingArguments like above
 then do like below
"""

args = TrainingArguments(
    f"{model_name}-COICOP-classifier", # <-- same name as before 
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=26, # <-- here adding 6 more epochs, as compared to before 
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

from transformers.trainer_utils import get_last_checkpoint
last_checkpoint = get_last_checkpoint('name-of-model') # dir that it is saved in

trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.evaluate()
trainer.save_model()  # saveing the model again, having trained it some more

