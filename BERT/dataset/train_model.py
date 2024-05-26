# -*- coding: utf-8 -*-
"""
This is the notebook which performs the final model of textual analysis. \
We expect you to have run the translation.ipynb notebook and thus have data files for french/italian and german listings. \
This code was inspired from the Hugging Face tutorial: https://huggingface.co/docs/transformers/main_classes/trainer.

"""

import numpy as np
import pandas as pd
from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch 

"""Specify the number of epochs and batch size as command line arguments"""
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--samples', type=int, default=200)
# args = parser.parse_args()
nbr_epochs = 1
batch_size = 8
nbr_samples = 100

"""BERT model"""

# Read from files 
#PATH_DATA = '/home/mthery/code/data/'
PATH_DATA = 'BERT/dataset/'
# Load data and create train/test set
df_train = pd.read_csv(PATH_DATA + "train.csv")
df_test = pd.read_csv(PATH_DATA + "test.csv")

# Convert labels to float
df_train['label'] = df_train['label'].astype(float)

X = df_train
dataset = Dataset.from_pandas(X[['tweet', 'label']], preserve_index=False) 
dataset = dataset.train_test_split(test_size=0.1) 

# Num_labels = 1 indicates we are doing a regression. 
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=1)

# Tokenize descriptions to transform text into 512 tokens.
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased") 

def tokenize_function(data):
    return tokenizer(text=data['tweet'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model.resize_token_embeddings(len(tokenizer))

# At each epoch, we compute the mean poisson deviance on the validation set. 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

# We trained the model with the specified arguments. 
training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=8,
                                  num_train_epochs=nbr_epochs,
                                  save_total_limit = 2,
                                  save_strategy = "epoch",
                                  load_best_model_at_end="True"
                                  )
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics= compute_metrics
)
trainer.train()
trainer.evaluate()

# We save the model/tokenizer for later use. 
model.save_pretrained("model")
tokenizer.save_pretrained("tokenizer")

"""Prediction on test set"""
# Load the model/tokenizer
model = AutoModelForSequenceClassification.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

trainer = Trainer(model=model)

def pipeline_prediction_french(X_test):
  dataset = Dataset.from_pandas(X_test[['tweet']],preserve_index=True) 
  tokenized_datasets = dataset.map(tokenize_function)
  raw_pred, _, _ = trainer.predict(tokenized_datasets) 
  return raw_pred

# pred_french = pipeline_prediction_french(data_fr_test)
# pred_french= pd.DataFrame(pred_french).set_index(data_fr_test.index).sort_index()
# print("The accuracy for data on test set is:", mean_squared_error(data_fr_test['pred'], pred_french))

used_data = df_test
used_data['pred_bert'] = pipeline_prediction_french(df_test)
acc = accuracy_score(used_data['label'], used_data['pred_bert'])
F1 = f1_score(used_data['label'], used_data['pred_bert'], average='macro')
Precision = precision_score(used_data['label'], used_data['pred_bert'], average='macro')
Recall = recall_score(used_data['label'], used_data['pred_bert'], average='macro')

print(f"Accuracy: {acc:.3f}, F1: {F1:.3f}, Precision: {Precision:.3f}, Recall: {Recall:.3f}")

# Save french predictions
used_data.to_pickle(PATH_DATA + 'pred_bert.csv')
