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
from sklearn.model_selection import train_test_split
import torch 

"""Specify the number of epochs and batch size as command line arguments"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--samples', type=int, default=200)
args = parser.parse_args()
nbr_french_epochs = args.french_epochs
nbr_german_epochs = args.german_epochs
batch_size = args.batch_size
nbr_samples = args.samples 

# Create training/testing sets
IDS_TRAIN = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/reference_ids_train.npy'
IDS_TEST = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/reference_ids_test.npy'

ids_train = np.load(IDS_TRAIN, allow_pickle=True)
ids_test = np.load(IDS_TEST, allow_pickle=True)

"""French model"""

# Read from files 
PATH_FRENCH = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/data_fr.pkl'
PATH_FRENCH_SAVE = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Predictions/'

# Load data and create train/test set
data_fr = pd.read_pickle(PATH_FRENCH)
data_fr_train = data_fr[data_fr['Property Reference Id'].isin(ids_train)]
data_fr_test = data_fr[data_fr['Property Reference Id'].isin(ids_test)]
#data_fr_train, data_fr_test = train_test_split(data_fr, test_size=0.2, random_state=42)

X = data_fr_train.rename(columns= {'Demand':'label'})
# remove rows with missing labels or missing descriptions
X = X[X['label'].notna()]
X = X[X['tweet'].notna()]
dataset = Dataset.from_pandas(X[['tweet', 'label']], preserve_index=False) 
dataset = dataset.train_test_split(test_size=0.1) 
#dataset

# This loads the model for the french language. 
# Num_labels = 1 indicates we are doing a regression. 
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=1)

# Tokenize descriptions to transform text into 512 tokens.
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased-base") 

def tokenize_function_french(data):
    return tokenizer(text=data['tweet'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function_french, batched=True)

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
                                  num_train_epochs=nbr_french_epochs,
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
  tokenized_datasets = dataset.map(tokenize_function_french)
  raw_pred, _, _ = trainer.predict(tokenized_datasets) 
  return raw_pred

pred_french = pipeline_prediction_french(data_fr_test)
pred_french= pd.DataFrame(pred_french).set_index(data_fr_test.index).sort_index()
print("The poisson deviance for french data on valid set is:", mean_squared_error(data_fr_test['pred'], pred_french))

# Save french predictions
pred_french.to_pickle(PATH_FRENCH_SAVE + 'pred_french.pkl')