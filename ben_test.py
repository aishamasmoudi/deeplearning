import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import torch
from sklearn.model_selection import train_test_split



# Load the JSON file into a DataFrame
data = pd.read_json('Data/dataset.json')

# Display the first few rows of the DataFrame
print(data.head())



rows = []
for post_id, content in data.items():
    text = " ".join(content["post_tokens"])
    # Take the majority label or the first annotator's label as the example
    label = content["annotators"][0]["label"]
    rows.append({"text": text, "label": label})

data_row = pd.DataFrame(rows)
print(data_row.head())

#split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data_row['text'], data_row['label'], test_size=0.2, random_state=42
)


#3. Bert encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Encode text
encoded_data = tokenizer.batch_encode_plus(
    data_row['text'].tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=512, 
    return_tensors='pt'
)

#4. Create PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TextDataset(encoded_data, data_row['label'].tolist())

#5. Train a Classifier Using BERT
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load BERT with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)

# Create Trainer instance
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset               
)

# Train the model
trainer.train()


