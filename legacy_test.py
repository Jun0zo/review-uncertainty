# train with my nn

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BertWithNN(nn.Module):
    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        dropout_output = self.dropout(pooled_output)
        out = self.fc(dropout_output)
        return out
    

# Load and preprocess the data
df = pd.read_csv("data/train_data.csv")
labels = df["Label"].map({"높음": 0, "보통": 1, "낮음": 2})
texts = df["Text"]

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")

# Create the input dataset
input_dataset = TensorDataset(
    encoded_texts["input_ids"],
    encoded_texts["attention_mask"],
    torch.tensor(labels.tolist())
)

# Split the data into train, validation, and test sets
train_size = int(0.8 * len(input_dataset))
val_size = (len(input_dataset) - train_size) // 2
test_size = len(input_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(input_dataset, [train_size, val_size, test_size])

# Create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the BERT model for sequence classification
model = BertWithNN()
model.load_state_dict(torch.load('models/bert_monte_carlo_dropout_model.pth'))

epochs = 10
num_samples = 100  # Choose an appropriate number of samples

# Evaluation on the validation set
# model.eval()

cnt = 0
for text, label in zip(texts, labels):
    output_list = []
    for _ in range(num_samples):
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs, dim=1)

            output_list.append(outputs)
    
    monte_carlo_preds = torch.cat(output_list, dim=0)
    # print(torch.mean(output_list, dim=0))
    # print(torch.std(output_list, dim=0))
    # print(torch.var(output_list, dim=0))
    prediction_mean = torch.mean(monte_carlo_preds, dim=0)
    prediction_std = torch.std(monte_carlo_preds, dim=0)
    prediction_entropy = -torch.sum(prediction_mean * torch.log(prediction_mean), dim=0)

    print('t', torch.argmax(torch.mean(monte_carlo_preds, dim=0)), label, prediction_std - torch.min(prediction_std))
    
    if torch.argmax(torch.mean(monte_carlo_preds, dim=0)) == label:
        cnt += 1
    else:
        print("no ! ")
    
print('acc :', cnt / len(texts))