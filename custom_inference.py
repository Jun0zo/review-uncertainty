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
text = "컬러 이쁘네요~ 블랙손잡이바디가 갖고싶어서 여러개보던중에 맘에들어서 구매했어요~ 내구성은 좀 써봐야 확실히알겠지만 캐리어만봐도 여행전 설레이는기분이 더 드네요 ^^ 가까이찍은사진있으니 자세히 보세요~"

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_texts = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Initialize the BERT model for sequence classification
model = BertWithNN()
model.load_state_dict(torch.load('models/bert_monte_carlo_dropout_model.pth'))
print(model.parameters())

num_samples = 5  # Choose an appropriate number of samples
predictions = []


output_list = []
for _ in range(num_samples):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs, dim=1)

        output_list.append(outputs)
        print(predicted_labels)
        print(outputs)

print(output_list)

output_list = torch.cat(output_list, dim=0)
print(torch.mean(output_list, dim=0))
print(torch.std(output_list, dim=0))
print(torch.var(output_list, dim=0))

