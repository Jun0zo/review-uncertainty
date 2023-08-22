# train with my nn

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
from Class.BertWithMCDO import BertWithMCDO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load and preprocess the data
# text = "당신의 눈앞에 펼쳐진 세상은 무궁무진한 이야기로 가득 차 있습니다. 강렬한 햇살 아래 피어나는 꽃들의 화려한 색깔, 바닷가에서 파도와 함께 춤추는 모래알들의 소리, 그리고 사람들의 다양한 감정과 이야기들이 어우러져 하나의 큰 그림을 그려냅니다."

text = "안녕하세요"

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_texts = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Initialize the BERT model for sequence classification
model = BertWithMCDO()
model.load('models/bert_monte_carlo_dropout_model(base).pth')
print(model.parameters())

num_samples = 5  # Choose an appropriate number of samples
predictions = []


output_list = []
for _ in range(num_samples):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs, embedding = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs, dim=1)

        output_list.append(outputs)
        print(predicted_labels)

output_list = torch.cat(output_list, dim=0)
print(torch.mean(output_list, dim=0))
print(sum(torch.mean(output_list, dim=0)))
print(sum(torch.std(output_list, dim=0)))
print(sum(torch.var(output_list, dim=0)))

