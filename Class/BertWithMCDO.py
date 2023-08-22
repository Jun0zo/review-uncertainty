import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
from typing import Tuple

class BertWithMCDO(nn.Module):
    """
    BERT model for sequence classification with Monte Carlo Dropout
    """
    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithMCDO, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 512)  # Adding the first FC layer
        self.fc2 = nn.Linear(512, num_labels)  # Adding the second FC layer

    def load(self, path):
        """
        Load the model from the path
        :param path: path to load the model
        :return: None
        
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        Save the model to the path
        :param path: path to save the model
        :return: None
        
        """
        torch.save(self.state_dict(), path)

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor] :
        """
        Feed input to BERT and the classifier to compute logits.
        :param input_ids: input ids
        :param attention_mask: attention mask
        :return: logits

        """
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        dropout_output = self.dropout(pooled_output)
        fc1_output = self.fc1(dropout_output)
        fc1_output = nn.functional.relu(fc1_output)  # Applying ReLU activation
        fc2_output = self.fc2(fc1_output)
        return fc2_output, bert_output