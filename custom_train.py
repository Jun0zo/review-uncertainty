# train with my nn

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

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
print(model.parameters())
# Define the optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device: ", device)
model.to(device)

epochs = 100
loss_graph = []
accuracy_graph = []


for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        # outputs = model(input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()
        
    


    
    # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    avg_train_loss = total_loss / len(train_loader)
    loss_graph.append(avg_train_loss)
    avg_train_accuracy = correct_predictions / len(train_dataset)
    print('avg train accuracy: ', avg_train_accuracy)
    accuracy_graph.append(avg_train_accuracy)

# save loss graph and accuracy graph
import matplotlib.pyplot as plt
plt.plot(loss_graph)
plt.plot(accuracy_graph)
plt.savefig('loss_accuracy_graph.png')



# Evaluation on the validation set
model.eval()
total_val_loss = 0
correct_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()

        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

val_accuracy = correct_predictions / len(val_dataset)
avg_val_loss = total_val_loss / len(val_loader)
print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Evaluation on the test set
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

test_accuracy = correct_predictions / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

# model.save_pretrained("model2_1000")
torch.save(model.state_dict(), 'bert_monte_carlo_dropout_model.pth')
# model.load_state_dict(torch.load('bert_monte_carlo_dropout_model.pth'))
if __name__ == "__main__":
    pass