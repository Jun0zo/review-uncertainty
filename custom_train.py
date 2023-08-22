# train with my nn

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from Class.BertWithMCDO import BertWithMCDO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load and preprocess the data
df = pd.read_csv("data/train_data.csv")
labels = df["Label"].map({"높음": 2, "보통": 1, "낮음": 0})
texts = df["Text"]

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
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
model = BertWithMCDO()

# Define the optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device: ", device)
model.to(device)

epochs = 100
loss_graph = {'train': [], 'validation': []}
accuracy_graph = {'train': [], 'validation': []}

# train the model (train and valid set)
model.train()
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs, embedding = model(input_ids, attention_mask=attention_mask)
        # outputs = model(input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()
        
    # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    avg_train_loss = total_loss / len(train_loader)
    loss_graph['train'].append(avg_train_loss)
    avg_train_accuracy = correct_predictions / len(train_dataset)
    print('avg train accuracy: ', avg_train_accuracy)
    accuracy_graph['train'].append(avg_train_accuracy)

    # evaluate on validation set
    model.eval()
    total_val_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs, embedding = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predicted_labels == labels).item()

        val_accuracy = correct_predictions / len(val_dataset)
        avg_val_loss = total_val_loss / len(val_loader)
        loss_graph['validation'].append(avg_val_loss)
        accuracy_graph['validation'].append(val_accuracy)

# Plot the loss and accuracy graphs
plt.plot(loss_graph['train'], label='Train Loss')
plt.plot(loss_graph['validation'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('outputs/loss_graph.png')

plt.plot(accuracy_graph['train'], label='Train Accuracy')
plt.plot(accuracy_graph['validation'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('outputs/accuracy_graph.png')

plt.plot(accuracy_graph['train'], label='Accuracy')
plt.plot(loss_graph['train'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy and Loss')
plt.savefig('outputs/training_acc_loss.png')

# Evaluation on the test set
model.eval()
correct_predictions = 0

# testset
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs, embedding = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

test_accuracy = correct_predictions / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save("models/bert_monte_carlo_dropout_model(multi-case).pth")