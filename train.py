import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Load and preprocess the data
df = pd.read_csv("train_data.csv")
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
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device: ", device)
model.to(device)

epochs = 1000
loss_graph = []
accuracy_graph = []
correct_predictions = 0

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # draw loss graph
        loss_graph.append(loss.item())

        # draw accuracy graph
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()
        accuracy_graph.append(correct_predictions / len(train_dataset))


    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

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

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_val_loss += loss.item()

        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
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

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

test_accuracy = correct_predictions / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save_pretrained("model_1000")

if __name__ == "__main__":
    pass