import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
import os

print("PyTorch Version:", torch.__version__)

# 1. Load Processed Data
print("Loading data...")
X_train_df = pd.read_csv('data/processed/X_train.csv')
X_test_df = pd.read_csv('data/processed/X_test.csv')
y_train_df = pd.read_csv('data/processed/y_train.csv')
y_test_df = pd.read_csv('data/processed/y_test.csv')

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
y_train = torch.tensor(y_train_df.values.ravel(), dtype=torch.long)
y_test = torch.tensor(y_test_df.values.ravel(), dtype=torch.long)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Define the ANN Architecture
class CreditRiskANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CreditRiskANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

input_dimension = X_train.shape[1] # Should be 93
num_classes = len(torch.unique(y_train)) # P1, P2, P3, P4
print(f"Initializing ANN with input_dim={input_dimension} and num_classes={num_classes}")

model = CreditRiskANN(input_dimension, num_classes)

# 3. Setup Training Configuration
# Use CrossEntropyLoss (it handles Softmax internally)
# Incorporate Class Weights to handle imbalance (more defaults vs good loans)
class_counts = torch.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (num_classes * class_counts.float())
print("Class Weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 4. Training Loop
num_epochs = 20
print("\nStarting Training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 5. Evaluation
print("\nEvaluating Model...")
model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Metrics
# Target mapping: 0=P1, 1=P2, 2=P3, 3=P4
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Determine num_classes properly for roc_auc handling
n_classes = len(np.unique(all_labels))
if n_classes > 2: # ovr for multiclass
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
else:
    roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])

print(f"\n--- TEST SET METRICS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Save the model
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/ann_model.pth')
print("\nModel saved to saved_models/ann_model.pth")
