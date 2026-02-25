"""
UPI Fraud Detection - LSTM (RNN) Model
Trains an LSTM that reads sequences of 10 past transactions per user
and classifies the most recent one as fraudulent or legitimate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import json

print("PyTorch version:", torch.__version__)

# ─────────────────────────────────────────────
# 1. Load Processed Sequence Data
# ─────────────────────────────────────────────
PROJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')

print("Loading sequence data...")
X_train = np.load(os.path.join(PROC_DIR, 'X_train.npy'))
X_test  = np.load(os.path.join(PROC_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(PROC_DIR, 'y_train.npy'))
y_test  = np.load(os.path.join(PROC_DIR, 'y_test.npy'))

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"Fraud rate (train): {y_train.mean()*100:.2f}%  |  (test): {y_test.mean()*100:.2f}%")

SEQ_LEN   = X_train.shape[1]   # 10
N_FEATURES = X_train.shape[2]  # 10 features

# Convert to PyTorch tensors
X_tr = torch.tensor(X_train, dtype=torch.float32)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.float32)

# ─────────────────────────────────────────────
# 2. DataLoaders
# ─────────────────────────────────────────────
BATCH_SIZE = 256

train_loader = DataLoader(
    TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False
)

# ─────────────────────────────────────────────
# 3. LSTM Model Architecture
# ─────────────────────────────────────────────
class FraudLSTM(nn.Module):
    """
    Two-layer stacked LSTM followed by fully-connected classification head.
    Input  : (batch, seq_len=10, n_features=10)
    Output : (batch, 1) — probability of fraud
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(FraudLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Use only the last time-step output
        last_out = lstm_out[:, -1, :]          # (batch, hidden_dim)
        logit = self.classifier(last_out)       # (batch, 1)
        return logit.squeeze(1)                 # (batch,)


model = FraudLSTM(input_dim=N_FEATURES, hidden_dim=128, num_layers=2, dropout=0.3)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. Loss, Optimizer & Class Weighting
# ─────────────────────────────────────────────
# Fraud is rare (~2.3%) — weight the positive class to boost recall
fraud_count  = int(y_train.sum())
legit_count  = int(len(y_train) - fraud_count)
# Cap pos_weight at 5 to avoid over-aggressive fraud prediction
raw_weight   = legit_count / fraud_count
pos_weight   = torch.tensor([min(raw_weight, 5.0)], dtype=torch.float32)
print(f"\nPos-weight for BCEWithLogitsLoss: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────
NUM_EPOCHS = 20
print(f"\nStarting training for {NUM_EPOCHS} epochs...")

history = {'train_loss': [], 'val_loss': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    history['train_loss'].append(epoch_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(test_loader.dataset)
    history['val_loss'].append(val_loss)

    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
all_logits, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

all_probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
all_labels = np.array(all_labels, dtype=int)

# Classification threshold = 0.5
all_preds = (all_probs >= 0.5).astype(int)

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
roc_auc   = roc_auc_score(all_labels, all_probs)
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n{'='*40}")
print(f"{'TEST SET METRICS':^40}")
print(f"{'='*40}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")

# ─────────────────────────────────────────────
# 7. Save Model & Metrics
# ─────────────────────────────────────────────
SAVE_DIR = os.path.join(PROJ_DIR, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'lstm_model.pth'))
print(f"\nModel saved to saved_models/lstm_model.pth")

metrics = {
    'accuracy' : round(accuracy, 4),
    'precision': round(precision, 4),
    'recall'   : round(recall, 4),
    'f1'       : round(f1, 4),
    'roc_auc'  : round(roc_auc, 4),
    'confusion_matrix': cm.tolist(),
    'history'  : history,
}
with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to saved_models/metrics.json")
