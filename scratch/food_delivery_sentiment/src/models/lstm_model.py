"""
Food Delivery Rating Prediction - LSTM (RNN) Model
Processes variable-length cuisine sequences via an Embedding + LSTM,
then combines with restaurant numerical features to predict
the rating category (Poor / Average / Good / Very Good / Excellent).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

print("PyTorch version:", torch.__version__)

# ─────────────────────────────────────────────
# 1. Load Processed Data
# ─────────────────────────────────────────────
PROJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')

print("Loading processed data...")
X_seq_train = np.load(os.path.join(PROC_DIR, 'X_seq_train.npy'))
X_seq_test  = np.load(os.path.join(PROC_DIR, 'X_seq_test.npy'))
X_num_train = np.load(os.path.join(PROC_DIR, 'X_num_train.npy'))
X_num_test  = np.load(os.path.join(PROC_DIR, 'X_num_test.npy'))
y_train     = np.load(os.path.join(PROC_DIR, 'y_train.npy'))
y_test      = np.load(os.path.join(PROC_DIR, 'y_test.npy'))

with open(os.path.join(PROC_DIR, 'metadata.json')) as f:
    meta = json.load(f)

VOCAB_SIZE   = meta['vocab_size']
MAX_SEQ_LEN  = meta['max_seq_len']
NUM_FEATURES = len(meta['num_features'])
NUM_CLASSES  = meta['num_classes']
CLASS_NAMES  = meta['class_names']

print(f"Train: {X_seq_train.shape[0]:,}  |  Test: {X_seq_test.shape[0]:,}")
print(f"Vocab size: {VOCAB_SIZE}  |  Seq len: {MAX_SEQ_LEN}  |  Num features: {NUM_FEATURES}")

# ─────────────────────────────────────────────
# 2. DataLoaders
# ─────────────────────────────────────────────
BATCH_SIZE = 128

def make_loader(X_seq, X_num, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X_seq, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(y,     dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_seq_train, X_num_train, y_train, shuffle=True)
test_loader  = make_loader(X_seq_test,  X_num_test,  y_test,  shuffle=False)

# ─────────────────────────────────────────────
# 3. LSTM Model Architecture
# ─────────────────────────────────────────────
class ZomatoLSTM(nn.Module):
    """
    Embedding → LSTM → concatenate with numerical features → FC classifier.

    Input A (cuisine sequence): (batch, seq_len)  dtype=long
    Input B (numeric features) : (batch, num_feat) dtype=float

    Processing:
      1. Embed cuisine tokens → (batch, seq_len, embed_dim)
      2. LSTM processes the sequence → take final hidden state (batch, hidden_dim)
      3. Concatenate with scaled numeric vector → (batch, hidden_dim + num_feat)
      4. FC head → (batch, num_classes)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_feat,
                 num_classes, num_lstm_layers=2, dropout=0.3):
        super(ZomatoLSTM, self).__init__()

        # Embedding layer (pad_idx=0 → zero gradients for padding)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # LSTM over cuisine token sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Classifier head: LSTM output + numeric features
        combined_dim = hidden_dim + num_feat
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq, x_num):
        # x_seq: (batch, seq_len)
        embedded = self.embedding(x_seq)        # (batch, seq_len, embed_dim)
        lstm_out, (h_n, _) = self.lstm(embedded)
        # Use the last layer's final hidden state
        last_hidden = h_n[-1]                   # (batch, hidden_dim)
        combined = torch.cat([last_hidden, x_num], dim=1)  # (batch, hidden+num_feat)
        return self.classifier(combined)         # (batch, num_classes)


model = ZomatoLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=32,
    hidden_dim=128,
    num_feat=NUM_FEATURES,
    num_classes=NUM_CLASSES,
    num_lstm_layers=2,
    dropout=0.3,
)
print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. Class Weights for Imbalance
# ─────────────────────────────────────────────
class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
total = class_counts.sum()
class_weights = torch.tensor(
    [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in class_counts],
    dtype=torch.float32
)
print(f"\nClass weights: {class_weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────
NUM_EPOCHS = 30
print(f"\nStarting training for {NUM_EPOCHS} epochs...")

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for X_s, X_n, y_b in train_loader:
        optimizer.zero_grad()
        outputs = model(X_s, X_n)
        loss    = criterion(outputs, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X_s.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # Validation
    model.eval()
    val_loss, correct, total_val = 0.0, 0, 0
    with torch.no_grad():
        for X_s, X_n, y_b in test_loader:
            out  = model(X_s, X_n)
            loss = criterion(out, y_b)
            val_loss += loss.item() * X_s.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total_val += y_b.size(0)

    val_loss /= len(test_loader.dataset)
    val_acc   = correct / total_val
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for X_s, X_n, y_b in test_loader:
        out   = model(X_s, X_n)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_b.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n{'='*45}")
print(f"{'TEST SET METRICS':^45}")
print(f"{'='*45}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"\nConfusion Matrix ({CLASS_NAMES}):")
print(cm)

# ─────────────────────────────────────────────
# 7. Save Model & Metrics
# ─────────────────────────────────────────────
SAVE_DIR = os.path.join(PROJ_DIR, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'lstm_model.pth'))
print(f"\nModel saved to saved_models/lstm_model.pth")

metrics = {
    'accuracy'        : round(accuracy, 4),
    'precision'       : round(precision, 4),
    'recall'          : round(recall, 4),
    'f1'              : round(f1, 4),
    'confusion_matrix': cm.tolist(),
    'class_names'     : CLASS_NAMES,
    'history'         : history,
    'vocab_size'      : VOCAB_SIZE,
}
with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to saved_models/metrics.json")
