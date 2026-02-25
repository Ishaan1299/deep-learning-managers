"""
Food Delivery Sentiment - Plot Results
Generates all visualizations for the report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJ_DIR, 'saved_models')
VIZ_DIR  = os.path.join(PROJ_DIR, 'visualizations')
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
FOOD_DIR = os.path.join(PROJ_DIR, '..', 'Food_Delivery')
os.makedirs(VIZ_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, 'metrics.json')) as f:
    metrics = json.load(f)

CLASS_NAMES = metrics['class_names']
print("Loaded metrics:", {k: v for k, v in metrics.items()
                          if k not in ('confusion_matrix', 'history', 'class_names')})

# ── 1. Training & Validation Loss + Accuracy Curves ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

epochs = range(1, len(metrics['history']['train_loss']) + 1)

ax1.plot(epochs, metrics['history']['train_loss'], 'b-o', markersize=3, label='Train Loss')
ax1.plot(epochs, metrics['history']['val_loss'],   'r-s', markersize=3, label='Val Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, [v * 100 for v in metrics['history']['val_acc']],
         'g-^', markersize=3, label='Val Accuracy')
ax2.axhline(metrics['accuracy'] * 100, color='orange', linestyle='--',
            label=f"Final Test Acc: {metrics['accuracy']*100:.1f}%")
ax2.set_title('Validation Accuracy per Epoch', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('LSTM Cuisine-Sequence Model — Training Curves\nFood Delivery Rating Prediction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'loss_curve.png'), dpi=150)
plt.close()
print("Saved: loss_curve.png")

# ── 2. Confusion Matrix ──────────────────────────────────────────────────
cm = np.array(metrics['confusion_matrix'])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalised)', fontweight='bold')
axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')

plt.suptitle('LSTM Restaurant Rating Classifier — Confusion Matrices',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── 3. Overall Metric Bar Chart ──────────────────────────────────────────
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_values = [metrics['accuracy'], metrics['precision'],
                 metrics['recall'],   metrics['f1']]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metric_names, metric_values, color=colors, width=0.5,
              edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.set_title('LSTM Rating Predictor — Test Set Performance\nFood Delivery (Zomato)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Score')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'metric_bars.png'), dpi=150)
plt.close()
print("Saved: metric_bars.png")

# ── 4. Per-Class Precision & Recall ─────────────────────────────────────
from sklearn.metrics import classification_report
y_test  = np.load(os.path.join(PROC_DIR, 'y_test.npy'))

# Reload predictions using saved model
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ZomatoLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_feat,
                 num_classes, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_lstm_layers,
                            batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        combined_dim = hidden_dim + num_feat
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x_seq, x_num):
        emb = self.embedding(x_seq)
        _, (h_n, _) = self.lstm(emb)
        return self.classifier(torch.cat([h_n[-1], x_num], dim=1))

with open(os.path.join(PROC_DIR, 'metadata.json')) as f:
    meta = json.load(f)

mdl = ZomatoLSTM(meta['vocab_size'], 32, 128, len(meta['num_features']), 5)
mdl.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'lstm_model.pth'), weights_only=True))
mdl.eval()

X_s = torch.tensor(np.load(os.path.join(PROC_DIR, 'X_seq_test.npy')),  dtype=torch.long)
X_n = torch.tensor(np.load(os.path.join(PROC_DIR, 'X_num_test.npy')),  dtype=torch.float32)
y_t = torch.tensor(y_test, dtype=torch.long)

loader = DataLoader(TensorDataset(X_s, X_n, y_t), batch_size=256, shuffle=False)
preds_all = []
with torch.no_grad():
    for xs, xn, _ in loader:
        preds_all.extend(mdl(xs, xn).argmax(dim=1).numpy())

preds_all = np.array(preds_all)

from sklearn.metrics import precision_score, recall_score, f1_score
prec_per = precision_score(y_test, preds_all, average=None, zero_division=0)
rec_per  = recall_score(y_test, preds_all, average=None, zero_division=0)
f1_per   = f1_score(y_test, preds_all, average=None, zero_division=0)

x = np.arange(len(CLASS_NAMES))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w,   prec_per, w, label='Precision', color='steelblue')
ax.bar(x,       rec_per,  w, label='Recall',    color='darkorange')
ax.bar(x + w,   f1_per,   w, label='F1-Score',  color='green')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class Precision, Recall & F1 — LSTM Zomato Rating Predictor',
             fontweight='bold')
ax.set_ylabel('Score'); ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'per_class_metrics.png'), dpi=150)
plt.close()
print("Saved: per_class_metrics.png")

# ── 5. Rating Distribution in Dataset (EDA) ─────────────────────────────
import json as _json
label_counts = np.bincount(np.load(os.path.join(PROC_DIR, 'y_train.npy')), minlength=5) + \
               np.bincount(y_test, minlength=5)

colors_pie = ['#e74c3c','#e67e22','#3498db','#2ecc71','#9b59b6']
fig, ax = plt.subplots(figsize=(7, 6))
wedges, texts, autotexts = ax.pie(
    label_counts, labels=CLASS_NAMES, autopct='%1.1f%%',
    colors=colors_pie, startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Zomato Restaurant Rating Distribution\n(India, 2019-2023)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'rating_distribution.png'), dpi=150)
plt.close()
print("Saved: rating_distribution.png")

print("\nAll visualizations saved to visualizations/")
