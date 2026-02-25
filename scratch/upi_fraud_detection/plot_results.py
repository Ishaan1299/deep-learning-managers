"""
UPI Fraud Detection - Plot Results
Loads saved metrics and generates visualizations.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc

PROJ_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(PROJ_DIR, 'saved_models')
VIZ_DIR  = os.path.join(PROJ_DIR, 'visualizations')
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Load metrics ────────────────────────────────────────────────────────
with open(os.path.join(SAVE_DIR, 'metrics.json'), 'r') as f:
    metrics = json.load(f)

print("Loaded metrics:", {k: v for k, v in metrics.items() if k not in ('confusion_matrix', 'history')})

# ── 1. Training & Validation Loss Curve ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(1, len(metrics['history']['train_loss']) + 1)
ax.plot(epochs, metrics['history']['train_loss'], 'b-o', markersize=4, label='Training Loss')
ax.plot(epochs, metrics['history']['val_loss'],   'r-s', markersize=4, label='Validation Loss')
ax.set_title('LSTM Training & Validation Loss\nUPI Fraud Detection', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE Loss')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'loss_curve.png'), dpi=150)
plt.close()
print("Saved: loss_curve.png")

# ── 2. Confusion Matrix ──────────────────────────────────────────────────
cm = np.array(metrics['confusion_matrix'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Legitimate', 'Fraud'],
    yticklabels=['Legitimate', 'Fraud'],
    ax=ax
)
ax.set_title('Confusion Matrix — LSTM Fraud Classifier', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── 3. Metric Bar Chart ──────────────────────────────────────────────────
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metric_values = [metrics['accuracy'], metrics['precision'],
                 metrics['recall'],   metrics['f1'], metrics['roc_auc']]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metric_names, metric_values,
              color=['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336'],
              width=0.55, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.set_title('LSTM Fraud Detector — Test Set Performance Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'metric_bars.png'), dpi=150)
plt.close()
print("Saved: metric_bars.png")

# ── 4. ROC Curve (rebuild from saved data) ──────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reload model to regenerate probabilities
class FraudLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)

X_test  = np.load(os.path.join(PROC_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(PROC_DIR, 'y_test.npy'))

n_features = X_test.shape[2]
model = FraudLSTM(input_dim=n_features)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'lstm_model.pth'), weights_only=True))
model.eval()

X_te = torch.tensor(X_test, dtype=torch.float32)
y_te = torch.tensor(y_test, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_te, y_te), batch_size=512, shuffle=False)

all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in loader:
        logits = model(xb)
        probs  = torch.sigmoid(logits)
        all_probs.extend(probs.numpy())
        all_labels.extend(yb.numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc_val = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve — LSTM UPI Fraud Detector', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'roc_curve.png'), dpi=150)
plt.close()
print("Saved: roc_curve.png")

# ── 5. UPI Transaction Volume Growth (NPCI data) ────────────────────────
import pandas as pd
import sys
sys.path.insert(0, PROJ_DIR)

UPI_DIR = os.path.join(PROJ_DIR, '..', 'UPI_Fraud')
monthly_records = []
for yr_tag in ['2021-22', '2022-23', '2023-24']:
    fpath = os.path.join(UPI_DIR, f'Product-Statistics-UPI-Upi-monthly-statistics-{yr_tag}-monthly.xlsx')
    raw = pd.read_excel(fpath, header=None)
    raw.columns = ['Month', 'Volume_Mn', 'Avg_Daily_Volume_Mn', 'Value_Cr', 'Avg_Daily_Value_Cr']
    raw = raw.iloc[1:].dropna(subset=['Month']).reset_index(drop=True)
    monthly_records.append(raw)
monthly_df = pd.concat(monthly_records, ignore_index=True)
monthly_df['Volume_Mn'] = pd.to_numeric(monthly_df['Volume_Mn'], errors='coerce')
monthly_df['Value_Cr']  = pd.to_numeric(monthly_df['Value_Cr'],  errors='coerce')
monthly_df = monthly_df.dropna().reset_index(drop=True)

fig, ax1 = plt.subplots(figsize=(12, 5))
x = range(len(monthly_df))
ax1.bar(x, monthly_df['Volume_Mn'], color='steelblue', alpha=0.7, label='Volume (Mn txns)')
ax1.set_xlabel('Month (Apr 2021 → Mar 2024)')
ax1.set_ylabel('Transaction Volume (Millions)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(list(x)[::3])
ax1.set_xticklabels(monthly_df['Month'].iloc[::3].tolist(), rotation=45, ha='right', fontsize=8)

ax2 = ax1.twinx()
ax2.plot(x, monthly_df['Value_Cr'] / 1e5, 'r-o', markersize=4, label='Value (Lakh Cr.)')
ax2.set_ylabel('Transaction Value (Lakh Crores)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('NPCI UPI Transaction Growth (2021–2024)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'upi_growth_trend.png'), dpi=150)
plt.close()
print("Saved: upi_growth_trend.png")

print("\nAll visualizations saved to visualizations/")
