import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 1. Define the exact same architecture to load the saved weights
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

def plot_confusion_matrix():
    print("Loading test data...")
    X_test_df = pd.read_csv('data/processed/X_test.csv')
    y_test_df = pd.read_csv('data/processed/y_test.csv')

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_df.values.ravel(), dtype=torch.long)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Loading trained model...")
    input_dimension = X_test.shape[1]
    num_classes = len(torch.unique(y_test))
    
    model = CreditRiskANN(input_dimension, num_classes)
    model.load_state_dict(torch.load('saved_models/ann_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # Target mapping was: 0=P1, 1=P2, 2=P3, 3=P4
    labels = ['P1 (Best)', 'P2 (Good)', 'P3 (Moderate)', 'P4 (High Risk)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Neural Network Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('Actual True Risk Level', fontsize=14)
    plt.xlabel('Predicted Risk Level', fontsize=14)
    
    # Make sure text fits and layout is tight
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('visualizations', exist_ok=True)
    save_path = 'visualizations/confusion_matrix.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Confusion Matrix saved to: {save_path}")
    
    # Also just show it if running interactively
    # plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()
