# upi_fraud_detection/src/models

LSTM model implementation for binary UPI fraud session classification.

## Files

| File | What | When to read |
|---|---|---|
| `lstm_model.py` | Defines `FraudLSTM` (2-layer LSTM 128 hidden units + Dropout→Linear→BatchNorm→ReLU→Linear classifier); loads `.npy` sequences, trains 20 epochs with BCEWithLogitsLoss + pos_weight + gradient clipping, evaluates Accuracy/Precision/Recall/F1/ROC-AUC, saves weights and `metrics.json` | Modifying LSTM depth or hidden size, changing epochs or learning rate, retraining the model |
