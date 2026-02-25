# food_delivery_sentiment/src/models

LSTM model implementation for 5-class Zomato restaurant rating prediction.

## Files

| File | What | When to read |
|---|---|---|
| `lstm_model.py` | Defines `ZomatoLSTM` (Embedding(111,32,padding_idx=0) → 2-layer LSTM(128) → concat with 5 numeric features → Linear(133→128)→BatchNorm→ReLU→Linear(128→64)→BatchNorm→ReLU→Linear(64→5)); trains 30 epochs with weighted CrossEntropyLoss + gradient clipping, evaluates weighted Accuracy/Precision/Recall/F1 and confusion matrix, saves weights and `metrics.json` | Modifying embedding dim, LSTM hidden size, or classifier depth; changing learning rate or epochs; retraining the model |
