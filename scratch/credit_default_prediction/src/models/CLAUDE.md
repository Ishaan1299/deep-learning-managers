# credit_default_prediction/src/models

ANN model implementation for credit default risk classification.

## Files

| File | What | When to read |
|---|---|---|
| `ann_model.py` | Defines `CreditRiskANN` (3 hidden layers: 128→64→32, ReLU, BatchNorm, Dropout); loads processed CSVs, trains for 20 epochs with weighted CrossEntropyLoss, evaluates on test set, saves weights to `saved_models/ann_model.pth` | Modifying architecture, changing hyperparameters, retraining the model |
