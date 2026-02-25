# UPI Fraud Detection (LSTM/RNN)

2-layer stacked LSTM that reads sequences of 15 UPI transactions per user and classifies sessions as fraudulent or legitimate. Data is synthetically generated and calibrated to NPCI official monthly statistics.

## Files

| File | What | When to read |
|---|---|---|
| `data_prep.py` | Loads NPCI Excel files, calibrates synthetic data parameters, generates 80K user sessions (15 steps × 12 features), applies label noise, scales and saves `.npy` arrays | Modifying data generation, changing sequence length or fraud rate, debugging feature shapes |
| `src/models/lstm_model.py` | Defines `FraudLSTM` (2-layer LSTM 128 units + classifier head), trains with BCEWithLogitsLoss + pos_weight, evaluates with Accuracy/Precision/Recall/F1/ROC-AUC, saves model and `metrics.json` | Modifying LSTM architecture, retraining, reviewing fraud detection metrics |
| `plot_results.py` | Loads `metrics.json` and saved model; generates loss curve, confusion matrix, metric bars, ROC curve, and NPCI UPI growth trend chart | Regenerating or adding visualizations |
| `implementation_plan.md.resolved` | Technical architecture plan: sequence design, LSTM layers, class weighting strategy | Reviewing design decisions, referencing model specifications |
| `walkthrough.md.resolved` | Full 25-page academic report: UPI ecosystem context, ATO fraud pattern, LSTM methodology, results, managerial policy recommendations | Writing or reviewing the project report |
| `data/` | Raw NPCI source files reference and generated `.npy` sequence arrays | Tracing NPCI data, debugging input shapes |
| `saved_models/` | Trained LSTM weights `lstm_model.pth` and `metrics.json` | Loading model, checking saved performance numbers |
| `visualizations/` | 5 generated PNG charts (loss curve, confusion matrix, metric bars, ROC curve, NPCI growth) | Viewing or including charts in the report |

## Commands

```bash
# Full pipeline (run in order from project root)
python data_prep.py
python src/models/lstm_model.py
python plot_results.py
```

## README.md

See `README.md` for LSTM design decisions, synthetic data rationale, and sequence construction logic.
