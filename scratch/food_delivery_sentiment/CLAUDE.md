# Food Delivery Sentiment / Rating Prediction (LSTM/RNN)

LSTM with Embedding layer that processes a restaurant's cuisine token sequence and operational features to predict its Zomato rating category (Poor / Average / Good / Very Good / Excellent).

## Files

| File | What | When to read |
|---|---|---|
| `data_prep.py` | Loads Zomato CSV (8,652 India restaurants) + 5 JSON API files (29,753 restaurants), filters "Not rated", tokenizes cuisine sequences, builds vocabulary (111 types), pads to 8 tokens, log-transforms votes and cost, encodes 5-class labels, saves `.npy` arrays and `metadata.json` | Modifying feature engineering, changing vocabulary size or sequence length, debugging data loading |
| `src/models/lstm_model.py` | Defines `ZomatoLSTM` (Embedding→2-layer LSTM→concat with numeric features→classifier head), trains 30 epochs with weighted CrossEntropyLoss, evaluates 5-class Accuracy/Precision/Recall/F1, saves model and `metrics.json` | Modifying embedding size or LSTM layers, retraining, reviewing per-class performance |
| `plot_results.py` | Loads `metrics.json` and saved model; generates loss+accuracy curves, count and normalised confusion matrices, metric bars, per-class precision/recall/F1, and rating distribution pie chart | Regenerating or adding visualizations |
| `implementation_plan.md.resolved` | Technical architecture plan: data fusion, tokenization, Embedding+LSTM design, class weighting | Reviewing design decisions, referencing model specifications |
| `walkthrough.md.resolved` | Full 25-page academic report: Zomato ecosystem context, cuisine sequence rationale, LSTM methodology, results, restaurant operations and platform insights | Writing or reviewing the project report |
| `data/` | Raw Zomato source files reference and generated `.npy` feature arrays | Tracing Zomato data, debugging input shapes |
| `saved_models/` | Trained LSTM weights `lstm_model.pth` and `metrics.json` | Loading model, checking saved performance numbers |
| `visualizations/` | 5 generated PNG charts (loss curve, both confusion matrices, metric bars, per-class metrics, rating distribution) | Viewing or including charts in the report |

## Commands

```bash
# Full pipeline (run in order from project root)
python data_prep.py
python src/models/lstm_model.py
python plot_results.py
```

## README.md

See `README.md` for Embedding+LSTM design decisions, data fusion rationale, and sequence construction logic.
