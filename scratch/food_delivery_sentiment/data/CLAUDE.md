# food_delivery_sentiment/data

Source data reference and generated processed feature arrays for the LSTM rating predictor.

## Directories

| Directory | What | When to read |
|---|---|---|
| `raw/` | Empty placeholder; actual Zomato source files live in `../Food_Delivery/` (sibling to project root) | Locating Zomato CSV and JSON source files |
| `processed/` | Generated `.npy` arrays and `metadata.json` output by `data_prep.py`: `X_seq_train/test` (cuisine token sequences, shape N×8), `X_num_train/test` (5 scaled numeric features), `y_train/test` (class labels 0–4), `metadata.json` (vocab size, class names, feature list) — do not edit manually | Debugging input shapes, confirming vocabulary size, checking class label mapping |
