# upi_fraud_detection/data

Source data reference and generated processed sequence arrays for the LSTM fraud detector.

## Directories

| Directory | What | When to read |
|---|---|---|
| `raw/` | Empty placeholder; actual NPCI source files live in `../UPI_Fraud/` (sibling to project root) | Locating NPCI Excel source files |
| `processed/` | Generated `.npy` arrays output by `data_prep.py`: `X_train`, `X_test` (shape: N×15×12), `y_train`, `y_test` — do not edit manually | Debugging sequence shapes, confirming data pipeline ran successfully |
