# credit_default_prediction/data

Raw source Excel files and generated preprocessed train/test splits for the ANN credit risk model.

## Files

| File | What | When to read |
|---|---|---|
| `Internal_Bank_Dataset.xlsx` | Internal bank records: trade lines, loan history, repayment behavior per customer (PROSPECTID key) | Tracing raw feature origins, investigating preprocessing |
| `External_Cibil_Dataset.xlsx` | External CIBIL bureau data: delinquency history, credit utilization, target `Approved_Flag` (P1–P4) | Tracing raw feature origins, checking target variable distribution |
| `Unseen_Dataset.xlsx` | Hold-out set with no labels — used for final blind inference demo | Running predictions on truly unseen customers |
| `processed/` | Generated train/test CSVs output by `data_prep.py` (do not edit manually) | Debugging shape/feature mismatches before model training |
