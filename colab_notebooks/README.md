# Deep Learning for Managers — Google Colab Notebooks

Three self-contained Google Colab notebooks for the "Deep Learning for Managers" course, covering Indian business problems using ANN and LSTM (RNN) models.

> The original project scripts are in `../scratch/`. This folder contains Colab-adapted versions only — the scratch folder is untouched.

---

## Folder Structure

```
colab_notebooks/
├── README.md
├── 01_credit_default_prediction/
│   └── credit_default_prediction_ANN.ipynb
├── 02_upi_fraud_detection/
│   └── upi_fraud_detection_LSTM.ipynb
└── 03_food_delivery_sentiment/
    └── food_delivery_sentiment_LSTM.ipynb
```

---

## Project Summaries

### 01 — Credit Default Prediction (ANN)
| | |
|---|---|
| **Model** | Artificial Neural Network (MLP) |
| **Task** | 4-class credit risk classification: P1 (Best) → P4 (High Risk) |
| **Data** | Merged Indian bank internal data + CIBIL external credit scores |
| **Architecture** | Input → 128 → 64 → 32 → 4 classes (ReLU, BatchNorm, Dropout) |
| **Loss** | Weighted CrossEntropyLoss (handles class imbalance) |
| **Target Accuracy** | ~91% |

### 02 — UPI Fraud Detection (LSTM / RNN)
| | |
|---|---|
| **Model** | 2-layer stacked LSTM |
| **Task** | Binary classification — fraudulent vs. legitimate UPI session |
| **Data** | Synthetically generated sequences (80,000 sessions × 15 transactions × 12 features), calibrated to NPCI official monthly statistics |
| **Architecture** | LSTM(128, 2 layers) → Dropout → Linear(64) → Linear(1) |
| **Loss** | BCEWithLogitsLoss with positive class weighting |
| **Target Accuracy** | ~93% |

### 03 — Food Delivery Sentiment / Rating Prediction (LSTM + Embedding)
| | |
|---|---|
| **Model** | Embedding + 2-layer LSTM + numerical feature fusion |
| **Task** | 5-class restaurant rating: Poor / Average / Good / Very Good / Excellent |
| **Data** | Zomato India dataset (CSV + JSON, ~38,000 restaurants) |
| **Architecture** | Embedding(vocab,32) → LSTM(128, 2 layers) → concat(numeric) → Linear(128) → Linear(64) → Linear(5) |
| **Loss** | Weighted CrossEntropyLoss |
| **Target Accuracy** | ~77% |

---

## How to Run on Google Colab

### Step 1 — Upload the notebook
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click `File` → `Upload notebook`
3. Select the `.ipynb` file from the relevant folder

### Step 2 — Enable GPU (recommended)
1. Click `Runtime` → `Change runtime type`
2. Set **Hardware accelerator** to `T4 GPU`
3. Click **Save**

### Step 3 — Run all cells
1. Click `Runtime` → `Run all`
2. When the upload cell executes, a file picker will appear — upload the required data files (see table below)
3. All subsequent cells run automatically

---

## Data Files Required

| Notebook | Files to Upload | Source Location (in scratch/) |
|---|---|---|
| Credit Default (ANN) | `Internal_Bank_Dataset.xlsx` `External_Cibil_Dataset.xlsx` | `scratch/credit_default_prediction/data/` |
| UPI Fraud (LSTM) | `Product-Statistics-UPI-...-2021-22-monthly.xlsx` `Product-Statistics-UPI-...-2022-23-monthly.xlsx` `Product-Statistics-UPI-...-2023-24-monthly.xlsx` | `scratch/UPI_Fraud/` |
| Food Delivery (LSTM) | `zomato.csv` `file1.json` `file2.json` `file3.json` `file4.json` `file5.json` | `scratch/Food_Delivery/` |

> **Note for UPI Fraud:** The NPCI Excel files are optional. If you skip the upload, the notebook automatically falls back to a hardcoded average transaction value of Rs. 1,650 — the synthetic data generation still runs normally.

---

## Notebook Cell Structure

Each notebook follows the same 10-cell layout:

| Cell | Type | Content |
|---|---|---|
| 1 | Markdown | Project title, description, architecture overview |
| 2 | Code | `!pip install` all dependencies |
| 3 | Markdown | Data upload instructions |
| 4 | Code | `files.upload()` — interactive file picker |
| 5 | Markdown | Data preparation overview |
| 6 | Code | Full data preprocessing pipeline |
| 7 | Markdown | Model architecture description |
| 8 | Code | Model definition + training + evaluation |
| 9 | Markdown | Visualization descriptions |
| 10 | Code | All charts displayed inline |

---

## Colab Adaptations Made

The following changes were made from the original `scratch/` scripts to make the notebooks Colab-compatible:

- `%matplotlib inline` used instead of `matplotlib.use('Agg')`
- `plt.show()` added after every `plt.savefig()` for inline chart display
- GPU support added: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- No `__file__` references — all paths are relative to Colab's working directory (`/content/`)
- `torch.load(..., weights_only=True)` for PyTorch 2.x compatibility
- File uploads handled via `google.colab.files.upload()`
- All output directories (`data/processed/`, `saved_models/`, `visualizations/`) created with `os.makedirs(..., exist_ok=True)`

---

## Dependencies

All installed automatically in Cell 2 of each notebook:

```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```
