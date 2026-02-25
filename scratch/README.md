# Deep Learning for Managers — Project Architecture

Three independent deep learning projects, each targeting a different Indian business problem and using a different model type as required by the course.

## Project Overview

| Project | Model | Task | Test Accuracy |
|---|---|---|---|
| `credit_default_prediction/` | ANN (MLP) | 4-class credit risk tier (P1–P4) | ~91% |
| `upi_fraud_detection/` | LSTM (RNN) | Binary fraud vs. legitimate session | ~93% |
| `food_delivery_sentiment/` | LSTM + Embedding (RNN) | 5-class restaurant rating (Poor→Excellent) | ~77% |

## Model Selection Rationale

**ANN for Credit Default:** The input is a flat vector of 93 engineered customer features (CIBIL + bank data). There is no sequential or temporal relationship between features — each feature is an independent measurement of the customer at a point in time. An ANN (Multi-Layer Perceptron) is the correct architecture: it learns non-linear interactions across all 93 features simultaneously without imposing any order.

**LSTM for UPI Fraud:** Fraud follows a temporal behavioral pattern — account takeover attacks have a characteristic arc: (1) normal baseline activity → (2) device change → (3) velocity spike → (4) large transfers to new payees. This sequential dependency across time steps is exactly what LSTM hidden states capture. A flat ANN processing a single transaction would miss the contextual escalation.

**LSTM + Embedding for Food Delivery:** A restaurant's cuisine list is a variable-length sequence of tokens (e.g., `['North Indian', 'Chinese', 'Fast Food']`). The ordering and combination of cuisines signals market positioning (focused quality vs. generalist). An Embedding layer learns dense cuisine representations; the LSTM processes the token sequence to capture cuisine combination patterns. A plain ANN would require sparse one-hot encoding and lose positional context.

## Shared Project Structure

Each project follows the same layout:
```
<project>/
├── data_prep.py              # Data loading, cleaning, feature engineering
├── src/models/lstm_model.py  # Model definition + training + evaluation
├── plot_results.py           # Visualization generation
├── data/
│   ├── raw/                  # (empty placeholder for raw data symlinks)
│   └── processed/            # Generated .npy / .csv splits (gitignored)
├── saved_models/             # Trained model weights + metrics.json
├── visualizations/           # Generated PNG charts
├── implementation_plan.md.resolved   # Technical architecture plan
└── walkthrough.md.resolved           # Full academic report (25–30 pages)
```

## Running Order

For each project, always run scripts in this order:
1. `python data_prep.py` — generates `data/processed/`
2. `python src/models/lstm_model.py` — trains model, saves to `saved_models/`
3. `python plot_results.py` — reads `saved_models/metrics.json`, writes `visualizations/`
