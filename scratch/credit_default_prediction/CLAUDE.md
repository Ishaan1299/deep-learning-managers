# Credit Default Prediction (ANN)

ANN-based 4-class credit risk classifier using merged Indian bank and CIBIL data. Predicts risk tier P1 (low) to P4 (high default risk).

## Files

| File | What | When to read |
|---|---|---|
| `data_prep.py` | Merges Internal Bank + External CIBIL Excel files, handles -99999 missing values, encodes categoricals, scales features, outputs train/test CSVs | Modifying preprocessing, adding features, debugging data issues |
| `src/models/ann_model.py` | ANN architecture (3 hidden layers: 128→64→32), training loop, weighted CrossEntropyLoss, saves model weights and metrics | Modifying model architecture, retraining, reviewing metrics |
| `plot_results.py` | Loads saved model and test data, generates confusion matrix PNG | Regenerating visualizations after model changes |
| `app.py` | Flask REST API serving the trained ANN at `/predict`; accepts JSON sliders, returns risk tier probabilities | Running or extending the live inference API |
| `index.html` | Interactive browser demo UI (Tailwind + Chart.js); calls `app.py` at `localhost:5000/predict` | Running or modifying the live demo front-end |
| `beginner_guide.md` | Non-technical layperson explanation of data prep and ANN training steps | Preparing manager-facing presentation or non-technical Q&A |
| `implementation_plan.md.resolved` | Technical architecture plan: data pipeline, model layers, evaluation strategy | Understanding design decisions, referencing model specs |
| `walkthrough.md.resolved` | Full 25-page academic report: problem statement, methodology, results, managerial insights | Writing or reviewing the project report |
| `data/` | Raw Excel source files and generated train/test CSV splits | Sourcing data, tracing preprocessing inputs |
| `saved_models/` | Trained ANN weights `ann_model.pth` | Loading model for inference via `app.py` |
| `visualizations/` | Generated confusion matrix PNG | Viewing or including in report/presentation |

## Commands

```bash
# Full pipeline (run in order)
python data_prep.py
python src/models/ann_model.py
python plot_results.py

# Start live demo API (requires flask, flask-cors)
python app.py
# Then open index.html in a browser
```

## README.md

See `README.md` for ANN design decisions and invisible architectural context.
