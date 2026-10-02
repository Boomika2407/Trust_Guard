# 🛡️ TrustGuard: Credit Card Fraud Detection

End-to-end fraud detection on the Kaggle credit card dataset (284,807 transactions, 0.17% fraud).
XGBoost with `scale_pos_weight`, precision-recall threshold tuning, and SHAP explanations,
served through a Streamlit web app. Runs entirely in VS Code, with no Jupyter notebook needed.

Based on [profpius/credit-card-fraud-detection](https://github.com/profpius/credit-card-fraud-detection) (MIT).

## Results (held-out test set)

| Model | F1 (Fraud) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.1042 | 0.9686 |
| Decision Tree | 0.3584 | 0.8886 |
| Random Forest | 0.8171 | 0.9193 |
| **XGBoost (selected)** | **0.8506** | **0.9725** |

Tuned threshold: precision 0.9383, recall 0.80, F1 0.8636.

## Project structure

```
fraudshield/
├── app.py                  # Streamlit UI (run this)
├── train.py                # Retrain the model from creditcard.csv
├── requirements.txt
├── src/
│   ├── config.py           # Paths, feature schema, scaling constants
│   ├── features.py         # Raw columns -> 31 model features
│   ├── predictor.py        # Model loading, scoring, SHAP contributions
│   ├── rules.py            # v14 / v4 triage rules (block / review / allow)
│   ├── charts.py           # Dark-themed matplotlib figures
│   └── samples.py          # Known fraud / legit transactions
├── models/                 # fraud_pipeline.pkl, time_scaler.pkl
├── plots/                  # SHAP and importance images
├── assets/style.css        # UI theme
├── data/                   # Put creditcard.csv here (not included)
└── .vscode/                # Debug configs for VS Code
```

## Run in VS Code

1. Install Python 3.10+ and VS Code with the **Python** extension.
2. **File > Open Folder** and choose `fraudshield`.
3. Open a terminal (**Ctrl+`**) and run:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. You can also press **F5** and pick *Streamlit: run app.py*.

## Using the app

- **Single Transaction**: type in Time, Amount and V1-V28, or load a known fraud or legit sample, then
  click *Analyze*. You get a verdict, probability, and a SHAP chart of the top drivers.
- **Batch CSV Upload**: upload a Kaggle-format CSV (`Time, V1..V28, Amount`, optional `Class`).
  Download the scored results. If `Class` is present, TP / FN / FP and precision / recall are shown.
- **Model Insights**: model comparison, SHAP plots and key findings.
- **Sidebar**: adjust the decision threshold and toggle the triage rules
  (auto-block when `v14 < -5` and `v4 > 5`, manual review if only one holds).

## Retrain (optional)

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
save it as `data/creditcard.csv`, then run:

```bash
python train.py --data data/creditcard.csv
```

This reproduces the notebook pipeline: de-duplicate, engineer `hour`, `amount_scaled` and `time_scaled`,
stratified 80/20 split, benchmark four models, select by ROC-AUC, tune the threshold on the PR curve,
and save `models/fraud_pipeline.pkl`, `models/metrics.json` and evaluation plots. The app picks up
the new threshold and metrics automatically.

## Notes

- Batch files must come from the same Kaggle dataset; V1-V28 are PCA components specific to it.
- Prediction at the default 0.5 threshold matches the original app. Lower the slider to favour recall.
- For portfolio and academic use only, not for production financial decisions.
