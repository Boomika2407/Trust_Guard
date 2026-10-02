"""Train the fraud model from the Kaggle creditcard.csv (script version of the notebook).

Usage:
    python train.py --data data/creditcard.csv

Steps mirror the notebook: clean -> engineer features -> stratified split ->
benchmark 4 models -> pick best by ROC-AUC -> tune threshold on the PR curve ->
save models/fraud_pipeline.pkl, models/metrics.json and evaluation plots.
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config

SEED = 42


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {dupes:,} duplicate rows -> {len(df):,} remain")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Add hour, scale amount/time with the constants the app uses, drop raw columns."""
    out = df.copy()
    out["hour"] = (out["time"] // 3600) % 24
    out["amount_scaled"] = StandardScaler().fit_transform(out[["amount"]]).ravel()
    time_scaler = StandardScaler().fit(out[["time"]])
    out["time_scaled"] = time_scaler.transform(out[["time"]]).ravel()
    joblib.dump(time_scaler, config.MODELS_DIR / "time_scaler.pkl")
    print(f"Time mean/std: {time_scaler.mean_[0]:.6f} / {time_scaler.scale_[0]:.6f}")
    print(f"Amount mean/std: {out['amount'].mean():.6f} / {out['amount'].std(ddof=0):.6f}")
    print("If these differ from src/config.py, update TIME_*/AMOUNT_* so the app scales identically.")
    X = out[config.FEATURE_COLS]
    return X, out["class"]


def benchmark(X_train, X_test, y_train, y_test):
    neg, pos = np.bincount(y_train)
    scale = neg / pos
    print(f"scale_pos_weight = {scale:.2f}")
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100,
                                                n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(scale_pos_weight=scale, n_jobs=-1, random_state=SEED,
                                 eval_metric="logloss"),
    }
    rows, probs = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        probs[name] = p
        rows.append({"Model": name,
                     "F1 (Fraud)": round(f1_score(y_test, model.predict(X_test)), 4),
                     "ROC-AUC": round(roc_auc_score(y_test, p), 4)})
        print(f"\n{'=' * 40}\n{name}\n{classification_report(y_test, model.predict(X_test))}")
    table = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print("\n", table.to_string(index=False))
    return models, probs, table


def tune_threshold(y_test, prob) -> dict:
    prec, rec, thr = precision_recall_curve(y_test, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(f1.argmax())
    return {"threshold": float(thr[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "f1": float(f1[i])}


def save_plots(models, probs, y_test, best_prob, thr):
    config.PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, p):.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(loc="lower right")
    plt.title("ROC curves"); plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "roc_comparison.png", dpi=150); plt.close()

    cm = confusion_matrix(y_test, (best_prob >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (tuned threshold)")
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv", help="path to Kaggle creditcard.csv")
    args = ap.parse_args()

    config.MODELS_DIR.mkdir(exist_ok=True)
    df = load_and_clean(args.data)
    X, y = engineer(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    models, probs, table = benchmark(X_train, X_test, y_train, y_test)
    best_name = table.iloc[0]["Model"]
    best_prob = probs[best_name]
    tuned = tune_threshold(y_test, best_prob)
    print(f"\nBest model: {best_name}")
    print(f"Tuned threshold {tuned['threshold']:.4f} | precision {tuned['precision']:.4f} | "
          f"recall {tuned['recall']:.4f} | F1 {tuned['f1']:.4f}")

    pipeline = Pipeline([("model", models[best_name])])
    joblib.dump(pipeline, config.MODEL_PATH)

    metrics = {**tuned, "roc_auc": float(roc_auc_score(y_test, best_prob)),
               "fraud_rate": round(float(y.mean() * 100), 3),
               "n_transactions": int(len(df)), "best_model": best_name,
               "comparison": table.to_dict("records")}
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    save_plots(models, probs, y_test, best_prob, tuned["threshold"])
    print(f"\nSaved {config.MODEL_PATH.name}, {config.METRICS_PATH.name} and plots.")


if __name__ == "__main__":
    main()
