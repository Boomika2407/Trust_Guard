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


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
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
    return X, out["class"], out


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


    # Figure 9: XGBoost built-in importance (weight, gain, cover)
    xgb_model = models["XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, importance_type in zip(axes, ["weight", "gain", "cover"]):
        scores = xgb_model.get_booster().get_score(importance_type=importance_type)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        features, values = zip(*sorted_scores)
        ax.barh(features[::-1], values[::-1], color="#a855f7", alpha=0.8)
        ax.set_title(f"Importance: {importance_type.capitalize()}")
        ax.set_xlabel(importance_type.capitalize())
    fig.suptitle("XGBoost Built-in Feature Importance (Weight, Gain, Cover)", fontsize=13)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "feature_importance_xgb.png", dpi=150); plt.close(fig)


def save_xgb_importance_plot(models: dict) -> None:
    xgb_model = models["XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, importance_type in zip(axes, ["weight", "gain", "cover"]):
        scores = xgb_model.get_booster().get_score(importance_type=importance_type)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        features, values = zip(*sorted_scores)
        ax.barh(features[::-1], values[::-1], color="#a855f7", alpha=0.8)
        ax.set_title(f"Importance: {importance_type.capitalize()}")
        ax.set_xlabel(importance_type.capitalize())
    fig.suptitle("XGBoost Built-in Feature Importance (Weight, Gain, Cover)", fontsize=13)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "feature_importance_xgb.png", dpi=150); plt.close(fig)


def save_eda_plots(df: pd.DataFrame) -> None:
    # Figure 1: Class distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["class"].value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#22c55e", "#ef4444"], alpha=0.8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Class Distribution (Legitimate vs Fraud)")
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "class_distribution.png", dpi=150); plt.close(fig)

    # Figure 3: Fraud rate by hour
    fig, ax = plt.subplots(figsize=(10, 5))
    hour_stats = df.groupby("hour")["class"].mean() * 100
    ax.bar(hour_stats.index, hour_stats.values, color="#a855f7", alpha=0.8)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate (%) by Hour"); ax.set_xticks(range(24))
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "fraud_rate_by_hour.png", dpi=150); plt.close(fig)

    # Figure 4: Correlation of each feature with Class
    corr = df[config.FEATURE_COLS + ["class"]].corr()["class"].drop("class").sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, alpha=0.8)
    ax.set_xlabel("Correlation with Class"); ax.set_title("Feature Correlation with Fraud Class")
    ax.axvline(0, color="white", linewidth=0.8)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "feature_correlation.png", dpi=150); plt.close(fig)


def save_amount_by_class_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, label, color in [(0, "Legitimate", "#22c55e"), (1, "Fraud", "#ef4444")]:
        ax.hist(df[df["class"] == cls]["amount"] + 1, bins=60, alpha=0.7,
                label=label, color=color, log=True)
    ax.set_xlabel("Transaction Amount ($)"); ax.set_ylabel("Count (log scale)")
    ax.set_title("Transaction Amount by Class (log scale)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "amount_by_class.png", dpi=150); plt.close(fig)


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

    # Figure 7: Precision and recall vs threshold (XGBoost)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, best_prob)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr_arr, prec_arr[:-1], label="Precision", color="#a855f7")
    ax.plot(thr_arr, rec_arr[:-1], label="Recall", color="#22c55e")
    ax.axvline(thr, color="#ef4444", linestyle="--", label=f"Tuned threshold ({thr:.2f})")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs Decision Threshold (XGBoost)")
    ax.legend(); fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "precision_recall_threshold.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv", help="path to Kaggle creditcard.csv")
    args = ap.parse_args()

    config.MODELS_DIR.mkdir(exist_ok=True)
    df = load_and_clean(args.data)
    X, y, df_eng = engineer(df)
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
    save_xgb_importance_plot(models)
    save_amount_by_class_plot(df_eng)
    save_eda_plots(df_eng)
    print(f"\nSaved {config.MODEL_PATH.name}, {config.METRICS_PATH.name} and plots.")


if __name__ == "__main__":
    main()
