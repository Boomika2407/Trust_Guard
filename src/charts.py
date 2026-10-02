"""Matplotlib figures styled for the dark FraudShield theme."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

BG = "#0b1525"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN, RED, BLUE = "#22c55e", "#ef4444", "#3b82f6"


def _style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)


def probability_figure(probability: float):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    _style(ax, fig)
    values = [1 - probability, probability]
    bars = ax.barh(["Legitimate", "Fraudulent"], values, color=[GREEN, RED], height=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2%}",
                va="center", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probability", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def contribution_figure(contribs: pd.Series, feature_values: pd.Series, top_n: int = 15):
    """Horizontal bar chart of the largest SHAP contributions for one transaction."""
    top = contribs.reindex(contribs.abs().sort_values().tail(top_n).index)
    labels = [f"{name} = {feature_values[name]:.2f}" for name in top.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    _style(ax, fig)
    ax.barh(labels, top.values, color=[RED if v > 0 else GREEN for v in top.values], height=0.6)
    ax.axvline(0, color="#334155", linewidth=1.2)
    ax.set_title(f"Top {top_n} feature contributions (SHAP, log-odds)", color=TEXT, fontsize=11, pad=12)
    ax.set_xlabel("Red pushes toward fraud, green pushes toward legitimate", color=MUTED, fontsize=8)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def distribution_figure(legit: int, fraud: int):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    _style(ax, fig)
    ax.bar(["Legitimate", "Fraudulent"], [legit, fraud], color=[GREEN, RED], width=0.45)
    ax.set_title("Prediction distribution", color=TEXT, fontsize=10, pad=10)
    ax.set_ylabel("Count", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig
