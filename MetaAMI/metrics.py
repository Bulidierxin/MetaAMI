from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, jaccard_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def choose_threshold(y_true, y_prob, strategy="fixed", fixed_threshold=0.14):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if strategy == "fixed":
        return float(fixed_threshold)
    grid = np.round(np.arange(0.05, 0.951, 0.005), 3)
    if strategy == "f1":
        scores = [f1_score(y_true, y_prob >= th, zero_division=0) for th in grid]
        return float(grid[int(np.argmax(scores))])
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return float(thresholds[int(np.argmax(tpr - fpr))])
    raise ValueError("threshold_strategy must be one of: fixed, f1, youden")


def binary_metrics(y_true, y_prob, threshold=0.14) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sens,
        "specificity": spec,
        "youden_j": sens + spec - 1,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
        "g_mean": float(np.sqrt(sens * spec)),
        "jaccard": jaccard_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def summarize_metrics(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [c for c in df.columns if c not in {"fold", "repeat", "model", "rp", "group"}]
    pm = {"fold": "mean±std"}
    for c in numeric:
        if pd.api.types.is_numeric_dtype(df[c]):
            pm[c] = f"{df[c].mean():.3f} ± {df[c].std(ddof=1):.3f}"
    return pd.concat([df, pd.DataFrame([pm])], ignore_index=True)
