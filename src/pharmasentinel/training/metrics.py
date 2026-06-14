"""
Evaluation metrics for all four PharmaSentinel tasks.

Includes:
  - Per-task metric computation
  - Fairness metrics (equalized odds across conditions)
  - Calibration metrics (ECE, reliability diagram data)
  - Aggregate benchmark table builder
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr


# ── Per-task metrics ──────────────────────────────────────────────────────────

def sentiment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "f1_macro":   f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa":      cohen_kappa_score(y_true, y_pred),
    }


def condition_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "f1_macro":   f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {
        "mae":         mean_absolute_error(y_true, y_pred),
        "rmse":        float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "pearson_r":   float(pearson_r),
        "spearman_r":  float(spearman_r),
    }


def helpfulness_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return rating_metrics(y_true, y_pred)


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for a binary or multi-class classifier.

    y_prob should be the probability of the predicted class (max prob per sample).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    confidence = y_prob.max(axis=-1) if y_prob.ndim > 1 else y_prob
    predicted  = y_prob.argmax(axis=-1) if y_prob.ndim > 1 else (y_prob >= 0.5).astype(int)
    correct    = (predicted == y_true).astype(float)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        avg_confidence = confidence[mask].mean()
        avg_accuracy   = correct[mask].mean()
        ece += (mask.sum() / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


# ── Fairness ──────────────────────────────────────────────────────────────────

def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    positive_label: int = 2,
) -> Dict[str, float]:
    """
    Compute the Equalized Odds gap across condition groups.

    Returns the max TPR and FPR difference between any pair of groups,
    which is the standard fairness metric used in algorithmic auditing.
    """
    unique_groups = np.unique(groups)
    tprs, fprs = [], []

    for g in unique_groups:
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:
            continue
        tp = ((yt == positive_label) & (yp == positive_label)).sum()
        fn = ((yt == positive_label) & (yp != positive_label)).sum()
        fp = ((yt != positive_label) & (yp == positive_label)).sum()
        tn = ((yt != positive_label) & (yp != positive_label)).sum()
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))

    return {
        "max_tpr_gap": float(max(tprs) - min(tprs)) if tprs else 0.0,
        "max_fpr_gap": float(max(fprs) - min(fprs)) if fprs else 0.0,
        "n_groups":    int(len(tprs)),
    }


# ── Benchmark table builder ───────────────────────────────────────────────────

def build_benchmark_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format a dict-of-dicts into a Markdown benchmark table.

    Example input:
        {
          "TF-IDF + LR": {"accuracy": 0.82, "f1_macro": 0.79},
          "DistilBERT":  {"accuracy": 0.88, "f1_macro": 0.86},
        }
    """
    if not results:
        return ""

    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    header = "| Model | " + " | ".join(m.upper() for m in metrics) + " |"
    divider = "|---" * (len(metrics) + 1) + "|"
    rows = []
    for model in models:
        vals = " | ".join(f"{results[model].get(m, 0):.4f}" for m in metrics)
        rows.append(f"| {model} | {vals} |")

    return "\n".join([header, divider] + rows)
