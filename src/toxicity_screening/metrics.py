from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(p, bins[1:-1], right=True)
    ece = 0.0
    for bin_index in range(n_bins):
        selected = indices == bin_index
        if selected.any():
            ece += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return float(ece)


def recall_at_fixed_precision(y_true, probability, minimum_precision: float = 0.8) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probability)
    eligible = recall[precision >= minimum_precision]
    return float(eligible.max()) if eligible.size else 0.0


def precision_at_fixed_recall(y_true, probability, minimum_recall: float = 0.8) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probability)
    eligible = precision[recall >= minimum_recall]
    return float(eligible.max()) if eligible.size else 0.0


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float | int | list[list[int]]]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result: dict[str, float | int | list[list[int]]] = {
        "n": int(len(y)),
        "positive_prevalence": float(y.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "nll": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "recall_at_precision_0.80": recall_at_fixed_precision(y, p, 0.8),
        "precision_at_recall_0.80": precision_at_fixed_recall(y, p, 0.8),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return result


def select_threshold(y_true, probability, strategy: str, minimum_precision: float = 0.8) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], p]))
    if strategy == "maximize_mcc":
        return float(max(candidates, key=lambda t: matthews_corrcoef(y, p >= t)))
    if strategy == "maximize_recall_subject_to_precision":
        feasible = [t for t in candidates if precision_score(y, p >= t, zero_division=0) >= minimum_precision]
        return float(max(feasible, key=lambda t: recall_score(y, p >= t, zero_division=0))) if feasible else 0.5
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def bootstrap_confidence_interval(
    y_true,
    probability,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int = 2000,
    seed: int = 20260723,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    y = np.asarray(y_true)
    p = np.asarray(probability)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(float(metric(y[indices], p[indices])))
    if not values:
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan"), "valid_bootstraps": 0}
    return {
        "estimate": float(metric(y, p)),
        "lower": float(np.quantile(values, alpha / 2)),
        "upper": float(np.quantile(values, 1 - alpha / 2)),
        "valid_bootstraps": len(values),
    }
