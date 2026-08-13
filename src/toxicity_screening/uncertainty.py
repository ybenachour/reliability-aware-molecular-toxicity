from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def ensemble_uncertainty(probability_matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(probability_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected shape (n_models, n_samples)")
    mean = matrix.mean(axis=0)
    standard_deviation = matrix.std(axis=0, ddof=1 if matrix.shape[0] > 1 else 0)
    entropy = -(mean * np.log(mean + 1e-8) + (1 - mean) * np.log(1 - mean + 1e-8))
    return {"mean": mean, "standard_deviation": standard_deviation, "predictive_entropy": entropy}


def model_ensemble_probabilities(models: Sequence[object], x: np.ndarray) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for model in models:
        if hasattr(model, "predict_proba"):
            predictions.append(np.asarray(model.predict_proba(x))[:, 1])
        elif callable(model):
            predictions.append(np.asarray(model(x), dtype=float))
        else:
            raise TypeError(f"Unsupported ensemble member: {type(model)}")
    return np.vstack(predictions)


def monte_carlo_dropout(
    predict_once: Callable[[], np.ndarray],
    passes: int = 30,
) -> dict[str, np.ndarray]:
    samples = np.vstack([np.asarray(predict_once(), dtype=float) for _ in range(passes)])
    return ensemble_uncertainty(samples)


class SplitConformalBinaryClassifier:
    """Label-conditional split conformal prediction sets from calibrated probabilities.

    Nonconformity is 1 - P(true class). The prediction set contains each class whose
    nonconformity does not exceed the class-specific calibration quantile.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.quantiles_: dict[int, float] = {}

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "SplitConformalBinaryClassifier":
        p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
        y = np.asarray(labels, dtype=int)
        if p.shape[0] != y.shape[0]:
            raise ValueError("probability and labels must align")
        for label in (0, 1):
            selected = y == label
            if not selected.any():
                raise ValueError(f"Calibration data contain no examples for class {label}")
            true_probability = p[selected] if label == 1 else 1.0 - p[selected]
            scores = 1.0 - true_probability
            n = len(scores)
            quantile_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.quantiles_[label] = float(np.quantile(scores, quantile_level, method="higher"))
        return self

    def predict_sets(self, probability: np.ndarray) -> list[set[int]]:
        if set(self.quantiles_) != {0, 1}:
            raise RuntimeError("Conformal predictor is not fitted")
        p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
        sets: list[set[int]] = []
        for value in p:
            prediction_set: set[int] = set()
            if value <= self.quantiles_[0]:  # 1 - P(y=0) = p
                prediction_set.add(0)
            if 1.0 - value <= self.quantiles_[1]:
                prediction_set.add(1)
            sets.append(prediction_set)
        return sets

    def coverage(self, probability: np.ndarray, labels: np.ndarray) -> float:
        sets = self.predict_sets(probability)
        y = np.asarray(labels, dtype=int)
        return float(np.mean([int(label in prediction_set) for label, prediction_set in zip(y, sets)]))

    def mean_set_size(self, probability: np.ndarray) -> float:
        return float(np.mean([len(value) for value in self.predict_sets(probability)]))
