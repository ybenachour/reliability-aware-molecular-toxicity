from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def fit(self, probability: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def predict(self, probability: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PlattCalibrator(ProbabilityCalibrator):
    model: LogisticRegression | None = None

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        x = logit(np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)).reshape(-1, 1)
        self.model = LogisticRegression(solver="lbfgs").fit(x, labels)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator is not fitted")
        x = logit(np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


@dataclass
class IsotonicCalibrator(ProbabilityCalibrator):
    model: IsotonicRegression | None = None

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip").fit(probability, labels)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator is not fitted")
        return np.asarray(self.model.predict(probability))


@dataclass
class TemperatureCalibrator(ProbabilityCalibrator):
    temperature: float = 1.0

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        logits = logit(np.clip(np.asarray(probability), 1e-6, 1 - 1e-6))
        labels = np.asarray(labels)

        def objective(log_temperature: float) -> float:
            temperature = np.exp(log_temperature)
            scaled = expit(logits / temperature)
            return float(-np.mean(labels * np.log(scaled + 1e-8) + (1 - labels) * np.log(1 - scaled + 1e-8)))

        result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
        if not result.success:
            raise RuntimeError(f"Temperature optimization failed: {result.message}")
        self.temperature = float(np.exp(result.x))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        logits = logit(np.clip(np.asarray(probability), 1e-6, 1 - 1e-6))
        return expit(logits / self.temperature)
