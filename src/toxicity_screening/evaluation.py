from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import binary_metrics


def _stable_sigmoid(scores: np.ndarray) -> np.ndarray:
    """Compute sigmoid probabilities without overflow-prone matrix operations."""
    values = np.asarray(scores, dtype=np.float64)
    probabilities = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-values[nonnegative])
    )
    negative_exp = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return probabilities


def safe_logistic_positive_probability(
    fitted_pipeline: Pipeline,
    x: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict class-1 probabilities without BLAS-backed matrix products.

    The fitted scaler and logistic-regression coefficients are collapsed into
    one effective coefficient vector. Scores are then computed batchwise with
    elementwise multiplication and ``np.sum`` only; ``@``, ``dot``, sklearn's
    ``decision_function``, and its native ``predict_proba`` path are avoided.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not isinstance(fitted_pipeline, Pipeline):
        raise TypeError("Expected a fitted sklearn Pipeline")

    try:
        scaler = fitted_pipeline.named_steps["scale"]
        classifier = fitted_pipeline.named_steps["model"]
    except KeyError as error:
        raise ValueError(
            "Expected pipeline steps named 'scale' and 'model'"
        ) from error

    if not isinstance(scaler, StandardScaler):
        raise TypeError("The 'scale' step must be StandardScaler")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("The 'model' step must be LogisticRegression")

    classes = np.asarray(classifier.classes_)
    if classes.shape != (2,) or classes.tolist() != [0, 1]:
        raise ValueError(
            "Safe logistic prediction currently requires binary classes [0, 1]"
        )

    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError(
            "Expected one logistic coefficient row for binary classification"
        )

    effective_coefficients = coefficients[0].copy()
    scale = getattr(scaler, "scale_", None)
    if scale is not None:
        scale_array = np.asarray(scale, dtype=np.float64)
        if scale_array.shape != effective_coefficients.shape:
            raise ValueError("Scaler and classifier feature dimensions differ")
        effective_coefficients = effective_coefficients / scale_array

    intercept = float(np.asarray(classifier.intercept_, dtype=np.float64)[0])
    if bool(getattr(scaler, "with_mean", False)):
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        intercept -= float(np.sum(mean * effective_coefficients))

    features = np.asarray(x)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}")
    if features.shape[1] != effective_coefficients.shape[0]:
        raise ValueError(
            "Feature matrix and logistic coefficient dimensions differ: "
            f"{features.shape[1]} versus {effective_coefficients.shape[0]}"
        )

    probabilities = np.empty(features.shape[0], dtype=np.float64)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        batch = np.asarray(features[start:stop], dtype=np.float64)
        scores = np.sum(
            batch * effective_coefficients,
            axis=1,
            dtype=np.float64,
        )
        scores = scores + intercept
        probabilities[start:stop] = _stable_sigmoid(scores)

    return probabilities


class SafeLogisticPipeline(Pipeline):
    """Pipeline whose inference path avoids native BLAS matrix products."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = safe_logistic_positive_probability(self, x)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, x: np.ndarray) -> np.ndarray:
        positive = safe_logistic_positive_probability(self, x)
        return (positive >= 0.5).astype(int)


def predict_qsar_probability(
    model: Any,
    x: np.ndarray,
    *,
    model_name: str | None = None,
) -> np.ndarray:
    """Return class-1 probabilities using the safest available backend."""
    if model_name == "logistic_regression" or isinstance(
        model, SafeLogisticPipeline
    ):
        return safe_logistic_positive_probability(model, x)

    probability_matrix = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probability_matrix.ndim != 2:
        raise ValueError("predict_proba must return a 2D array")

    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(f"Could not identify class 1 in model classes: {classes}")
    return probability_matrix[:, int(positive_indices[0])]


def build_qsar_estimators(seed: int = 20260723) -> dict[str, Any]:
    """Build conservative, single-threaded QSAR controls.

    Logistic fitting uses ``liblinear`` and logistic inference uses a custom
    elementwise implementation. Tree and XGBoost estimators are constrained to
    one worker to avoid OpenMP runtime conflicts.

    This module deliberately has no PyTorch import. Neural architectures live
    in ``toxicity_screening.neural_models`` and are initialized only by neural
    notebooks.
    """
    estimators: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": SafeLogisticPipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        penalty="l2",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        ),
        "svm": SVC(
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    except ImportError:
        pass
    return estimators


def fit_evaluate_estimators(
    estimators: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fitted: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        model = clone(estimator).fit(x_train, y_train)
        probability = predict_qsar_probability(
            model,
            x_test,
            model_name=name,
        )
        fitted[name] = model
        metrics[name] = binary_metrics(y_test, probability, threshold)
    return fitted, metrics


def endpoint_transfer(
    single_task_score: float,
    multitask_score: float,
    neutral_tolerance: float = 0.005,
) -> dict[str, float | str]:
    difference = float(multitask_score - single_task_score)
    if difference > neutral_tolerance:
        category = "positive"
    elif difference < -neutral_tolerance:
        category = "negative"
    else:
        category = "neutral"
    return {
        "single_task": single_task_score,
        "multitask": multitask_score,
        "transfer": difference,
        "category": category,
    }
