from __future__ import annotations

"""Compatibility wrapper around the preserved pipeline implementation.

The original implementation is loaded from ``pipeline.pre_qsar_crash_fix.py``.
All of its public and internal symbols remain available, while the classical
QSAR training entry point is overridden so that it does not initialize PyTorch,
writes persistent progress records, and routes logistic inference through a
BLAS-independent probability backend.
"""

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

_BASE_PATH = Path(__file__).with_name("pipeline.pre_qsar_crash_fix.py")
if not _BASE_PATH.exists():
    raise ImportError(f"Required preserved pipeline source is missing: {_BASE_PATH}")

_MODULE_NAME = f"{__package__}._pipeline_base_runtime"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load preserved pipeline source: {_BASE_PATH}")

_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _BASE
_SPEC.loader.exec_module(_BASE)

# Preserve the complete original API, including intentionally internal helpers
# used by the orchestration notebooks.
for _name, _value in vars(_BASE).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from .evaluation import predict_qsar_probability


def train_qsar_baselines(
    root: str | Path = ".",
    profile: str | None = None,
) -> pd.DataFrame:
    """Fit endpoint-specific QSAR controls under the persistent scaffold split.

    Classical QSAR execution uses Python/NumPy-only seeding and never invokes
    ``set_global_seed``, which initializes PyTorch. A progress JSON file is
    updated before and after model fitting and prediction stages. Progress-file
    failures are diagnostic only and never terminate model training.
    """
    root = _root(root)
    train_config = load_yaml(root / "configs/training_config.yaml")
    seed = int(train_config["seed"])
    profile = profile or os.environ.get("TOX_SCREEN_PROFILE", "smoke")
    profile_config = train_config["profiles"][profile]
    cap = profile_config["sample_cap_per_endpoint"]

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    progress_path = root / "reports" / "qsar_training_progress.json"

    def report_progress(status: str, **details: Any) -> None:
        payload = {
            "updated_at": utc_now(),
            "status": status,
            "profile": profile,
            "seed": seed,
            **details,
        }
        try:
            atomic_write_json(payload, progress_path)
        except OSError as error:
            print(
                "[QSAR] progress_write_warning "
                f"status={status} error={type(error).__name__}: {error}",
                flush=True,
            )

        detail_text = " ".join(
            f"{key}={value}" for key, value in details.items()
        )
        suffix = f" {detail_text}" if detail_text else ""
        print(f"[QSAR] {status}{suffix}", flush=True)

    report_progress("matrix_load_started")
    X, feature_index, records = _load_model_matrix(root)
    records = records.merge(
        feature_index,
        on="molecule_id",
        how="left",
        validate="many_to_one",
    )
    report_progress(
        "matrix_load_completed",
        molecules=int(len(feature_index)),
        records=int(len(records)),
        features=int(X.shape[1]),
    )

    report_progress("estimator_registry_started")
    estimators = build_qsar_estimators(seed)
    report_progress(
        "estimator_registry_completed",
        models=list(estimators),
    )

    result_rows: list[dict[str, Any]] = []

    for endpoint in ENDPOINTS:
        report_progress("endpoint_started", endpoint=endpoint)
        endpoint_frame = records.loc[
            (records["endpoint"] == endpoint)
            & records["label"].notna()
        ].copy()
        if endpoint_frame.empty:
            report_progress(
                "endpoint_failed",
                endpoint=endpoint,
                reason="no_observed_labels",
            )
            raise PipelineDataError(
                f"No observed labels for endpoint {endpoint}"
            )

        partitions = {
            partition: endpoint_frame.loc[
                endpoint_frame["scaffold_split"] == partition
            ]
            for partition in ["train", "validation", "test"]
        }
        if any(part.empty for part in partitions.values()):
            report_progress(
                "endpoint_failed",
                endpoint=endpoint,
                reason="empty_partition",
            )
            raise PipelineDataError(
                f"Endpoint {endpoint} has an empty scaffold partition"
            )

        train_frame = _sample_training_rows(
            partitions["train"],
            cap,
            seed,
        )
        x_train = X[train_frame["row"].astype(int)]
        y_train = train_frame["label"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            report_progress(
                "endpoint_failed",
                endpoint=endpoint,
                reason="one_training_class",
            )
            raise PipelineDataError(
                f"Endpoint {endpoint} training partition has one class"
            )

        fitted: dict[str, Any] = {}
        for model_name, estimator in estimators.items():
            report_progress(
                "model_fit_started",
                endpoint=endpoint,
                model=model_name,
                train_rows=int(len(train_frame)),
                features=int(x_train.shape[1]),
            )
            try:
                model = clone(estimator).fit(x_train, y_train)
            except Exception as exc:
                failure = {
                    "endpoint": endpoint,
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                try:
                    atomic_write_json(
                        failure,
                        root
                        / "reports"
                        / f"model_failure_{endpoint}_{model_name}.json",
                    )
                except OSError as write_error:
                    print(
                        "[QSAR] failure_report_write_warning "
                        f"endpoint={endpoint} model={model_name} "
                        f"error={type(write_error).__name__}: {write_error}",
                        flush=True,
                    )
                report_progress(
                    "model_fit_failed",
                    endpoint=endpoint,
                    model=model_name,
                    error=failure["error"],
                )
                continue

            fitted[model_name] = model
            report_progress(
                "model_fit_completed",
                endpoint=endpoint,
                model=model_name,
            )

            for partition_name in ["validation", "test"]:
                part = partitions[partition_name]
                prediction_backend = (
                    "elementwise_logistic"
                    if model_name == "logistic_regression"
                    else "native_predict_proba"
                )
                report_progress(
                    "prediction_started",
                    endpoint=endpoint,
                    model=model_name,
                    partition=partition_name,
                    rows=int(len(part)),
                    backend=prediction_backend,
                )
                y = part["label"].astype(int).to_numpy()
                probability = predict_qsar_probability(
                    model,
                    X[part["row"].astype(int)],
                    model_name=model_name,
                )
                metrics = binary_metrics(
                    y,
                    probability,
                    threshold=0.5,
                )
                result_rows.append(
                    {
                        "endpoint": endpoint,
                        "model": model_name,
                        "split_strategy": "global_scaffold",
                        "partition": partition_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "confusion_matrix"
                        },
                    }
                )
                prediction_frame = part[
                    [
                        "molecule_id",
                        "standardized_smiles",
                        "scaffold",
                        "label",
                    ]
                ].copy()
                prediction_frame["endpoint"] = endpoint
                prediction_frame["model"] = model_name
                prediction_frame["partition"] = partition_name
                prediction_frame["probability"] = probability
                prediction_frame.to_csv(
                    root
                    / "results"
                    / "predictions"
                    / f"{endpoint}_{model_name}_{partition_name}.csv",
                    index=False,
                )
                report_progress(
                    "prediction_completed",
                    endpoint=endpoint,
                    model=model_name,
                    partition=partition_name,
                    backend=prediction_backend,
                )

            report_progress(
                "model_completed",
                endpoint=endpoint,
                model=model_name,
            )

        if not fitted:
            report_progress(
                "endpoint_failed",
                endpoint=endpoint,
                reason="all_models_failed",
            )
            raise PipelineDataError(
                f"All QSAR estimators failed for endpoint {endpoint}"
            )

        joblib.dump(
            fitted,
            root / "models" / "qsar" / f"{endpoint}_estimators.joblib",
        )
        report_progress(
            "endpoint_completed",
            endpoint=endpoint,
            fitted_models=list(fitted),
        )

    metrics_frame = pd.DataFrame(result_rows)
    metrics_frame.to_csv(
        root / "results" / "metrics" / "qsar_baselines.csv",
        index=False,
    )
    report_progress(
        "completed",
        endpoints=int(metrics_frame["endpoint"].nunique()),
        metric_rows=int(len(metrics_frame)),
    )
    return metrics_frame
