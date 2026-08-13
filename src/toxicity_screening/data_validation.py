from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when downloaded or derived data violate declared assumptions."""


def require_columns(frame: pd.DataFrame, columns: Iterable[str], dataset_name: str = "dataset") -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{dataset_name} missing required columns: {missing}")


def validate_binary_label(series: pd.Series, *, allow_missing: bool = True, name: str = "label") -> None:
    values = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
    if not values.issubset({0, 1, 0.0, 1.0}):
        raise DataValidationError(f"{name} contains non-binary values: {sorted(values)}")
    if not allow_missing and series.isna().any():
        raise DataValidationError(f"{name} contains missing values")


def endpoint_summary(frame: pd.DataFrame, label_column: str) -> dict[str, float | int]:
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    observed = labels.dropna()
    positives = int((observed == 1).sum())
    negatives = int((observed == 0).sum())
    return {
        "records": int(len(frame)),
        "observed_labels": int(observed.size),
        "missing_labels": int(labels.isna().sum()),
        "positives": positives,
        "negatives": negatives,
        "positive_prevalence": float(positives / observed.size) if observed.size else np.nan,
    }


def validate_dataset_registry(registry: pd.DataFrame) -> None:
    required = {
        "dataset_name", "endpoint", "source", "official_url", "version",
        "download_date", "license", "citation", "raw_filename", "checksum",
        "number_of_records", "label_definition", "measurement_units",
        "assay_context", "notes",
    }
    require_columns(registry, required, "dataset_registry")
    if registry["checksum"].astype(str).str.fullmatch(r"[0-9a-f]{64}").fillna(False).eq(False).any():
        raise DataValidationError("dataset_registry contains invalid SHA-256 checksums")
    if registry["dataset_name"].duplicated().any():
        raise DataValidationError("dataset_registry dataset_name values must be unique")


def write_dataset_card(metadata: dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Dataset card: {metadata.get('dataset_name', 'unknown')}", ""]
    for key, value in metadata.items():
        lines.append(f"## {key.replace('_', ' ').title()}")
        lines.append(str(value))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
