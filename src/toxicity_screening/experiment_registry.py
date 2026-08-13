from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import stable_id, utc_now


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    created_at: str
    endpoint: str
    model_family: str
    split_strategy: str
    seed: int
    parameters_json: str
    metrics_json: str
    artifact_path: str | None
    status: str
    notes: str


class ExperimentRegistry:
    """Append-only CSV experiment registry requiring no server process."""

    columns = list(ExperimentRecord.__dataclass_fields__)

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        endpoint: str,
        model_family: str,
        split_strategy: str,
        seed: int,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        artifact_path: str | None = None,
        status: str = "completed",
        notes: str = "",
    ) -> ExperimentRecord:
        created_at = utc_now()
        record = ExperimentRecord(
            run_id=stable_id(endpoint, model_family, split_strategy, seed, created_at),
            created_at=created_at,
            endpoint=endpoint,
            model_family=model_family,
            split_strategy=split_strategy,
            seed=int(seed),
            parameters_json=json.dumps(parameters, sort_keys=True, default=str),
            metrics_json=json.dumps(metrics, sort_keys=True, default=str),
            artifact_path=artifact_path,
            status=status,
            notes=notes,
        )
        frame = pd.DataFrame([asdict(record)], columns=self.columns)
        frame.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        return record

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.path)
