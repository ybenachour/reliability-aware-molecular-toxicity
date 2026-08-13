from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

# Load the preserved implementation as a private package module so its relative
# imports continue to resolve inside toxicity_screening.
_BACKUP_PATH = Path(__file__).with_name("external_validation.pre_dtype_fix.py")
_SPEC = importlib.util.spec_from_file_location(
    "toxicity_screening._external_validation_pre_dtype_fix",
    _BACKUP_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load preserved external-validation module: {_BACKUP_PATH}")

_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

# Re-export the preserved public API, except for run_external_validation, which
# is wrapped below to load only the four columns needed for overlap auditing.
for _name in dir(_LEGACY):
    if not _name.startswith("_") and _name != "run_external_validation":
        globals()[_name] = getattr(_LEGACY, _name)


def run_external_validation(
    root: str | Path = ".",
    *,
    profile: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run external validation with deterministic development-table dtypes.

    The preserved implementation is unchanged. During this call only, pandas
    is instructed to read the development table using the four columns required
    for structural-overlap auditing and to preserve them as nullable strings.
    This removes the mixed-type DtypeWarning and reduces memory usage.
    """

    root_path = _LEGACY._root(root)
    development_path = (root_path / "data/processed/endpoint_records.csv").resolve()
    original_read_csv = _LEGACY.pd.read_csv

    def _read_csv_with_development_schema(
        filepath_or_buffer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            candidate = Path(os.fspath(filepath_or_buffer)).resolve()
        except (TypeError, ValueError, OSError):
            candidate = None

        if candidate == development_path:
            kwargs = dict(kwargs)
            kwargs.setdefault(
                "usecols",
                [
                    "endpoint",
                    "standardized_smiles",
                    "inchikey",
                    "scaffold",
                ],
            )
            kwargs.setdefault(
                "dtype",
                {
                    "endpoint": "string",
                    "standardized_smiles": "string",
                    "inchikey": "string",
                    "scaffold": "string",
                },
            )
            kwargs.setdefault("low_memory", False)

        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    _LEGACY.pd.read_csv = _read_csv_with_development_schema
    try:
        return _LEGACY.run_external_validation(
            root_path,
            profile=profile,
            seed=seed,
        )
    finally:
        _LEGACY.pd.read_csv = original_read_csv


__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
    }
)
