from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DuplicateResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class DuplicateAudit:
    raw_records: int
    unique_keys: int
    exact_duplicate_rows: int
    duplicate_molecules: int
    conflicting_molecules: int


def audit_duplicates(frame: pd.DataFrame, key: str, label: str) -> DuplicateAudit:
    valid = frame.dropna(subset=[key]).copy()
    grouped = valid.groupby(key, dropna=False)[label]
    observed_nunique = grouped.apply(lambda s: s.dropna().nunique())
    return DuplicateAudit(
        raw_records=int(len(frame)),
        unique_keys=int(valid[key].nunique()),
        exact_duplicate_rows=int(frame.duplicated().sum()),
        duplicate_molecules=int((valid[key].value_counts() > 1).sum()),
        conflicting_molecules=int((observed_nunique > 1).sum()),
    )


def resolve_duplicates(
    frame: pd.DataFrame,
    *,
    key: str,
    label: str,
    policy: str = "remove_conflicts",
    quality_column: str | None = None,
) -> pd.DataFrame:
    """Resolve duplicate molecule labels according to a declared policy.

    Policies:
      - remove_conflicts: retain one row only when all observed labels agree.
      - majority_vote: use strict majority; ties become missing.
      - quality_filter: retain highest-quality row; requires quality_column.
      - uncertain: conflicting labels become missing and uncertainty flag is set.
    """
    allowed = {"remove_conflicts", "majority_vote", "quality_filter", "uncertain"}
    if policy not in allowed:
        raise DuplicateResolutionError(f"Unknown policy {policy}; choose from {sorted(allowed)}")
    if quality_column and quality_column not in frame.columns:
        raise DuplicateResolutionError(f"Missing quality column: {quality_column}")

    rows: list[pd.Series] = []
    for molecule_key, group in frame.groupby(key, dropna=False, sort=False):
        group = group.copy()
        labels = pd.to_numeric(group[label], errors="coerce").dropna()
        conflict = labels.nunique() > 1
        representative = group.iloc[0].copy()
        representative["duplicate_count"] = len(group)
        representative["label_conflict"] = bool(conflict)
        representative["duplicate_policy"] = policy

        if policy == "remove_conflicts":
            if conflict:
                continue
            representative[label] = labels.iloc[0] if not labels.empty else np.nan
        elif policy == "majority_vote":
            counts = labels.value_counts()
            if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
                representative[label] = np.nan
            else:
                representative[label] = counts.index[0]
        elif policy == "quality_filter":
            if quality_column is None:
                raise DuplicateResolutionError("quality_filter requires quality_column")
            best = group.sort_values(quality_column, ascending=False, na_position="last").iloc[0]
            representative = best.copy()
            representative["duplicate_count"] = len(group)
            representative["label_conflict"] = bool(conflict)
            representative["duplicate_policy"] = policy
        else:  # uncertain
            representative[label] = np.nan if conflict else (labels.iloc[0] if not labels.empty else np.nan)
            representative["uncertain_label"] = bool(conflict)
        rows.append(representative)
    if not rows:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)
