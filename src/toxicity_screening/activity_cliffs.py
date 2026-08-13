from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import DataStructs


@dataclass(frozen=True)
class CliffDefinition:
    minimum_tanimoto: float = 0.85
    label_difference: float = 1.0
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048


def find_binary_activity_cliffs(
    frame: pd.DataFrame,
    fingerprints: list,
    *,
    label_column: str,
    molecule_id_column: str = "molecule_id",
    definition: CliffDefinition = CliffDefinition(),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    for i in range(len(frame)):
        if pd.isna(labels.iloc[i]):
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1 :])
        for offset, similarity in enumerate(similarities, start=i + 1):
            if pd.isna(labels.iloc[offset]):
                continue
            difference = abs(float(labels.iloc[i]) - float(labels.iloc[offset]))
            if similarity >= definition.minimum_tanimoto and difference >= definition.label_difference:
                records.append({
                    "molecule_a": frame.iloc[i][molecule_id_column],
                    "molecule_b": frame.iloc[offset][molecule_id_column],
                    "tanimoto": float(similarity),
                    "label_a": float(labels.iloc[i]),
                    "label_b": float(labels.iloc[offset]),
                })
    return pd.DataFrame(records)
