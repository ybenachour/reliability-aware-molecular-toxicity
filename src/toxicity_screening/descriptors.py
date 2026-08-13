from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


DEFAULT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NHOHCount",
    "NOCount", "MolMR", "LabuteASA", "BalabanJ", "BertzCT",
]
_DESCRIPTOR_MAP = dict(Descriptors.descList)


def available_descriptors(names: Iterable[str] | None = None) -> list[str]:
    requested = list(names or DEFAULT_DESCRIPTOR_NAMES)
    missing = sorted(set(requested) - set(_DESCRIPTOR_MAP))
    if missing:
        raise ValueError(f"Unknown RDKit descriptors: {missing}")
    return requested


def descriptor_vector(smiles: str, names: Iterable[str] | None = None) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    descriptor_names = available_descriptors(names)
    values: dict[str, float] = {}
    for name in descriptor_names:
        try:
            value = float(_DESCRIPTOR_MAP[name](mol))
        except Exception:
            value = np.nan
        values[name] = value
    return values


def descriptor_frame(smiles: Iterable[str], names: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame([descriptor_vector(value, names) for value in smiles])
