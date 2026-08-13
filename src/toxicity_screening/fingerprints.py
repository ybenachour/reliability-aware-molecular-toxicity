from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def morgan_generator(radius: int = 2, n_bits: int = 2048, use_chirality: bool = True):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )


def morgan_bit_vector(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return morgan_generator(radius, n_bits, use_chirality).GetFingerprint(mol)


def morgan_array(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray:
    fp = morgan_bit_vector(smiles, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
    array = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def fingerprint_matrix(smiles: Iterable[str], **kwargs) -> np.ndarray:
    return np.vstack([morgan_array(value, **kwargs) for value in smiles])


def tanimoto_similarity(fp_a, fp_b) -> float:
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def max_tanimoto_to_reference(query_fp, reference_fps) -> float:
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, list(reference_fps))
    return float(max(similarities)) if similarities else float("nan")


def bit_environment_map(smiles: str, radius: int = 2, n_bits: int = 2048) -> dict[int, list[tuple[int, int]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    morgan_generator(radius, n_bits).GetFingerprint(mol, additionalOutput=additional)
    return {int(key): list(value) for key, value in additional.GetBitInfoMap().items()}
