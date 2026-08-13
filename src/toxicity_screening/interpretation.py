from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from .fingerprints import bit_environment_map


def permutation_importance_table(model, x, y, feature_names, *, seed: int = 20260723, repeats: int = 10):
    import pandas as pd

    result = permutation_importance(model, x, y, n_repeats=repeats, random_state=seed, scoring="average_precision")
    return pd.DataFrame({
        "feature": list(feature_names),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def shap_values(model, x_background: np.ndarray, x_explain: np.ndarray) -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP to compute model explanations") from exc
    explainer = shap.Explainer(model, x_background)
    return explainer(x_explain)


def map_fingerprint_bits(smiles: str, bit_importances: dict[int, float], *, radius: int = 2, n_bits: int = 2048):
    environments = bit_environment_map(smiles, radius=radius, n_bits=n_bits)
    return [
        {"bit": bit, "importance": float(value), "atom_radius_pairs": environments.get(bit, [])}
        for bit, value in sorted(bit_importances.items(), key=lambda item: abs(item[1]), reverse=True)
    ]


def integrated_gradients(model, inputs, baseline=None, target=None):
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("Install Captum for integrated gradients") from exc
    if baseline is None:
        import torch
        baseline = torch.zeros_like(inputs)
    return IntegratedGradients(model).attribute(inputs, baselines=baseline, target=target)


def atom_deletion_attributions(
    smiles: str,
    predict_probability,
) -> list[dict[str, object]]:
    """Estimate atom importance by deleting terminal atoms when chemically valid.

    This is a conservative perturbation diagnostic. Failed or sanitization-invalid edits
    are reported and excluded rather than assigned zero effect.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    baseline = float(predict_probability(smiles))
    records: list[dict[str, object]] = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        editable = Chem.RWMol(mol)
        try:
            editable.RemoveAtom(index)
            perturbed = editable.GetMol()
            Chem.SanitizeMol(perturbed)
            perturbed_smiles = Chem.MolToSmiles(perturbed, canonical=True, isomericSmiles=True)
            probability = float(predict_probability(perturbed_smiles))
            records.append(
                {
                    "atom_index": index,
                    "atomic_number": atom.GetAtomicNum(),
                    "perturbed_smiles": perturbed_smiles,
                    "baseline_probability": baseline,
                    "perturbed_probability": probability,
                    "attribution": baseline - probability,
                    "status": "success",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "atom_index": index,
                    "atomic_number": atom.GetAtomicNum(),
                    "perturbed_smiles": None,
                    "baseline_probability": baseline,
                    "perturbed_probability": None,
                    "attribution": None,
                    "status": f"invalid_edit: {type(exc).__name__}",
                }
            )
    return records


def explanation_rank_stability(rankings: list[list[object]], top_k: int = 20) -> float:
    """Mean pairwise Jaccard overlap across top-k explanation rankings."""
    if len(rankings) < 2:
        return float("nan")
    sets = [set(ranking[:top_k]) for ranking in rankings]
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values.append(len(sets[i] & sets[j]) / max(1, len(union)))
    return float(np.mean(values))
