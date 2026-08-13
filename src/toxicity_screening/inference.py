from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .fingerprints import morgan_array, morgan_bit_vector
from .ood import assess_ood
from .scaffolds import scaffold_novelty
from .standardization import standardize_smiles


@dataclass
class EndpointBundle:
    endpoint: str
    model: Any
    calibrator: Any | None
    threshold: float
    fingerprint_radius: int
    fingerprint_bits: int
    use_chirality: bool
    training_fingerprints: list[Any]
    training_scaffolds: set[str]
    ad_borderline_threshold: float
    ad_outside_threshold: float
    uncertainty_threshold: float
    ensemble_models: list[Any] | None = None
    metadata: dict[str, Any] | None = None


def load_bundle(path: str | Path) -> EndpointBundle:
    payload = joblib.load(path)
    if isinstance(payload, EndpointBundle):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported bundle format at {path}")
    return EndpointBundle(**payload)


def _predict_member(model: Any, features: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features.reshape(1, -1))[:, 1][0])
    raise TypeError(f"Model lacks predict_proba: {type(model)}")


def _recommend(endpoint: str, probability: float, threshold: float, abstain: bool) -> str:
    if abstain:
        return "Abstain because the molecule is outside the applicability domain or uncertainty is excessive"
    if probability >= max(threshold, 0.75):
        return "High-priority toxicity testing"
    if probability >= min(threshold, 0.40):
        return "Experimental confirmation recommended"
    return "Low predicted concern; experimental confirmation remains appropriate"


def predict_smiles(smiles: str, bundles: list[EndpointBundle], molecule_id: str = "query") -> pd.DataFrame:
    standardized = standardize_smiles(smiles)
    if standardized.standardization_status != "success":
        return pd.DataFrame([
            {
                "molecule_id": molecule_id,
                "original_smiles": smiles,
                "standardized_smiles": None,
                "endpoint": bundle.endpoint,
                "predicted_class": None,
                "raw_probability": np.nan,
                "calibrated_probability": np.nan,
                "uncertainty": np.nan,
                "applicability_domain": "unavailable",
                "nearest_training_similarity": np.nan,
                "scaffold_novelty": None,
                "ood_warning": True,
                "abstention_status": "abstain",
                "interpretation": f"Structure standardization failed: {standardized.failure_reason}",
                "recommendation": "Prediction unsupported; provide a valid molecular structure",
            }
            for bundle in bundles
        ])

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        features = morgan_array(
            standardized.standardized_smiles,
            radius=bundle.fingerprint_radius,
            n_bits=bundle.fingerprint_bits,
            use_chirality=bundle.use_chirality,
        )
        raw = _predict_member(bundle.model, features)
        if bundle.ensemble_models:
            member_probabilities = np.array([_predict_member(model, features) for model in bundle.ensemble_models])
            uncertainty = float(member_probabilities.std(ddof=1 if len(member_probabilities) > 1 else 0))
            raw = float(member_probabilities.mean())
        else:
            uncertainty = 0.0
        calibrated = float(bundle.calibrator.predict(np.array([raw]))[0]) if bundle.calibrator else raw

        query_fp = morgan_bit_vector(
            standardized.standardized_smiles,
            radius=bundle.fingerprint_radius,
            n_bits=bundle.fingerprint_bits,
            use_chirality=bundle.use_chirality,
        )
        if bundle.training_fingerprints:
            from rdkit import DataStructs
            nearest_similarity = float(max(DataStructs.BulkTanimotoSimilarity(query_fp, bundle.training_fingerprints)))
        else:
            nearest_similarity = float("nan")
        if np.isnan(nearest_similarity) or nearest_similarity < bundle.ad_outside_threshold:
            ad_status = "outside"
        elif nearest_similarity < bundle.ad_borderline_threshold:
            ad_status = "borderline"
        else:
            ad_status = "inside"
        novel_scaffold = scaffold_novelty(standardized.scaffold or "", bundle.training_scaffolds)
        ood = assess_ood(
            applicability_domain=ad_status,
            nearest_training_similarity=nearest_similarity,
            scaffold_novelty=novel_scaffold,
            uncertainty=uncertainty,
            uncertainty_threshold=bundle.uncertainty_threshold,
        )
        predicted_class = None if ood.abstention_status == "abstain" else int(calibrated >= bundle.threshold)
        rows.append({
            "molecule_id": molecule_id,
            "original_smiles": smiles,
            "standardized_smiles": standardized.standardized_smiles,
            "endpoint": bundle.endpoint,
            "predicted_class": predicted_class,
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "uncertainty": uncertainty,
            "applicability_domain": ad_status,
            "nearest_training_similarity": nearest_similarity,
            "scaffold_novelty": novel_scaffold,
            "ood_warning": ood.ood_warning,
            "abstention_status": ood.abstention_status,
            "interpretation": json.dumps({"ood_reasons": ood.reasons}),
            "recommendation": _recommend(bundle.endpoint, calibrated, bundle.threshold, ood.abstention_status == "abstain"),
        })
    return pd.DataFrame(rows)


def load_bundles(model_dir: str | Path) -> list[EndpointBundle]:
    paths = sorted(Path(model_dir).glob("*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No endpoint bundles found in {model_dir}")
    return [load_bundle(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Endpoint-specific molecular toxicity screening")
    parser.add_argument("--smiles")
    parser.add_argument("--input-csv")
    parser.add_argument("--output-csv", default="results/predictions/inference.csv")
    parser.add_argument("--model-dir", default="models/calibrated")
    args = parser.parse_args()
    if bool(args.smiles) == bool(args.input_csv):
        parser.error("Provide exactly one of --smiles or --input-csv")
    bundles = load_bundles(args.model_dir)
    if args.smiles:
        output = predict_smiles(args.smiles, bundles)
        print(output.to_string(index=False))
        return
    input_frame = pd.read_csv(args.input_csv)
    required = {"molecule_id", "smiles"}
    if not required.issubset(input_frame.columns):
        raise ValueError(f"Input CSV requires columns: {sorted(required)}")
    output = pd.concat([
        predict_smiles(row.smiles, bundles, str(row.molecule_id))
        for row in input_frame.itertuples(index=False)
    ], ignore_index=True)
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
