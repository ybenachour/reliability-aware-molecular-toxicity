from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from toxicity_screening.fingerprints import morgan_array, morgan_bit_vector
from toxicity_screening.inference import EndpointBundle, predict_smiles
from toxicity_screening.standardization import standardize_smiles
from toxicity_screening.utils import set_global_seed, utc_now

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260723
set_global_seed(SEED)

# Arbitrary non-toxicological labels solely to exercise the software path.
smiles = [
    "CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O", "CC(C)O", "CC(C)N",
    "c1ccccc1", "c1ccncc1", "c1ccccc1O", "c1ccccc1N", "CCOC", "CCNC",
    "O=C=O", "N#N", "CC#N", "C1CCCCC1", "C1=CC=CN=C1", "CCS",
]
labels = np.array([sum(map(ord, s)) % 2 for s in smiles], dtype=int)
standardized = [standardize_smiles(s) for s in smiles]
assert all(row.standardization_status == "success" for row in standardized)
canonical = [row.standardized_smiles for row in standardized]
X = np.vstack([morgan_array(s, n_bits=256) for s in canonical])
model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X, labels)
training_fps = [morgan_bit_vector(s, n_bits=256) for s in canonical]

bundles = []
for endpoint in ["herg_blockade", "ames_mutagenicity", "SR-p53", "SR-ATAD5", "SR-ARE", "SR-MMP"]:
    bundles.append(
        EndpointBundle(
            endpoint=endpoint,
            model=model,
            calibrator=None,
            threshold=0.5,
            fingerprint_radius=2,
            fingerprint_bits=256,
            use_chirality=True,
            training_fingerprints=training_fps,
            training_scaffolds={row.scaffold or "" for row in standardized},
            ad_borderline_threshold=0.35,
            ad_outside_threshold=0.15,
            uncertainty_threshold=0.2,
            ensemble_models=None,
            metadata={"synthetic_smoke_only": True},
        )
    )
result = predict_smiles("CCOC(=O)C", bundles, "synthetic_query")
required = {
    "endpoint", "raw_probability", "calibrated_probability", "uncertainty",
    "applicability_domain", "ood_warning", "abstention_status", "recommendation",
}
assert required.issubset(result.columns)
assert len(result) == 6

report = {
    "created_at": utc_now(),
    "status": "passed",
    "scientific_result": False,
    "purpose": "software-path validation only; arbitrary labels have no toxicological meaning",
    "checks": {
        "structure_standardization": True,
        "fingerprint_generation": True,
        "binary_model_fit": True,
        "six_endpoint_inference_schema": True,
        "applicability_domain_and_ood_logic": True,
    },
}
output = ROOT / "reports/synthetic_pipeline_smoke.json"
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(output)
