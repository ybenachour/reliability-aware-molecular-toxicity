import numpy as np
from sklearn.linear_model import LogisticRegression

from toxicity_screening.inference import EndpointBundle, predict_smiles
from toxicity_screening.fingerprints import morgan_array, morgan_bit_vector


def bundle():
    smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    x = np.vstack([morgan_array(s, n_bits=128) for s in smiles])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(x, y)
    return EndpointBundle(
        endpoint="test_endpoint",
        model=model,
        calibrator=None,
        threshold=0.5,
        fingerprint_radius=2,
        fingerprint_bits=128,
        use_chirality=True,
        training_fingerprints=[morgan_bit_vector(s, n_bits=128) for s in smiles],
        training_scaffolds={"", "c1ccccc1"},
        ad_borderline_threshold=0.3,
        ad_outside_threshold=0.1,
        uncertainty_threshold=0.2,
    )


def test_valid_inference_schema():
    result = predict_smiles("CCO", [bundle()])
    required = {"endpoint", "calibrated_probability", "uncertainty", "applicability_domain", "ood_warning", "recommendation"}
    assert required.issubset(result.columns)
    assert len(result) == 1


def test_invalid_structure_abstains():
    result = predict_smiles("not_a_smiles", [bundle()])
    assert result.iloc[0].abstention_status == "abstain"
    assert result.iloc[0].predicted_class is None
