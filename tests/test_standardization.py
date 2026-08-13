from toxicity_screening.standardization import standardize_smiles


def test_standardization_retains_audit_fields():
    result = standardize_smiles("CC(=O)[O-].[Na+]")
    assert result.standardization_status == "success"
    assert result.standardized_smiles is not None
    assert result.inchikey
    assert result.transformation_log.startswith("[")


def test_invalid_smiles_is_explicit_failure():
    result = standardize_smiles("not_a_smiles")
    assert result.standardization_status == "failed"
    assert result.failure_reason
    assert result.standardized_smiles is None
