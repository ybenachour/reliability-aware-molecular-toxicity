from toxicity_screening.ood import assess_ood


def test_outside_domain_forces_abstention():
    result = assess_ood(
        applicability_domain="outside",
        nearest_training_similarity=0.1,
        scaffold_novelty=True,
        uncertainty=0.01,
    )
    assert result.abstention_status == "abstain"
    assert result.ood_warning


def test_novel_scaffold_warns_without_automatic_abstention():
    result = assess_ood(
        applicability_domain="inside",
        nearest_training_similarity=0.7,
        scaffold_novelty=True,
        uncertainty=0.01,
    )
    assert result.ood_warning
    assert result.abstention_status == "supported"
