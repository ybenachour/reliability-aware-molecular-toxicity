from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OODAssessment:
    applicability_domain: str
    nearest_training_similarity: float
    scaffold_novelty: bool
    uncertainty: float
    ood_warning: bool
    abstention_status: str
    reasons: tuple[str, ...]


def assess_ood(
    *,
    applicability_domain: str,
    nearest_training_similarity: float,
    scaffold_novelty: bool,
    uncertainty: float,
    uncertainty_threshold: float = 0.20,
) -> OODAssessment:
    reasons: list[str] = []
    if applicability_domain == "outside":
        reasons.append("outside_applicability_domain")
    if scaffold_novelty:
        reasons.append("novel_scaffold")
    if uncertainty > uncertainty_threshold:
        reasons.append("high_uncertainty")
    abstain = applicability_domain == "outside" or uncertainty > uncertainty_threshold
    return OODAssessment(
        applicability_domain=applicability_domain,
        nearest_training_similarity=float(nearest_training_similarity),
        scaffold_novelty=bool(scaffold_novelty),
        uncertainty=float(uncertainty),
        ood_warning=bool(reasons),
        abstention_status="abstain" if abstain else "supported",
        reasons=tuple(reasons),
    )
