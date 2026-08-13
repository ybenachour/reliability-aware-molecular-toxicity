from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import DataStructs
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


@dataclass
class SimilarityApplicabilityDomain:
    borderline_threshold: float = 0.45
    outside_threshold: float = 0.30
    reference_fingerprints: Sequence | None = None

    def fit(self, fingerprints: Sequence) -> "SimilarityApplicabilityDomain":
        self.reference_fingerprints = list(fingerprints)
        if not self.reference_fingerprints:
            raise ValueError("Applicability domain needs training fingerprints")
        return self

    def score(self, fingerprint) -> float:
        if self.reference_fingerprints is None:
            raise RuntimeError("Applicability domain is not fitted")
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, list(self.reference_fingerprints))
        return float(max(values)) if values else float("nan")

    def status(self, score: float) -> str:
        if np.isnan(score) or score < self.outside_threshold:
            return "outside"
        if score < self.borderline_threshold:
            return "borderline"
        return "inside"


class DescriptorDistanceDomain:
    def __init__(self, quantile_inside: float = 0.95, quantile_borderline: float = 0.99):
        self.quantile_inside = quantile_inside
        self.quantile_borderline = quantile_borderline
        self.location_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.inside_threshold_: float | None = None
        self.borderline_threshold_: float | None = None

    def fit(self, x: np.ndarray) -> "DescriptorDistanceDomain":
        estimator = LedoitWolf().fit(x)
        self.location_ = estimator.location_
        self.precision_ = estimator.precision_
        distances = self.distance(x)
        self.inside_threshold_ = float(np.quantile(distances, self.quantile_inside))
        self.borderline_threshold_ = float(np.quantile(distances, self.quantile_borderline))
        return self

    def distance(self, x: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("Descriptor domain is not fitted")
        delta = np.asarray(x) - self.location_
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, self.precision_, delta))

    def status(self, x: np.ndarray) -> np.ndarray:
        d = self.distance(x)
        assert self.inside_threshold_ is not None and self.borderline_threshold_ is not None
        return np.where(d <= self.inside_threshold_, "inside", np.where(d <= self.borderline_threshold_, "borderline", "outside"))


def knn_similarity_summary(query: np.ndarray, reference: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    neighbors = NearestNeighbors(n_neighbors=min(k, len(reference)), metric="jaccard").fit(reference.astype(bool))
    distances, _ = neighbors.kneighbors(query.astype(bool))
    similarities = 1.0 - distances
    return {"mean_knn_similarity": similarities.mean(axis=1), "minimum_knn_similarity": similarities.min(axis=1)}


class LatentDistanceDomain(DescriptorDistanceDomain):
    """Mahalanobis applicability domain for learned molecular embeddings.

    Pass training encoder embeddings to ``fit`` and query embeddings to ``status``.
    The implementation is intentionally representation-agnostic so the same audit can
    be applied to fingerprint MLP and graph encoders.
    """

    pass
