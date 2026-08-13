import numpy as np

from toxicity_screening.metrics import binary_metrics, select_threshold


def test_binary_metrics_contains_reliability_and_operational_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.7, 0.9])
    result = binary_metrics(y, p)
    for key in ["roc_auc", "pr_auc", "mcc", "brier", "ece", "nll", "recall_at_precision_0.80"]:
        assert key in result


def test_mcc_threshold_selection():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(y, p, "maximize_mcc")
    assert 0.2 < threshold <= 0.8
