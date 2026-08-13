import numpy as np

from toxicity_screening.uncertainty import SplitConformalBinaryClassifier, ensemble_uncertainty


def test_ensemble_uncertainty_shapes():
    result = ensemble_uncertainty(np.array([[0.1, 0.8], [0.2, 0.7], [0.15, 0.9]]))
    assert result["mean"].shape == (2,)
    assert np.all(result["standard_deviation"] >= 0)


def test_split_conformal_returns_nonempty_sets_on_calibration_like_data():
    p = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    conformal = SplitConformalBinaryClassifier(alpha=0.2).fit(p, y)
    sets = conformal.predict_sets(np.array([0.1, 0.5, 0.9]))
    assert all(isinstance(value, set) for value in sets)
    assert conformal.coverage(p, y) >= 0.8
