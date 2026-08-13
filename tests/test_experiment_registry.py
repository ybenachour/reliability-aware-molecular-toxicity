from toxicity_screening.experiment_registry import ExperimentRegistry


def test_lightweight_registry_is_append_only(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiments.csv")
    first = registry.log(endpoint="x", model_family="rf", split_strategy="scaffold", seed=1, parameters={"n": 10}, metrics={"pr_auc": 0.5})
    second = registry.log(endpoint="x", model_family="rf", split_strategy="scaffold", seed=2, parameters={"n": 10}, metrics={"pr_auc": 0.6})
    frame = registry.read()
    assert len(frame) == 2
    assert first.run_id != second.run_id
