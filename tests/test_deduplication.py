import numpy as np
import pandas as pd

from toxicity_screening.deduplication import audit_duplicates, resolve_duplicates


def frame():
    return pd.DataFrame({"key": ["a", "a", "b", "b", "c"], "label": [1, 1, 0, 1, np.nan]})


def test_duplicate_audit_detects_conflict():
    audit = audit_duplicates(frame(), "key", "label")
    assert audit.duplicate_molecules == 2
    assert audit.conflicting_molecules == 1


def test_remove_conflicts_does_not_invent_label():
    resolved = resolve_duplicates(frame(), key="key", label="label", policy="remove_conflicts")
    assert set(resolved.key) == {"a", "c"}
    assert resolved.loc[resolved.key == "c", "label"].isna().all()


def test_uncertain_policy_masks_conflict():
    resolved = resolve_duplicates(frame(), key="key", label="label", policy="uncertain")
    assert np.isnan(resolved.loc[resolved.key == "b", "label"].iloc[0])
