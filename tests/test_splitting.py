import pandas as pd

from toxicity_screening.splitting import (
    SplitFractions,
    assert_no_group_leakage,
    global_multitask_scaffold_split,
)


def test_global_split_prevents_molecule_and_scaffold_leakage():
    frame = pd.DataFrame(
        {
            "inchikey": ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "scaffold": ["s1", "s1", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
            "endpoint": ["x", "y", "x", "x", "x", "x", "x", "x", "x", "x"],
        }
    )
    result = global_multitask_scaffold_split(
        frame,
        seed=7,
        fractions=SplitFractions(0.6, 0.2, 0.2),
    )
    assert_no_group_leakage(result, "inchikey", "split")
    assert_no_group_leakage(result, "scaffold", "split")
    assert set(result["split"]) == {"train", "validation", "test"}
