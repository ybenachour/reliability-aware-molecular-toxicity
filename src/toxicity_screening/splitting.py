from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        values = [self.train, self.validation, self.test]
        if any(value <= 0 for value in values) or not np.isclose(sum(values), 1.0):
            raise SplitError(f"Fractions must be positive and sum to one: {values}")


def random_split(
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    fractions: SplitFractions = SplitFractions(),
) -> pd.Series:
    fractions.validate()
    observed = frame[label_column].notna()
    stratify = frame[label_column] if observed.all() and frame[label_column].nunique() > 1 else None
    indices = np.arange(len(frame))
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=fractions.validation + fractions.test,
        random_state=seed,
        stratify=stratify,
    )
    rest_labels = frame.iloc[rest_idx][label_column]
    rest_stratify = rest_labels if rest_labels.notna().all() and rest_labels.nunique() > 1 else None
    val_fraction_of_rest = fractions.validation / (fractions.validation + fractions.test)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_fraction_of_rest,
        random_state=seed + 1,
        stratify=rest_stratify,
    )
    assignment = pd.Series(index=frame.index, dtype="object")
    assignment.iloc[train_idx] = "train"
    assignment.iloc[val_idx] = "validation"
    assignment.iloc[test_idx] = "test"
    return assignment


def scaffold_split(
    frame: pd.DataFrame,
    *,
    scaffold_column: str = "scaffold",
    seed: int,
    fractions: SplitFractions = SplitFractions(),
) -> pd.Series:
    """Greedy group split that never divides a scaffold across partitions."""
    fractions.validate()
    if scaffold_column not in frame:
        raise SplitError(f"Missing scaffold column {scaffold_column}")
    groups: dict[str, list[int]] = defaultdict(list)
    for position, scaffold in enumerate(frame[scaffold_column].fillna("")):
        groups[str(scaffold)].append(position)
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {
        "train": fractions.train * len(frame),
        "validation": fractions.validation * len(frame),
        "test": fractions.test * len(frame),
    }
    counts = {key: 0 for key in targets}
    assignments: dict[int, str] = {}
    for _, positions in group_items:
        partition = min(targets, key=lambda key: (counts[key] / targets[key], counts[key]))
        for position in positions:
            assignments[position] = partition
        counts[partition] += len(positions)
    series = pd.Series([assignments[i] for i in range(len(frame))], index=frame.index, dtype="object")
    assert_no_group_leakage(frame.assign(_split=series), scaffold_column, "_split")
    return series


def global_multitask_scaffold_split(
    frame: pd.DataFrame,
    *,
    molecule_column: str = "inchikey",
    scaffold_column: str = "scaffold",
    seed: int,
    fractions: SplitFractions = SplitFractions(),
) -> pd.DataFrame:
    """Assign molecules and scaffold-connected components globally.

    Molecules sharing any valid scaffold are assigned to the same split.
    Molecules associated with multiple scaffolds connect those scaffold
    groups, preventing both molecule and scaffold leakage.
    """
    fractions.validate()
    required = {molecule_column, scaffold_column}
    missing = required - set(frame.columns)
    if missing:
        raise SplitError(f"Missing global split columns: {sorted(missing)}")
    if frame[molecule_column].isna().any():
        missing_molecules = int(frame[molecule_column].isna().sum())
        raise SplitError(f"{missing_molecules} records lack a molecule identifier")

    relationships = frame[[molecule_column, scaffold_column]].copy()

    def normalize_scaffold(value: object) -> str | None:
        if pd.isna(value):
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    relationships["_scaffold_key"] = relationships[scaffold_column].map(normalize_scaffold)
    relationships = relationships.drop_duplicates([molecule_column, "_scaffold_key"])
    molecule_ids = relationships[molecule_column].drop_duplicates().tolist()

    parent = {molecule_id: molecule_id for molecule_id in molecule_ids}
    rank = {molecule_id: 0 for molecule_id in molecule_ids}

    def find(molecule_id: object) -> object:
        while parent[molecule_id] != molecule_id:
            parent[molecule_id] = parent[parent[molecule_id]]
            molecule_id = parent[molecule_id]
        return molecule_id

    def union(first_molecule: object, second_molecule: object) -> None:
        first_root = find(first_molecule)
        second_root = find(second_molecule)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            parent[first_root] = second_root
        elif rank[first_root] > rank[second_root]:
            parent[second_root] = first_root
        else:
            parent[second_root] = first_root
            rank[first_root] += 1

    first_molecule_for_scaffold: dict[str, object] = {}
    for molecule_id, scaffold_key in relationships[[molecule_column, "_scaffold_key"]].itertuples(
        index=False,
        name=None,
    ):
        if scaffold_key is None:
            continue
        first_molecule = first_molecule_for_scaffold.get(scaffold_key)
        if first_molecule is None:
            first_molecule_for_scaffold[scaffold_key] = molecule_id
        else:
            union(first_molecule, molecule_id)

    component_roots = {molecule_id: find(molecule_id) for molecule_id in molecule_ids}
    molecule_meta = pd.DataFrame(
        {
            molecule_column: molecule_ids,
            "_scaffold_component": [
                f"component::{component_roots[molecule_id]}" for molecule_id in molecule_ids
            ],
        }
    )
    molecule_meta["split"] = scaffold_split(
        molecule_meta,
        scaffold_column="_scaffold_component",
        seed=seed,
        fractions=fractions,
    )
    result = frame.merge(
        molecule_meta[[molecule_column, "split"]],
        on=molecule_column,
        how="left",
        validate="many_to_one",
    )
    if result["split"].isna().any():
        unassigned = int(result["split"].isna().sum())
        raise SplitError(f"{unassigned} records could not be assigned to a split")
    assert_no_group_leakage(result, molecule_column, "split")
    assert_no_group_leakage(result, scaffold_column, "split")
    return result


def assert_no_group_leakage(
    frame: pd.DataFrame,
    group_column: str,
    split_column: str,
) -> None:
    """Check that each valid group appears in only one split."""
    required = {group_column, split_column}
    missing = required - set(frame.columns)
    if missing:
        raise SplitError(f"Missing leakage-check columns: {sorted(missing)}")
    if frame[split_column].isna().any():
        missing_assignments = int(frame[split_column].isna().sum())
        raise SplitError(f"{missing_assignments} records lack a split assignment")
    normalized_groups = frame[group_column].astype("string").str.strip()
    valid_groups = normalized_groups.notna() & normalized_groups.ne("")
    observed = frame.loc[valid_groups, [group_column, split_column]]
    if observed.empty:
        return
    counts = observed.groupby(group_column, dropna=False)[split_column].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        examples = leaked.head(10).index.astype(str).tolist()
        raise SplitError(
            f"{group_column} leakage across splits for {len(leaked)} groups. "
            f"Examples: {examples}"
        )


def repeated_scaffold_assignments(
    frame: pd.DataFrame,
    seeds: Sequence[int],
    scaffold_column: str = "scaffold",
    fractions: SplitFractions = SplitFractions(),
) -> pd.DataFrame:
    output = frame.copy()
    for seed in seeds:
        output[f"split_seed_{seed}"] = scaffold_split(
            frame, scaffold_column=scaffold_column, seed=seed, fractions=fractions
        )
    return output
