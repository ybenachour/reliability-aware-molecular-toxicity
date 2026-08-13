from __future__ import annotations

from collections import Counter
from typing import Iterable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_smiles(smiles: str, generic: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic and scaffold.GetNumAtoms():
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False) if scaffold.GetNumAtoms() else ""


def scaffold_counts(scaffolds: Iterable[str]) -> Counter[str]:
    return Counter(scaffolds)


def scaffold_novelty(scaffold: str, training_scaffolds: set[str]) -> bool:
    return scaffold not in training_scaffolds
