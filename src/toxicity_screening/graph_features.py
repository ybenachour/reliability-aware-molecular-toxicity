from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem


ATOM_FEATURE_DIM = 15
BOND_FEATURE_DIM = 7


def _one_hot(value: Any, choices: list[Any]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    features: list[float] = []
    features += _one_hot(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35])
    features += [
        atom.GetDegree() / 4.0,
        atom.GetFormalCharge() / 3.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
        float(atom.HasProp("_ChiralityPossible")),
    ]
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
    ]


def smiles_to_pyg(smiles: str, y: torch.Tensor | None = None):
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch-geometric is required for graph models") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_indices: list[list[int]] = []
    edge_attributes: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attributes.extend([feat, feat])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    if y is not None:
        data.y = y
    return data
