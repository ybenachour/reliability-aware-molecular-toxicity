from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph_features import smiles_to_pyg


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError("Feature and label lengths differ")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class MaskedMultiTaskArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if labels.ndim != 2:
            raise ValueError("Multitask labels must be a 2D matrix")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        mask = ~np.isnan(labels)
        safe_labels = np.nan_to_num(labels, nan=0.0)
        self.labels = torch.as_tensor(safe_labels, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.mask[index]


class MolecularGraphDataset(Dataset):
    def __init__(self, smiles: Sequence[str], labels: np.ndarray):
        self.smiles = list(smiles)
        self.labels = np.asarray(labels)
        if len(self.smiles) != len(self.labels):
            raise ValueError("SMILES and label lengths differ")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int):
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return smiles_to_pyg(self.smiles[index], y=label)


class MaskedMolecularGraphDataset(Dataset):
    def __init__(self, smiles: Sequence[str], labels: np.ndarray):
        self.smiles = list(smiles)
        labels = np.asarray(labels, dtype=float)
        if labels.ndim != 2 or len(self.smiles) != len(labels):
            raise ValueError("SMILES and multitask labels must align as a 2D matrix")
        self.labels = np.nan_to_num(labels, nan=0.0)
        self.masks = (~np.isnan(labels)).astype(float)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int):
        data = smiles_to_pyg(self.smiles[index])
        data.y = torch.as_tensor(self.labels[index], dtype=torch.float32).view(1, -1)
        data.mask = torch.as_tensor(self.masks[index], dtype=torch.float32).view(1, -1)
        return data
