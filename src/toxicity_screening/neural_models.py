from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class FingerprintMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.25,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(x)).squeeze(-1)


class MultiTaskFingerprintMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tasks: int,
        shared_dims: Sequence[int] = (512, 256),
        head_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in shared_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(previous, head_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, 1),
                )
                for _ in range(tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return torch.cat([head(encoded) for head in self.heads], dim=1)
