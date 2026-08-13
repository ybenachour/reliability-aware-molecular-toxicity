from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _pyg_imports():
    try:
        from torch_geometric.nn import GATv2Conv, GINConv, global_mean_pool
    except ImportError as exc:
        raise ImportError("torch-geometric is required for GNN models") from exc
    return GATv2Conv, GINConv, global_mean_pool


class GINEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        _, GINConv, _ = _pyg_imports()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        previous = input_dim
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(previous, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            previous = hidden_dim
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        _, _, global_mean_pool = _pyg_imports()
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, edge_index)))
        return global_mean_pool(x, batch)


class GATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        GATv2Conv, _, _ = _pyg_imports()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        previous = input_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(previous, hidden_dim, heads=1, dropout=dropout))
            previous = hidden_dim
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        _, _, global_mean_pool = _pyg_imports()
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, edge_index)))
        return global_mean_pool(x, batch)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.2,
        architecture: str = "GIN",
    ) -> None:
        super().__init__()
        architecture = architecture.upper()
        if architecture == "GIN":
            self.encoder = GINEncoder(input_dim, hidden_dim, layers, dropout)
        elif architecture in {"GAT", "GATV2"}:
            self.encoder = GATEncoder(input_dim, hidden_dim, layers, dropout)
        else:
            raise ValueError(f"Unsupported graph architecture: {architecture}")
        self.output = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, data) -> torch.Tensor:
        embedding = self.encoder(data.x, data.edge_index, data.batch)
        return self.output(embedding).squeeze(-1)


class MultiTaskGraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tasks: int,
        hidden_dim: int = 128,
        layers: int = 4,
        head_dim: int = 64,
        dropout: float = 0.2,
        architecture: str = "GIN",
    ) -> None:
        super().__init__()
        architecture = architecture.upper()
        if architecture == "GIN":
            self.encoder = GINEncoder(input_dim, hidden_dim, layers, dropout)
        elif architecture in {"GAT", "GATV2"}:
            self.encoder = GATEncoder(input_dim, hidden_dim, layers, dropout)
        else:
            raise ValueError(f"Unsupported graph architecture: {architecture}")
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.encoder.output_dim, head_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, 1),
                )
                for _ in range(tasks)
            ]
        )

    def forward(self, data) -> torch.Tensor:
        embedding = self.encoder(data.x, data.edge_index, data.batch)
        return torch.cat([head(embedding) for head in self.heads], dim=1)
