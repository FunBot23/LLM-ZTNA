"""Two-layer GraphSAGE node classifier.
CPU inference is instant even for a few hundred nodes per window. Node-level output:
one logit per node, apply sigmoid for p(suspicious).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGENodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index).relu()
        return self.head(h).squeeze(-1)  # (N,) logits
