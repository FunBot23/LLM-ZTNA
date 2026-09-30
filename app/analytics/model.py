"""Small TabTransformer-style classifier.

  * Categorical features → each embedded as a token → self-attention over the small set of category tokens.
  * Continuous features → LayerNorm + concatenated with the pooled
    transformer output → MLP head → single sigmoid (p_malicious).
"""
from __future__ import annotations

import torch
from torch import nn


class TabTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_continuous: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cont_norm = nn.LayerNorm(num_continuous)
        self.head = nn.Sequential(
            nn.Linear(d_model + num_continuous, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, cat_ids: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        """
        cat_ids: (B, n_cat) long
        cont:    (B, n_cont) float
        returns: (B,) logits (apply sigmoid outside for probability)
        """
        # Embed each categorical column then stack into a sequence.
        tokens = torch.stack(
            [emb(cat_ids[:, i]) for i, emb in enumerate(self.cat_embeds)],
            dim=1,
        )  # (B, n_cat, d_model)

        transformed = self.transformer(tokens)  # (B, n_cat, d_model)
        pooled = transformed.mean(dim=1)  # (B, d_model)

        x = torch.cat([pooled, self.cont_norm(cont)], dim=1)
        return self.head(x).squeeze(-1)
