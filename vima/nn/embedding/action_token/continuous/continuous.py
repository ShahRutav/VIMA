from __future__ import annotations

import torch
import torch.nn as nn
from enlight.nn.mlp import build_mlp


class ContinuousActionEmbedding(nn.Module):
    def __init__(
        self, output_dim: int, *, input_dim: int, hidden_dim: int, hidden_depth: int
    ):
        super().__init__()

        self._layer = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        return self._layer(x)
