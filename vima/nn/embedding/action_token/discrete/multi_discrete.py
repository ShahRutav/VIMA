from __future__ import annotations

import torch
import torch.nn as nn


class MultiDiscreteActionEmbedding(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        num_embs: list[int],
        emb_dims: int | list[int],
    ):
        super().__init__()
        if isinstance(emb_dims, int):
            emb_dims = [emb_dims] * len(num_embs)
        elif isinstance(emb_dims, list):
            assert len(emb_dims) == len(num_embs)
        else:
            raise ValueError("emb_dims must be int or list")

        self._emb_layers = nn.ModuleList(
            [
                nn.Embedding(num_emb, emb_dim)
                for num_emb, emb_dim in zip(num_embs, emb_dims)
            ]
        )
        self._post_emb = (
            nn.Identity()
            if output_dim == sum(emb_dims)
            else nn.Linear(sum(emb_dims), output_dim)
        )
        self._output_dim = output_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x: torch.Tensor):
        """
        x: (..., n_actions) where n_actions == len(num_embs) == len(emb_dims)
        """
        assert x.shape[-1] == len(self._emb_layers)
        x = torch.cat(
            [emb(x[..., i].long()) for i, emb in enumerate(self._emb_layers)], dim=-1
        )  # (..., sum(emb_dims))
        x = self._post_emb(x)  # (..., output_dim)
        return x
