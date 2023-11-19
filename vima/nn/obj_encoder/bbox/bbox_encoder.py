from __future__ import annotations

import torch
import torch.nn as nn
import enlight.utils as U


class BBoxEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        n_max_bins: int = 256,
    ):
        super().__init__()
        self._embed_layer = nn.Embedding(
            num_embeddings=n_max_bins,
            embedding_dim=embed_dim,
        )
        self._embed_dim = embed_dim
        self._n_max_bins = n_max_bins

    def forward(self, x: torch.Tensor):
        """
        x: int64 (..., 4)
        """
        self._check_input(x)
        x = self._embed_layer(x)  # (..., 4, embed_dim)
        primitive_shape = x.shape[:-2]
        x = x.reshape(primitive_shape + (-1,))  # (..., 4 * embed_dim)
        return x

    @property
    def output_dim(self):
        return self._embed_dim * 4

    @U.call_once
    def _check_input(self, x: torch.Tensor):
        assert x.dtype == torch.int64
        assert x.shape[-1] == 4
        assert torch.all(x >= 0) and torch.all(x < self._n_max_bins)
