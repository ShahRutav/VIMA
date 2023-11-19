from __future__ import annotations

import torch
import torch.nn as nn
import enlight.utils as U

from vima.data.constants import N_OBJS


class ObjNameEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        is_lang_feat: bool = True,
        lang_feat_dim: int | None = None,
    ):
        super().__init__()
        self._embed_layer = (
            nn.Identity()
            if is_lang_feat
            else nn.Embedding(
                num_embeddings=N_OBJS,
                embedding_dim=embed_dim,
            )
        )
        if is_lang_feat and lang_feat_dim != embed_dim:
            self._lang_feat_layer = nn.Linear(lang_feat_dim, embed_dim)
        else:
            self._lang_feat_layer = None
        self._is_lang_feat = is_lang_feat
        self._lang_feat_dim = lang_feat_dim
        self._embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        """
        int64 if not lang_feat, else float32 with last_dim = lang_feat_dim
        """
        self._check_input(x)
        x = self._embed_layer(x)
        if self._is_lang_feat and self._lang_feat_layer is not None:
            x = self._lang_feat_layer(x)
        return x

    @property
    def output_dim(self):
        return self._embed_dim

    @U.call_once
    def _check_input(self, x: torch.Tensor):
        if self._is_lang_feat:
            assert x.dtype in [torch.float32, torch.float16]
            assert x.shape[-1] == self._lang_feat_dim
        else:
            assert x.dtype == torch.int64
            assert torch.all(x >= 0) and torch.all(x < N_OBJS)
