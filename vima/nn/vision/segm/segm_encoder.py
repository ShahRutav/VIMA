from __future__ import annotations

import torch
import torch.nn as nn

from vima.data.constants import N_OBJS, OBJ_NAME_TO_ID


class SegmEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        learn_padding: bool = False,
        is_lang_feat: bool = False,
        lang_feat_dim: int | None = None,
    ):
        super().__init__()
        self._embed_layer = (
            nn.Identity()
            if is_lang_feat
            else nn.Embedding(
                num_embeddings=N_OBJS,
                embedding_dim=embed_dim,
                padding_idx=None if learn_padding else OBJ_NAME_TO_ID["padding"],
            )
        )
        if is_lang_feat and lang_feat_dim != embed_dim:
            self._lang_feat_layer = nn.Linear(lang_feat_dim, embed_dim)
        else:
            self._lang_feat_layer = None
        self._is_lang_feat = is_lang_feat
        self._lang_feat_dim = lang_feat_dim
        self._input_checked = [] #Once()
        self._embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        """
        int64 (B, T, H, W) if not lang_feat, else float32 (B, T, H, W, lang_feat_dim)
        """
        if self._input_checked():
            if self._is_lang_feat:
                assert len(x.shape) == 5
                assert x.shape[-1] == self._lang_feat_dim
            else:
                assert len(x.shape) == 4
                assert torch.all(x >= 0) and torch.all(x < N_OBJS)

        x = self._embed_layer(x)  # (B, T, H, W, embed_dim)
        if self._is_lang_feat and self._lang_feat_layer is not None:
            x = self._lang_feat_layer(x)
        return x

    @property
    def output_dim(self):
        return self._embed_dim
