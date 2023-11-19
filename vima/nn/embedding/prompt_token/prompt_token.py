from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from enlight.learn import default_optimizer_groups


class PromptTokenEmbedding(nn.Module):
    def __init__(
        self,
        pretrained_lm_str: str,
        *,
        freeze_pretrained: bool,
    ):
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_lm_str)
        embed_weight = model.get_input_embeddings().weight.data
        _emb_dim = embed_weight.shape[1]
        self._embed_layer = nn.Embedding.from_pretrained(
            embed_weight, freeze=freeze_pretrained
        )
        del model
        self.output_dim = _emb_dim

    def forward(self, x: torch.Tensor):
        """
        x: any shape
        """
        x = self._embed_layer(x)
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "_embed_layer.*",
            ],
        )
