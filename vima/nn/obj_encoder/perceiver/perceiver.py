from __future__ import annotations

import torch
import torch.nn as nn
import enlight.utils as U
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverConfig,
    PerceiverModel,
)
from enlight.learn import transformer_lr_decay_optimizer_groups


class ObjectsPerceiverEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_latents: int,
        num_blocks: int,
        num_self_attends_per_block: int,
        num_self_attention_heads: int,
        num_cross_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()

        cfg = PerceiverConfig(
            d_model=embed_dim,
            d_latents=embed_dim,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_self_attention_heads=num_self_attention_heads,
            num_cross_attention_heads=num_cross_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.model = PerceiverModel(cfg)
        self.output_dim = embed_dim
        self._num_queries = num_latents

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: float32, (B, L_obj, E)
        mask: bool, (B, L_obj)

        out: (B, num_queries, E)
        """
        self._check_input(x, mask)
        out = self.model(inputs=x, attention_mask=mask).last_hidden_state
        return out

    @U.call_once
    def _check_input(self, x: torch.Tensor, mask: torch.Tensor):
        assert x.dim() == 3
        assert x.dtype == torch.float32
        assert mask.dtype == torch.bool
        assert x.dim() == mask.dim() + 1
        assert x.shape[:-1] == mask.shape

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return transformer_lr_decay_optimizer_groups(
            model=self.model,
            layer_0_params=["encoder.embeddings.*"],
            block_sequence_name="encoder.self_attends",
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            lr_layer_decay=lr_layer_decay,
        )
