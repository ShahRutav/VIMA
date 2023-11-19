from __future__ import annotations

import torch
import torch.nn as nn
import enlight.utils as U
from transformers import AutoModel
from enlight.learn import transformer_lr_decay_optimizer_groups

from .gpt_with_adapter import GPTModelWithAdapter


class PreTrainedHFCasualLM(nn.Module):
    def __init__(
        self,
        lm_str: str,
        *,
        freeze: bool = True,
        adapter_type: str | None = None,
        adapter_positions: int | list[int] | None = None,
        adapter_n_layers: int | list[int] | None = None,
    ):
        if adapter_type is not None:
            assert adapter_positions is not None
            assert adapter_n_layers is not None

        super().__init__()
        if adapter_type is None:
            self.lm = AutoModel.from_pretrained(lm_str)
            if freeze:
                U.freeze_params(self.lm)
        else:
            self.lm = GPTModelWithAdapter.from_pretrained(
                lm_str,
                adapter_type=adapter_type,
                adapter_positions=adapter_positions,
                adapter_n_layers=adapter_n_layers,
                freeze_backbone=freeze,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        custom_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        batch_first: bool = False,
    ):
        """
        x: (L, B, E) if batch_first == False else (B, L, E)
        custom_mask: (B, L_tgt) or (B, 1, L_tgt) concurrently work with the causal mask
            because of self-attention, L_tgt = L
        """
        self._check_input(x, custom_mask, batch_first)
        if batch_first:
            B, L, E = x.shape
        else:
            L, B, E = x.shape
            x = x.transpose(0, 1)

        attention_mask = None
        if custom_mask is not None:
            if custom_mask.dim() == 3:
                custom_mask = custom_mask.squeeze(dim=1)
            attention_mask = custom_mask.float().contiguous()
        out = self.lm(
            inputs_embeds=x.contiguous(),
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state
        assert out.shape == (B, L, E)
        if not batch_first:
            out = out.transpose(0, 1)
        return out

    @U.call_once
    def _check_input(
        self, x, custom_mask: torch.Tensor | None = None, batch_first: bool = False
    ):
        assert x.dim() == 3
        assert x.dtype == torch.float32

        if batch_first:
            B, L, E = x.shape
        else:
            L, B, E = x.shape

        if custom_mask is not None:
            assert custom_mask.shape == (B, L) or custom_mask.shape == (
                B,
                1,
                L,
            ), f"Expect `custom_mask` to have shape of either ({B, 1, L}) or ({B, L}), but got {custom_mask.shape}"
            # a simple sanity check on the mask
            assert torch.all(
                custom_mask.sum(dim=-1) > 0
            ), "each source token should attend to at least one target token"
            assert custom_mask.dtype == torch.bool

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "lm.tokens_embed.*",
                "lm.positions_embed.*",
            ],
            block_sequence_name="lm.h",
            no_decay_filter=["lm.tokens_embed.*", "lm.positions_embed.*"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids
