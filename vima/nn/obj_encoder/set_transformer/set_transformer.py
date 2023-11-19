from __future__ import annotations

from typing import Literal
import math

import torch
import torch.nn as nn
import enlight.utils as U
from einops import rearrange
from enlight.learn import default_optimizer_groups

from ...sequence_modeling.xattn_gpt.xattn import XAttention


class SetAttention(nn.Module):
    """
    "MAB" in the original paper
    """

    def __init__(
        self,
        dim_Q,
        dim_K,
        dim_V,
        num_heads,
        layer_norm=False,
        fp32_logits: bool = True,
    ):
        """
        Args:
            identity_key: do not transform K, use nn.Identity(), useful for attention
              pooling where key is the original features and we don't want to transform it.
              See CoCa paper: https://arxiv.org/abs/2205.01917
        """
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        assert self.dim_V % self.num_heads == 0
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if layer_norm:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        else:
            self.ln0 = nn.Identity()
            self.ln1 = nn.Identity()
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.act = nn.ReLU(inplace=True)
        self._fp32_logits = fp32_logits

    def forward(self, Q, K, mask=None):
        """
        mask: if not none, should be (B, L_src, L_trg)
        """
        if mask is not None:
            assert mask.shape[0] == Q.shape[0]
            assert mask.shape[1] == Q.shape[1]
            assert mask.shape[2] == K.shape[1]
            # check valid mask
            assert mask.dtype == torch.bool
            assert torch.all(
                mask.sum(dim=2) > 0
            ), "each source token should attend to at least one target token"
            # repeat mask num_heads times
            mask = torch.cat([mask] * self.num_heads, 0)
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if self._fp32_logits:
            Q_ = Q_.to(torch.float32)
            K_ = K_.to(torch.float32)
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            A.masked_fill_(mask == 0, -float("inf"))
        A = torch.softmax(A, 2)
        A = A.to(V_.dtype)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + self.act(self.fc_o(O))
        O = self.ln1(O)
        return O


class PoolingSetAttention(nn.Module):
    """
    "PMA" in the original paper
    """

    def __init__(
        self,
        dim,
        num_heads,
        pool_type: Literal["avg", "concat", "none", None] = None,
        layer_norm=False,
    ):
        """
        Args:
            pool_type: 'avg', 'concat', or None
              - 'avg': average pooling, returns [B, dim]
              - 'max': max pooling, returns [B, dim]
              - 'concat': concatenate the pooled features, returns [B, num_queries*dim]
              - None: don't pool and returns [B, num_queries, dim]
        """
        super().__init__()
        assert pool_type in ["avg", "concat", "none", "max", None]
        self._pool_type = pool_type
        self.mab = SetAttention(
            dim,
            dim,
            dim,
            num_heads=num_heads,
            layer_norm=layer_norm,
        )

    def forward(self, *, S, X, mask=None):
        O = self.mab(S, X, mask)
        if self._pool_type == "avg":
            return O.mean(dim=1)
        elif self._pool_type == "max":
            return O.max(dim=1)[0]
        elif self._pool_type == "concat":
            return rearrange(O, "b q d -> b (q d)")
        elif self._pool_type in ["none", None]:
            return O
        else:
            raise ValueError(f"Unknown pool_type: {self._pool_type}")


class ObjectsSetTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 4,
        num_queries: int = 1,
        layer_norm: bool = True,
        xattn_prompt: bool = False,
        xattn_n_heads: int | None = None,
        xattn_n_positions: int | None = None,
        xattn_ff_expanding: int | None = None,
        xattn_detach_qk: bool | None = None,
    ):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.S)
        self.pma = PoolingSetAttention(
            dim=embed_dim,
            num_heads=num_heads,
            pool_type=None,
            layer_norm=layer_norm,
        )
        self._num_queries = num_queries
        self._embed_dim = embed_dim

        self.prompt_xattention = None
        if xattn_prompt:
            assert xattn_n_heads is not None
            assert xattn_ff_expanding is not None
            assert xattn_detach_qk is not None
            self.prompt_xattention = XAttention(
                embed_dim,
                num_heads=xattn_n_heads,
                ff_expanding=xattn_ff_expanding,
                kv_n_positions=xattn_n_positions,
                detach_qk=xattn_detach_qk,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        xattn_prompt: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        prompt_tokens_batch_first: bool | None = None,
    ):
        """
        x: float32, (B, L_obj, E)
        mask: bool, (B, L_obj)

        out: (B, num_queries, E)
        """
        self._check_input(x, mask)
        queries = self.S.repeat(x.size(0), 1, 1)  # (B, num_queries, E)
        if xattn_prompt:
            assert self.prompt_xattention is not None
            assert prompt_tokens is not None
            assert prompt_mask is not None
            assert prompt_tokens_batch_first is not None
            assert prompt_tokens.dim() == 3
            assert prompt_mask.dim() == 2
            if not prompt_tokens_batch_first:
                prompt_tokens = prompt_tokens.transpose(0, 1)
            assert prompt_tokens.shape[0] == x.shape[0]  # (B, L_prompt, E)
            queries = self.prompt_xattention(
                q=queries,  # (B, num_queries, E)
                kv=prompt_tokens,  # (B, L_prompt, E)
                attention_mask=prompt_mask,  # (B, L_prompt)
                kv_position_ids=None,
            )  # (B, num_queries, E)

        # construct mask
        mask = mask.unsqueeze(1)  # (B, 1, L_obj)
        mask = mask.repeat(1, self._num_queries, 1)  # (B, num_queries, L_obj)
        out = self.pma(S=queries, X=x, mask=mask)  # (B, num_queries, E)
        return out

    @property
    def output_dim(self):
        return self._embed_dim

    @U.call_once
    def _check_input(self, x: torch.Tensor, mask: torch.Tensor):
        assert x.dim() == 3
        assert x.dtype in [torch.float32, torch.float16]
        assert mask.dtype == torch.bool
        assert x.dim() == mask.dim() + 1
        assert x.shape[:-1] == mask.shape

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        no_decay_filter = ["S"]
        if self.prompt_xattention is not None:
            no_decay_filter += [
                "prompt_xattention.layernorm.*",
                "prompt_xattention.kv_positions_embed.*",
            ]
        optim_groups, all_param_ids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=no_decay_filter,
        )
        return optim_groups, all_param_ids
