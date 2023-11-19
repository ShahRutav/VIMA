from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ...attention import GPTBlock


class ActionGPTSequenceModeling(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_heads: int,
        n_blocks: int,
        position_embed: Literal["relative", "absolute"],
        embed_drop: float,
        attn_drop: float,
        residual_drop: float,
    ):
        super().__init__()

        assert embed_dim > 0 and n_heads > 0 and n_blocks > 0
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        assert position_embed in ["relative", "absolute"]
        # TODO: add relative position embedding
        assert position_embed == "absolute", "TODO"
        self.position_embed = position_embed

        if position_embed == "absolute":
            # Note that we simply use a large value 256 here for the max context length.
            # Trajectories are unlikely to reach this value.
            # Similar practice can be found in Gato appendix C.3.
            self.abs_pos_embed = nn.Parameter(torch.zeros(256, 1, embed_dim))

        self.embed_drop = nn.Dropout(embed_drop)

        if position_embed == "absolute":
            self.blocks = nn.Sequential(
                *[
                    GPTBlock(
                        embed_dim=embed_dim,
                        num_heads=n_heads,
                        attn_drop=attn_drop,
                        residual_drop=residual_drop,
                        window=256,  # same value as in absolute position embedding
                    )
                    for _ in range(n_blocks)
                ]
            )

        self.ln_f = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """
        TODO: Inherit from minGPT, may need to tune numbers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, custom_mask: torch.Tensor | None = None):
        """
        x: (L, B, E)
        """
        L, B, E = x.shape
        if self.position_embed == "absolute":
            assert L <= self.abs_pos_embed.shape[0]
            pos_embed = self.abs_pos_embed[:L]
            x += pos_embed
        x = self.embed_drop(x)
        x = self.blocks({"x": x, "custom_mask": custom_mask})["x"]
        x = self.ln_f(x)
        assert x.shape == (L, B, E)
        return x
