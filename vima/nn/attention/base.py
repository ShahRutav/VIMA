import math
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["CausalSelfAttention", "GPTBlock"]


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, embed_dim, num_heads, window, attn_drop=0.1, residual_drop=0.1):
        """
        embed_dim: int, embedding dimension, i.e., d_k
        num_heads: int, number of self-attention heads
        window: int, max context length
        attn_drop: float, attention dropout probability
        residual_drop: float, residual dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.residual_drop = nn.Dropout(residual_drop)
        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(window, window)).view(1, 1, window, window),
            persistent=False,
        )
        self.n_head = num_heads

        self.log_attention: bool = False
        self._logged_attention_matrix = None

    def forward(self, x, custom_mask=None):
        """
        x: [T, B, feat]
        custom_mask: [B, 1, T] or [B, T, T]. A binary custom mask applied concurrently with the causal mask.
        """
        T, B, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # all (nh, B, T, hs)
        k = self.key(x).view(T, B, self.n_head, C // self.n_head).transpose(0, 2)
        q = self.query(x).view(T, B, self.n_head, C // self.n_head).transpose(0, 2)
        v = self.value(x).view(T, B, self.n_head, C // self.n_head).transpose(0, 2)

        # causal self-attention; Self-attend: (nh, B, T, hs) x (nh, B, hs, T) -> (nh, B, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # masking
        if custom_mask is None:
            # if no custom mask is provided, we just do a causal masking
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            # custom_mask should be either [B, 1, T] or [B, T, T]
            assert custom_mask.shape == (B, 1, T) or custom_mask.shape == (
                B,
                T,
                T,
            ), f"Expect `custom_mask` to have shape of either ({B, 1, T}) or ({B, T, T}), but got {custom_mask.shape}"
            # a simple sanity check on the mask
            assert torch.all(
                custom_mask.sum(dim=2) > 0
            ), "each source token should attend to at least one target token"
            assert custom_mask.dtype == torch.bool
            B_mask, T_mask = custom_mask.shape[0], custom_mask.shape[-1]
            assert B_mask == B and T_mask == T
            custom_mask = custom_mask.unsqueeze(dim=0)  # (1, B, 1, T) or (1, B, T, T)
            customized_causal_mask = torch.logical_or(
                self.mask[:, :, :T, :T] == 0, custom_mask == 0
            )
            # check invalid mask
            assert torch.all(
                (customized_causal_mask.sum(dim=-1) / T) < 1
            ), "each source token should attend to at least one target token"
            att = att.masked_fill(customized_causal_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        # log attention matrix for analysis
        if self.log_attention:
            self._logged_attention_matrix = att.detach().clone().cpu().numpy()
        att = self.attn_drop(att)
        y = att @ v  # (nh, B, T, T) x (nh, B, T, hs) -> (nh, B, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(0, 2).contiguous().view(T, B, C)  # (T, B, nh*hs)
        # output projection
        y = self.residual_drop(self.proj(y))
        return y

    @property
    def attention_matrix(self):
        assert (
            self.log_attention
        ), "Only log attention matrix when `self.log_attention` is True"
        assert self._logged_attention_matrix is not None, "Must call `forward` first!"
        return self._logged_attention_matrix


class GPTBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, embed_dim, num_heads, window, attn_drop=0.1, residual_drop=0.1):
        """
        See docstring for `CausalSelfAttention`.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            residual_drop=residual_drop,
            window=window,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(residual_drop),
        )

    def forward(self, input_dict):
        x, custom_mask = input_dict["x"], input_dict["custom_mask"]
        x = x + self.attn(self.ln1(x), custom_mask=custom_mask)
        x = x + self.mlp(self.ln2(x))
        return {"x": x, "custom_mask": custom_mask}

    @property
    def attention_matrix(self):
        return self.attn.attention_matrix
