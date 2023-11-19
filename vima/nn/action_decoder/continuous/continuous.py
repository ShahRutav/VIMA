from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
from enlight.nn.mlp import build_mlp


class ContinuousHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
    ):
        super().__init__()

        self._layer = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor):
        return DummyContinuous(self._layer(x))


class DummyContinuous:
    def __init__(self, x: torch.Tensor):
        self._x = x

    def mode(self):
        return self._x

    def sample(self):
        return self._x

    def imitation_loss(self, actions, mask=None, reduction="mean"):
        """compute L2 loss along the last dimension"""
        assert actions.shape == self._x.shape
        if mask is not None:
            assert mask.shape == self._x.shape[:-1]

        loss = ((actions - self._x) ** 2).sum(-1)
        if mask is not None:
            loss *= mask
        if reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def imitation_accuracy(self, actions, mask=None, reduction="mean"):
        """compute L1 loss along the last dimension"""
        assert actions.shape == self._x.shape
        if mask is not None:
            assert mask.shape == self._x.shape[:-1]

        loss = (actions - self._x).abs().sum(-1)
        if mask is not None:
            loss *= mask
        if reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
