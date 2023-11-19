from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

from .distributions import CategoricalNet, MultiCategoricalNet


class DiscreteActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: dict[str, int | list[int]],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        super().__init__()

        self._decoders = nn.ModuleDict()
        for k, v in action_dims.items():
            if isinstance(v, int):
                self._decoders[k] = CategoricalNet(
                    input_dim,
                    action_dim=v,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain,
                )
            elif isinstance(v, list):
                self._decoders[k] = MultiCategoricalNet(
                    input_dim,
                    action_dims=v,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain,
                )
            else:
                raise ValueError(f"Invalid action_dims value: {v}")

    def forward(self, x: torch.Tensor):
        return {k: v(x) for k, v in self._decoders.items()}
