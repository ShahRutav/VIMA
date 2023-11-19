from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

from .discrete.distributions import CategoricalNet, MultiCategoricalNet
from .continuous import ContinuousHead, GMMHead


class MixedActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_types: dict[str, str],
        action_dims: dict[str, int | list[int]],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
        use_gmm: bool = False,
        gmm_num_components: int = 4,
    ):
        assert set(action_types.keys()) == set(action_dims.keys())
        super().__init__()

        self._decoders = nn.ModuleDict()
        for k, v in action_dims.items():
            if action_types[k] == "discrete":
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
            elif action_types[k] == "continuous":
                assert isinstance(v, int)
                if use_gmm:
                    self._decoders[k] = GMMHead(
                        input_dim,
                        num_gaussians=gmm_num_components,
                        action_dim=v,
                        hidden_dim=hidden_dim,
                        hidden_depth=hidden_depth,
                        activation=activation,
                        norm_type=norm_type,
                    )
                else:
                    self._decoders[k] = ContinuousHead(
                        input_dim,
                        action_dim=v,
                        hidden_dim=hidden_dim,
                        hidden_depth=hidden_depth,
                        activation=activation,
                        norm_type=norm_type,
                    )
            else:
                raise ValueError(f"Invalid action_types value: {v}")

    def forward(self, x: torch.Tensor):
        return {k: v(x) for k, v in self._decoders.items()}
