from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical as _OneHotCategorical
from enlight.nn.mlp import build_mlp


class GMMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_gaussians: int,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
    ):
        super().__init__()

        self._categorical_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_gaussians,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )

        self._mu_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_gaussians * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._sigma_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_gaussians * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._action_dim = action_dim
        self._num_gaussians = num_gaussians

    def forward(self, x: torch.Tensor):
        pi_logits = self._categorical_mlp(x)
        mean = self._mu_mlp(x)  # (..., num_gaussians * action_dim)
        log_sigma = self._sigma_mlp(x)  # (..., num_gaussians * action_dim)
        mean = mean.reshape(*mean.shape[:-1], self._num_gaussians, self._action_dim)
        log_sigma = log_sigma.reshape(
            *log_sigma.shape[:-1], self._num_gaussians, self._action_dim
        )

        assert pi_logits.shape[-1] == self._num_gaussians
        assert mean.shape[-2:] == (self._num_gaussians, self._action_dim)
        assert log_sigma.shape[-2:] == (self._num_gaussians, self._action_dim)
        return MixtureOfGaussian(pi_logits, mean, log_sigma)


class MixtureOfGaussian:
    def __init__(
        self,
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ):
        """
        pi_logits: (..., num_gaussians)
        mu: (..., num_gaussians, dim)
        log_sigma: (..., num_gaussians, dim)
        """
        assert pi_logits.dim() + 1 == mu.dim() == log_sigma.dim()
        assert pi_logits.shape[-1] == mu.shape[-2] == log_sigma.shape[-2]
        assert mu.shape == log_sigma.shape

        self.pi = OneHotCategorical(logits=pi_logits)
        self.gaussians = torch.distributions.Normal(mu, F.softplus(log_sigma))

    def mode(self):
        return (self.pi.mode.unsqueeze(-1) * self.gaussians.mean).sum(-2)

    def sample(self):
        return (self.pi.sample().unsqueeze(-1) * self.gaussians.sample()).sum(-2)

    def imitation_loss(self, actions, mask=None, reduction="mean"):
        """
        MLE loss
        """
        assert actions.dim() == self.gaussians.mean.dim() - 1
        assert actions.shape[-1] == self.gaussians.mean.shape[-1]
        assert actions.shape[:-1] == self.gaussians.mean.shape[:-2]
        if mask is not None:
            assert mask.shape == actions.shape[:-1]

        logp = self.gaussians.log_prob(actions.unsqueeze(-2)).sum(
            -1
        )  # (..., num_gaussians)
        loss = -torch.logsumexp(self.pi.logits + logp, dim=-1)  # (...)

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
        """
        L1 distance between mode and actions
        """
        assert actions.dim() == self.gaussians.mean.dim() - 1
        assert actions.shape[-1] == self.gaussians.mean.shape[-1]
        assert actions.shape[:-1] == self.gaussians.mean.shape[:-2]
        if mask is not None:
            assert mask.shape == actions.shape[:-1]

        loss = (actions - self.mode()).abs().sum(-1)
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


class OneHotCategorical(_OneHotCategorical):
    @property
    def mode(self):
        probs = self._categorical.probs
        mode = probs.argmax(axis=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)
