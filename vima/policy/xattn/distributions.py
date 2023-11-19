from __future__ import annotations
from typing import Literal, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Categorical",
    "CategoricalHead",
    "CategoricalNet",
    "MultiCategorical",
    "MultiCategoricalHead",
    "MultiCategoricalNet",
]


def _build_mlp_distribution_net(
    input_dim: int,
    *,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to have the
            Gaussian centered around 0.0 in the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    mlp = build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        weight_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type,
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        nn.init.orthogonal_(mlp[-1].weight, gain=last_layer_gain)
    return mlp


class Categorical(torch.distributions.Categorical):
    """
    Mostly interface changes, add mode() function, no real difference from Categorical
    """

    def mode(self):
        return self.logits.argmax(dim=-1)

    def imitation_loss(self, actions, reduction="mean"):
        """
        actions: groundtruth actions from expert
        """
        assert actions.dtype == torch.long
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            return F.cross_entropy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions.reshape(-1),
                reduction=reduction,
            )
        return F.cross_entropy(self.logits, actions, reduction=reduction)

    def imitation_accuracy(self, actions, mask=None, reduction="mean", scale_100=False):
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            if mask is not None:
                assert mask.ndim == 2
                assert self.logits.shape[:2] == mask.shape
            actions = actions.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
            return classify_accuracy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions,
                mask=mask,
                reduction=reduction,
                scale_100=scale_100,
            )
        return classify_accuracy(
            self.logits, actions, mask=mask, reduction=reduction, scale_100=scale_100
        )

    def random_actions(self):
        """
        Generate a completely random action, NOT the same as sample(), more like
        action_space.sample()
        """
        return torch.randint(
            low=0,
            high=self.logits.size(-1),
            size=self.logits.size()[:-1],
            device=self.logits.device,
        )


class CategoricalHead(nn.Module):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)


class CategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = CategoricalHead()

    def forward(self, x):
        return self.head(self.mlp(x))


# ---------------- MultiCategorical -----------------
class MultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits, action_dims: list[int]):
        assert logits.dim() >= 2, logits.shape
        super().__init__(batch_shape=logits[:-1], validate_args=False)
        self._action_dims = tuple(action_dims)
        assert logits.size(-1) == sum(
            self._action_dims
        ), f"sum of action dims {self._action_dims} != {logits.size(-1)}"
        self._dists = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dims, dim=-1)
        ]

    def log_prob(self, actions):
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=-1)

    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=-1) for dist in self._dists], dim=-1
        )

    def imitation_loss(self, actions, weights=None, reduction="mean"):
        """
        Args:
            actions: groundtruth actions from expert
            weights: weight the imitation loss from each component in MultiDiscrete
            reduction: "mean" or "none"

        Returns:
            one torch float
        """
        assert actions.dtype == torch.long
        assert actions.shape[-1] == len(self._action_dims)
        assert reduction in ["mean", "none"]
        if weights is None:
            weights = [1.0] * len(self._dists)
        else:
            assert len(weights) == len(self._dists)

        aggregate = sum if reduction == "mean" else list
        return aggregate(
            dist.imitation_loss(a, reduction=reduction) * w
            for dist, a, w in zip(self._dists, torch.unbind(actions, dim=-1), weights)
        )

    def imitation_accuracy(self, actions, mask=None, reduction="mean", scale_100=False):
        """
        Returns:
            a 1D tensor of accuracies between 0 and 1 as float
        """
        return [
            dist.imitation_accuracy(
                a, mask=mask, reduction=reduction, scale_100=scale_100
            )
            for dist, a in zip(self._dists, torch.unbind(actions, dim=-1))
        ]

    def random_actions(self):
        return torch.stack([dist.random_actions() for dist in self._dists], dim=-1)


class MultiCategoricalHead(nn.Module):
    def __init__(self, action_dims: list[int]):
        super().__init__()
        self._action_dims = tuple(action_dims)

    def forward(self, x: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(logits=x, action_dims=self._action_dims)

    def extra_repr(self) -> str:
        return f"action_dims={list(self._action_dims)}"


class MultiCategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy
        Split head, does not share the NN weights

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dims:
            net = _build_mlp_distribution_net(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=norm_type,
                last_layer_gain=last_layer_gain,
            )
            self.mlps.append(net)
        self.head = MultiCategoricalHead(action_dims)

    def forward(self, x):
        return self.head(torch.cat([mlp(x) for mlp in self.mlps], dim=-1))


def classify_accuracy(
    output,
    target,
    topk: int | list[int] | tuple[int] = 1,
    mask=None,
    reduction="mean",
    scale_100=False,
):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0
    Args:
        topk: if int, return a single acc. If tuple, return a tuple of accs
        mask: shape [batch_size,], binary mask of whether to include this sample or not
    """
    if isinstance(topk, int):
        topk = [topk]
        is_int = True
    else:
        is_int = False

    batch_size = target.size(0)
    assert output.size(0) == batch_size
    if mask is not None:
        assert mask.dim() == 1
        assert mask.size(0) == batch_size

    assert reduction in ["sum", "mean", "none"]
    if reduction != "mean":
        assert not scale_100, f"reduce={reduction} does not support scale_100=True"

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if mask is not None:
            correct = mask * correct

        mult = 100.0 if scale_100 else 1.0
        res = []
        for k in topk:
            correct_k = correct[:k].int().sum(dim=0)
            if reduction == "mean":
                if mask is not None:
                    # fmt: off
                    res.append(
                        float(correct_k.float().sum().mul_(mult / mask.sum().item()).item())
                    )
                    # fmt: on
                else:
                    res.append(
                        float(correct_k.float().sum().mul_(mult / batch_size).item())
                    )
            elif reduction == "sum":
                res.append(int(correct_k.sum().item()))
            elif reduction == "none":
                res.append(correct_k)
            else:
                raise NotImplementedError(f"Unknown reduce={reduction}")

    if is_int:
        assert len(res) == 1, "INTERNAL"
        return res[0]
    else:
        return res
