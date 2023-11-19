import torch
import torch.nn as nn
from enlight.nn.mlp import build_mlp


class CLIPAdapter(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        *,
        n_layers: int,
    ):
        super().__init__()

        self.adapter = build_mlp(
            input_dim=emb_dim,
            output_dim=emb_dim,
            hidden_dim=emb_dim,
            num_layers=n_layers,
            add_input_activation=False,
        )
        self.residual_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self, x: torch.Tensor):
        """
        x: (..., E)
        """
        res = torch.sigmoid(self.residual_weight)
        return res * x + (1.0 - res) * self.adapter(x)
