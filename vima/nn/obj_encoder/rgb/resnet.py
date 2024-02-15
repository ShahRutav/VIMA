from __future__ import annotations

import torch
import torch.nn as nn
from enlight.nn.vision.resnet import create_resnet
from enlight.nn.mlp import build_mlp
import enlight.utils as U
# from torchvision.models.resnet import model_urls
from torch.hub import load_state_dict_from_url

from vima.nn.constant import VIMA_IMG_MEAN, VIMA_IMG_STD


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        resnet_name: str,
        *,
        output_dim: int,
        use_pretrained: bool = True,
        freeze: bool = True,
        add_top_mlp: bool = False,
        top_mlp_hidden_depth: int | None = None,
    ):
        if add_top_mlp:
            assert top_mlp_hidden_depth is not None

        super().__init__()
        self.resnet = create_resnet(
            resnet_name, output_dim=output_dim, return_last_spatial_map=True
        )
        if use_pretrained:
            state_dict = load_state_dict_from_url(
                model_urls[resnet_name], progress=True
            )
            self.resnet.load_state_dict(state_dict, strict=False)
        if freeze:
            U.freeze_params(self.resnet)
        self._freeze = freeze

        self.output_dim = output_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if add_top_mlp:
            self.fc = build_mlp(
                512,
                hidden_dim=output_dim,
                output_dim=output_dim,
                hidden_depth=top_mlp_hidden_depth,
            )
        else:
            self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        if self._freeze:
            with torch.no_grad():
                x = self.resnet(x)
        else:
            x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(*leading_dim, self.output_dim)
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return NotImplementedError
