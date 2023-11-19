from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ..vision import resnet
from .segm import SegmEncoder
#from vimasim.utils import global_once


class VisionEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        # ------ rgb ------
        rgb_encoder_cls: str,
        rgb_encoder_kwargs: dict,
        # ------ segm ------
        segm_embed_dim: int,
        segm_learn_padding: bool = False,
        segm_is_lang_feat: bool = False,
        segm_lang_feat_dim: int | None = None,
    ):
        super(VisionEncoder, self).__init__()
        rgb_encoder_kwargs["output_dim"] = output_dim
        rgb_encoder_kwargs["in_channels"] = 3 + segm_embed_dim
        rgb_encoder_cls = getattr(resnet, rgb_encoder_cls)
        self._rgb_encoder = rgb_encoder_cls(**rgb_encoder_kwargs)
        self._segm_encoder = SegmEncoder(
            embed_dim=segm_embed_dim,
            learn_padding=segm_learn_padding,
            is_lang_feat=segm_is_lang_feat,
            lang_feat_dim=segm_lang_feat_dim,
        )
        self._output_dim = output_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, rgb: torch.Tensor, segm: torch.Tensor):
        """
        rgb: (L, B, 3, H, W), [0, 255]
        segm: (L, B, H, W) or (L, B, H, W, lang_feat_dim)
        """
        #if global_once("VisionEncoder:rgb_range"):
        #    assert rgb.max() > 2, "raw img should be between [0, 255]"
        rgb = rgb.float() / 255.0

        # encode segm
        segm = self._segm_encoder(segm)  # (L, B, H, W, embed_dim)
        segm = rearrange(segm, "L B H W C -> L B C H W")
        assert segm.shape[:2] == rgb.shape[:2]
        assert segm.shape[-2:] == rgb.shape[-2:]

        # concat rgb and segm
        x = torch.cat([rgb, segm], dim=2)  # (L, B, 3 + embed_dim, H, W)
        L, B = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])  # (L*B, 3 + embed_dim, H, W)
        x = self._rgb_encoder(x)  # (L * B, output_dim)
        x = x.reshape(L, B, -1)
        return x
