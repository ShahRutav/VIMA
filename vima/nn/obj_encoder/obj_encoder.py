import torch
import torch.nn as nn

from .vit import ViTEncoder
from ..utils import build_mlp


class ObjEncoder(nn.Module):
    bbox_max_h = 128
    bbox_max_w = 256

    def __init__(
        self,
        *,
        transformer_emb_dim: int,
        views: list[str],
        vit_output_dim: int = 512,
        vit_resolution: int,
        vit_patch_size: int,
        vit_width: int,
        vit_layers: int,
        vit_heads: int,
        bbox_mlp_hidden_dim: int,
        bbox_mlp_hidden_depth: int,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = transformer_emb_dim

        self.cropped_img_encoder = ViTEncoder(
            output_dim=vit_output_dim,
            resolution=vit_resolution,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
        )

        self.bbox_mlp = nn.ModuleDict(
            {
                view: build_mlp(
                    4,
                    hidden_dim=bbox_mlp_hidden_dim,
                    hidden_depth=bbox_mlp_hidden_depth,
                    output_dim=bbox_mlp_hidden_dim,
                )
                for view in views
            }
        )

        self.pre_transformer_layer = nn.ModuleDict(
            {
                view: nn.Linear(
                    self.cropped_img_encoder.output_dim + bbox_mlp_hidden_dim,
                    transformer_emb_dim,
                )
                for view in views
            }
        )

    def forward(
        self,
        cropped_img,
        bbox,
        mask,
    ):
        """
        out: (..., n_objs * n_views, E)
        """
        img_feats = {
            view: self.cropped_img_encoder(cropped_img[view]) for view in self._views
        }
        # normalize bbox
        bbox = {view: bbox[view].float() for view in self._views}
        _normalizer = torch.tensor(
            [self.bbox_max_w, self.bbox_max_h, self.bbox_max_h, self.bbox_max_w],
            dtype=bbox[self._views[0]].dtype,
            device=bbox[self._views[0]].device,
        )
        bbox = {view: bbox[view] / _normalizer for view in self._views}
        bbox = {view: self.bbox_mlp[view](bbox[view]) for view in self._views}

        in_feats = {
            view: self.pre_transformer_layer[view](
                torch.concat([img_feats[view], bbox[view]], dim=-1)
            )
            for view in self._views
        }
        out = torch.concat([in_feats[view] for view in self._views], dim=-2)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim
