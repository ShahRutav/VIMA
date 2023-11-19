from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import enlight.utils as U
from enlight.nn.mlp import build_mlp
from enlight.learn import default_optimizer_groups

from .rgb import ResNetEncoder, ViTEncoder, ViTEncoderRectangular, GatoViTEncoder
from .set_transformer import ObjectsSetTransformer
from .perceiver import ObjectsPerceiverEncoder

import vima

class MultiViewObjectEncoder(nn.Module):
    bbox_max_h = 128
    bbox_max_w = 256

    def __init__(
        self,
        *,
        transformer_emb_dim: int,
        transformer_type: Literal["set", "perceiver"],
        transformer_num_queries: int = 1,
        views: list[str],
        set_transformer_num_heads: int = 4,
        set_transformer_layer_norm: bool = True,
        set_transformer_xattn_prompt: bool = False,
        set_transformer_xattn_n_heads: int | None = None,
        set_transformer_xattn_ff_expanding: int | None = None,
        set_transformer_xattn_detach_qk: bool | None = None,
        set_transformer_xattn_n_positions: int | None = None,
        perceiver_num_blocks: int | None = None,
        perceiver_num_self_attends_per_block: int | None = None,
        perceiver_num_self_attention_heads: int | None = None,
        perceiver_num_cross_attention_heads: int | None = None,
        perceiver_attention_probs_dropout_prob: float | None = None,
        img_encoder_output_dim: int = 512,
        img_encoder_type: Literal["resnet", "vit"] = "resnet",
        resnet_name: str | None = None,
        use_pretrained_resnet: bool = True,
        freeze_resnet: bool = True,
        resnet_add_top_mlp: bool = False,
        resnet_top_mlp_hidden_depth: int | None = None,
        vit_resolution: int | None = None,
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        add_bbox_mlp: bool = False,
        bbox_mlp_hidden_dim: int | None = None,
        bbox_mlp_hidden_depth: int | None = None,
    ):
        super().__init__()

        assert transformer_type in ["set", "perceiver"]
        if transformer_type == "set":
            assert set_transformer_num_heads is not None
            assert set_transformer_layer_norm is not None
        elif transformer_type == "perceiver":
            assert perceiver_num_blocks is not None
            assert perceiver_num_self_attends_per_block is not None
            assert perceiver_num_self_attention_heads is not None
            assert perceiver_num_cross_attention_heads is not None
            assert perceiver_attention_probs_dropout_prob is not None

        assert img_encoder_type in ["resnet", "vit"]
        if img_encoder_type == "resnet":
            assert resnet_name is not None
            assert use_pretrained_resnet is not None
            assert freeze_resnet is not None
            assert resnet_add_top_mlp is not None
            assert resnet_top_mlp_hidden_depth is not None
        elif img_encoder_type == "vit":
            assert vit_resolution is not None
            assert vit_patch_size is not None
            assert vit_width is not None
            assert vit_layers is not None
            assert vit_heads is not None

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = transformer_emb_dim

        # all views share the same rgb encoder
        if img_encoder_type == "resnet":
            self.cropped_img_encoder = ResNetEncoder(
                resnet_name=resnet_name,
                output_dim=img_encoder_output_dim,
                use_pretrained=use_pretrained_resnet,
                freeze=freeze_resnet,
                add_top_mlp=resnet_add_top_mlp,
                top_mlp_hidden_depth=resnet_top_mlp_hidden_depth,
            )
        elif img_encoder_type == "vit":
            self.cropped_img_encoder = ViTEncoder(
                output_dim=img_encoder_output_dim,
                resolution=vit_resolution,
                patch_size=vit_patch_size,
                width=vit_width,
                layers=vit_layers,
                heads=vit_heads,
            )

        # different views have different set transformers
        if transformer_type == "set":
            self.obj_transformers = nn.ModuleDict(
                {
                    view: ObjectsSetTransformer(
                        transformer_emb_dim,
                        num_heads=set_transformer_num_heads,
                        num_queries=transformer_num_queries,
                        layer_norm=set_transformer_layer_norm,
                        xattn_prompt=set_transformer_xattn_prompt,
                        xattn_n_heads=set_transformer_xattn_n_heads,
                        xattn_ff_expanding=set_transformer_xattn_ff_expanding,
                        xattn_detach_qk=set_transformer_xattn_detach_qk,
                        xattn_n_positions=set_transformer_xattn_n_positions,
                    )
                    for view in views
                }
            )
        elif transformer_type == "perceiver":
            self.obj_transformers = nn.ModuleDict(
                {
                    view: ObjectsPerceiverEncoder(
                        transformer_emb_dim,
                        num_latents=transformer_num_queries,
                        num_blocks=perceiver_num_blocks,
                        num_self_attends_per_block=perceiver_num_self_attends_per_block,
                        num_self_attention_heads=perceiver_num_self_attention_heads,
                        num_cross_attention_heads=perceiver_num_cross_attention_heads,
                        attention_probs_dropout_prob=perceiver_attention_probs_dropout_prob,
                    )
                    for view in views
                }
            )
        self._xf_type = transformer_type

        # different views have different bbox mlp if used
        if add_bbox_mlp:
            assert bbox_mlp_hidden_dim is not None
            assert bbox_mlp_hidden_depth is not None
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
        else:
            self.bbox_mlp = nn.ModuleDict({view: nn.Identity() for view in views})

        self.pre_transformer_layer = nn.ModuleDict(
            {
                view: nn.Linear(
                    self.cropped_img_encoder.output_dim + bbox_mlp_hidden_dim
                    if add_bbox_mlp
                    else 4,
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
        *,
        set_xf_xattn_prompt: bool = False,
        set_xf_xattn_prompt_tokens: torch.Tensor | None = None,
        set_xf_xattn_prompt_mask: torch.Tensor | None = None,
        set_xf_xattn_prompt_tokens_batch_first: bool | None = None,
    ):
        """
        out: (..., num_queries, E)
        """
        self._check_input(cropped_img, bbox, mask)
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
        if self._xf_type == "set":
            out_feats = {
                view: self.obj_transformers[view](
                    x=in_feats[view],
                    mask=mask[view],
                    xattn_prompt=set_xf_xattn_prompt,
                    prompt_tokens=set_xf_xattn_prompt_tokens,
                    prompt_mask=set_xf_xattn_prompt_mask,
                    prompt_tokens_batch_first=set_xf_xattn_prompt_tokens_batch_first,
                )
                for view in self._views
            }
        elif self._xf_type == "perceiver":
            out_feats = {
                view: self.obj_transformers[view](
                    x=in_feats[view],
                    mask=mask[view],
                )
                for view in self._views
            }
        else:
            raise ValueError(f"Unknown transformer type {self._xf_type}")
        out = torch.concat([out_feats[view] for view in self._views], dim=-1)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim * len(self._views)

    @U.call_once
    def _check_input(self, cropped_img, bbox, mask):
        assert isinstance(cropped_img, dict) or isinstance(cropped_img, U.DataDict)
        assert isinstance(bbox, dict) or isinstance(bbox, U.DataDict)
        assert isinstance(mask, dict) or isinstance(mask, U.DataDict)

        assert (
            set(cropped_img.keys())
            == set(bbox.keys())
            == set(mask.keys())
            == set(self._views)
        )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            img_encoder_param_group,
            img_encoder_param_ids,
        ) = self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        obj_xf_param_groups, obj_xf_param_ids = {}, {}
        for view in self._views:
            obj_xf_param_group, obj_xf_param_id = self.obj_transformers[
                view
            ].get_optimizer_groups(weight_decay, lr_layer_decay, lr_scale)
            obj_xf_param_groups[view] = obj_xf_param_group
            obj_xf_param_ids[view] = obj_xf_param_id
        obj_xf_param_groups = [p for v in obj_xf_param_groups.values() for p in v]
        obj_xf_param_ids = [p for v in obj_xf_param_ids.values() for p in v]

        other_param_group, other_param_id = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "bbox_mlp.*",
                "pre_transformer_layer.*",
            ],
            exclude_filter=lambda name, p: id(p)
            in img_encoder_param_ids + obj_xf_param_ids,
        )
        return (
            img_encoder_param_group + obj_xf_param_groups + other_param_group,
            img_encoder_param_ids + obj_xf_param_ids + other_param_id,
        )


class MultiViewRGBEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        use_mvp_vitb: bool = False,
        mvp_vitb_ckpt: str | None = None,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = emb_dim

        self.cropped_img_encoder = ViTEncoderRectangular(
            img_size=img_size,
            output_dim=emb_dim,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
            use_mvp_vitb=use_mvp_vitb,
            mvp_vitb_ckpt=mvp_vitb_ckpt,
        )

    def forward(
        self,
        rgb,
    ):
        self._check_input(rgb)
        img_feats = {view: self.cropped_img_encoder(rgb[view]) for view in self._views}
        out = torch.concat([img_feats[view] for view in self._views], dim=-1)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim * len(self._views)

    @U.call_once
    def _check_input(self, rgb):
        assert isinstance(rgb, dict) or isinstance(rgb, U.DataDict)

        assert set(rgb.keys()) == set(self._views)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )


class GatoMultiViewRGBEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        use_mvp_vitb: bool = False,
        mvp_vitb_ckpt: str | None = None,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self.output_dim = emb_dim

        self.cropped_img_encoder = GatoViTEncoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
            output_dim=emb_dim,
            use_mvp_vitb=use_mvp_vitb,
            mvp_vitb_ckpt=mvp_vitb_ckpt,
        )

    def forward(
        self,
        rgb,
    ):
        """
        input: (..., 3, H, W)
        output: (..., L * n_views, E)
        """
        self._check_input(rgb)
        img_feats = {
            view: self.cropped_img_encoder(rgb[view]) for view in self._views
        }  # dict of (..., L, E)
        out = torch.concat(
            [img_feats[view] for view in self._views], dim=-2
        )  # (..., L * n_views, E)
        return out

    @U.call_once
    def _check_input(self, rgb):
        assert isinstance(rgb, dict) or isinstance(rgb, U.DataDict)

        assert set(rgb.keys()) == set(self._views)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )

    @property
    def img_patch_len(self):
        return self.cropped_img_encoder.vit.img_patch_len * len(self._views)


class MultiViewObjectPoolingEncoder(nn.Module):
    bbox_max_h = 128
    bbox_max_w = 256

    def __init__(
        self,
        *,
        transformer_emb_dim: int,
        transformer_num_queries: int = 1,
        views: list[str],
        img_encoder_output_dim: int = 512,
        img_encoder_type: Literal["resnet", "vit"] = "resnet",
        resnet_name: str | None = None,
        use_pretrained_resnet: bool = True,
        freeze_resnet: bool = True,
        resnet_add_top_mlp: bool = False,
        resnet_top_mlp_hidden_depth: int | None = None,
        vit_resolution: int | None = None,
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        add_bbox_mlp: bool = False,
        bbox_mlp_hidden_dim: int | None = None,
        bbox_mlp_hidden_depth: int | None = None,
    ):
        super().__init__()

        assert img_encoder_type in ["resnet", "vit"]
        if img_encoder_type == "resnet":
            assert resnet_name is not None
            assert use_pretrained_resnet is not None
            assert freeze_resnet is not None
            assert resnet_add_top_mlp is not None
            assert resnet_top_mlp_hidden_depth is not None
        elif img_encoder_type == "vit":
            assert vit_resolution is not None
            assert vit_patch_size is not None
            assert vit_width is not None
            assert vit_layers is not None
            assert vit_heads is not None

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = transformer_emb_dim

        # all views share the same rgb encoder
        if img_encoder_type == "resnet":
            self.cropped_img_encoder = ResNetEncoder(
                resnet_name=resnet_name,
                output_dim=img_encoder_output_dim,
                use_pretrained=use_pretrained_resnet,
                freeze=freeze_resnet,
                add_top_mlp=resnet_add_top_mlp,
                top_mlp_hidden_depth=resnet_top_mlp_hidden_depth,
            )
        elif img_encoder_type == "vit":
            self.cropped_img_encoder = ViTEncoder(
                output_dim=img_encoder_output_dim,
                resolution=vit_resolution,
                patch_size=vit_patch_size,
                width=vit_width,
                layers=vit_layers,
                heads=vit_heads,
            )

        # different views have different bbox mlp if used
        if add_bbox_mlp:
            assert bbox_mlp_hidden_dim is not None
            assert bbox_mlp_hidden_depth is not None
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
        else:
            self.bbox_mlp = nn.ModuleDict({view: nn.Identity() for view in views})

        self.pre_transformer_layer = nn.ModuleDict(
            {
                view: nn.Linear(
                    self.cropped_img_encoder.output_dim + bbox_mlp_hidden_dim
                    if add_bbox_mlp
                    else 4,
                    transformer_emb_dim,
                )
                for view in views
            }
        )
        self._n_queries = transformer_num_queries

    def forward(
        self,
        cropped_img,
        bbox,
        mask,
    ):
        """
        out: (..., num_queries, E)
        """
        self._check_input(cropped_img, bbox, mask)
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
        out_feats = {
            view: (in_feats[view] * mask[view].unsqueeze(-1)).sum(dim=-2, keepdim=True)
            / mask[view].sum(dim=-1, keepdim=True).unsqueeze(-1)
            for view in self._views
        }
        out_feats = {
            view: torch.repeat_interleave(v, self._n_queries, dim=-2)
            for view, v in out_feats.items()
        }
        out = torch.concat([out_feats[view] for view in self._views], dim=-1)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim * len(self._views)

    @U.call_once
    def _check_input(self, cropped_img, bbox, mask):
        assert isinstance(cropped_img, dict) or isinstance(cropped_img, U.DataDict)
        assert isinstance(bbox, dict) or isinstance(bbox, U.DataDict)
        assert isinstance(mask, dict) or isinstance(mask, U.DataDict)

        assert (
            set(cropped_img.keys())
            == set(bbox.keys())
            == set(mask.keys())
            == set(self._views)
        )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            img_encoder_param_group,
            img_encoder_param_ids,
        ) = self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )

        other_param_group, other_param_id = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "bbox_mlp.*",
                "pre_transformer_layer.*",
            ],
            exclude_filter=lambda name, p: id(p) in img_encoder_param_ids,
        )
        return (
            img_encoder_param_group + other_param_group,
            img_encoder_param_ids + other_param_id,
        )


class MultiViewObjectAsTokenEncoder(nn.Module):
    bbox_max_h = 128
    bbox_max_w = 256

    def __init__(
        self,
        *,
        transformer_emb_dim: int,
        views: list[str],
        img_encoder_output_dim: int = 512,
        img_encoder_type: Literal["resnet", "vit"] = "resnet",
        resnet_name: str | None = None,
        use_pretrained_resnet: bool = True,
        freeze_resnet: bool = True,
        resnet_add_top_mlp: bool = False,
        resnet_top_mlp_hidden_depth: int | None = None,
        vit_resolution: int | None = None,
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        add_bbox_mlp: bool = False,
        bbox_mlp_hidden_dim: int | None = None,
        bbox_mlp_hidden_depth: int | None = None,
    ):
        super().__init__()

        assert img_encoder_type in ["resnet", "vit"]
        if img_encoder_type == "resnet":
            assert resnet_name is not None
            assert use_pretrained_resnet is not None
            assert freeze_resnet is not None
            assert resnet_add_top_mlp is not None
            assert resnet_top_mlp_hidden_depth is not None
        elif img_encoder_type == "vit":
            assert vit_resolution is not None
            assert vit_patch_size is not None
            assert vit_width is not None
            assert vit_layers is not None
            assert vit_heads is not None

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = transformer_emb_dim

        # all views share the same rgb encoder
        if img_encoder_type == "resnet":
            self.cropped_img_encoder = ResNetEncoder(
                resnet_name=resnet_name,
                output_dim=img_encoder_output_dim,
                use_pretrained=use_pretrained_resnet,
                freeze=freeze_resnet,
                add_top_mlp=resnet_add_top_mlp,
                top_mlp_hidden_depth=resnet_top_mlp_hidden_depth,
            )
        elif img_encoder_type == "vit":
            self.cropped_img_encoder = ViTEncoder(
                output_dim=img_encoder_output_dim,
                resolution=vit_resolution,
                patch_size=vit_patch_size,
                width=vit_width,
                layers=vit_layers,
                heads=vit_heads,
            )

        # different views have different bbox mlp if used
        if add_bbox_mlp:
            assert bbox_mlp_hidden_dim is not None
            assert bbox_mlp_hidden_depth is not None
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
        else:
            self.bbox_mlp = nn.ModuleDict({view: nn.Identity() for view in views})

        self.pre_transformer_layer = nn.ModuleDict(
            {
                view: nn.Linear(
                    self.cropped_img_encoder.output_dim + bbox_mlp_hidden_dim
                    if add_bbox_mlp
                    else 4,
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
        self._check_input(cropped_img, bbox, mask)
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

    @U.call_once
    def _check_input(self, cropped_img, bbox, mask):
        print(type(cropped_img))
        assert isinstance(cropped_img, dict) or isinstance(cropped_img, vima.utils.DataDict) or isinstance(cropped_img, U.DataDict)
        assert isinstance(bbox, dict) or isinstance(bbox, vima.utils.DataDict) or isinstance(bbox, U.DataDict)
        assert isinstance(mask, dict) or isinstance(mask, vima.utils.DataDict) or isinstance(mask, U.DataDict)

        assert (
            set(cropped_img.keys())
            == set(bbox.keys())
            == set(mask.keys())
            == set(self._views)
        )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            img_encoder_param_group,
            img_encoder_param_ids,
        ) = self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )

        other_param_group, other_param_id = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "bbox_mlp.*",
                "pre_transformer_layer.*",
            ],
            exclude_filter=lambda name, p: id(p) in img_encoder_param_ids,
        )
        return (
            img_encoder_param_group + other_param_group,
            img_encoder_param_ids + other_param_id,
        )


class MultiViewRGBPerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        use_mvp_vitb: bool = False,
        mvp_vitb_ckpt: str | None = None,
        perceiver_num_queries: int,
        perceiver_num_blocks: int,
        perceiver_num_self_attends_per_block: int,
        perceiver_num_self_attention_heads: int,
        perceiver_num_cross_attention_heads: int,
        perceiver_attention_probs_dropout_prob: float,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = emb_dim

        self.cropped_img_encoder = GatoViTEncoder(
            img_size=img_size,
            output_dim=emb_dim,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
            use_mvp_vitb=use_mvp_vitb,
            mvp_vitb_ckpt=mvp_vitb_ckpt,
        )
        self.peceiver = ObjectsPerceiverEncoder(
            emb_dim,
            num_latents=perceiver_num_queries,
            num_blocks=perceiver_num_blocks,
            num_self_attends_per_block=perceiver_num_self_attends_per_block,
            num_self_attention_heads=perceiver_num_self_attention_heads,
            num_cross_attention_heads=perceiver_num_cross_attention_heads,
            attention_probs_dropout_prob=perceiver_attention_probs_dropout_prob,
        )

    def forward(
        self,
        rgb,
    ):
        self._check_input(rgb)
        img_feats = {view: self.cropped_img_encoder(rgb[view]) for view in self._views}
        img_feats = torch.concat(
            [img_feats[view] for view in self._views], dim=-2
        )  # (B, L, E)
        masks = torch.ones(
            img_feats.shape[:2], device=img_feats.device, dtype=torch.bool
        )
        out = self.peceiver(img_feats, masks)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim

    @U.call_once
    def _check_input(self, rgb):
        assert isinstance(rgb, dict) or isinstance(rgb, U.DataDict)

        assert set(rgb.keys()) == set(self._views)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            img_encoder_pgroup,
            img_encoder_p_ids,
        ) = self.cropped_img_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        peceiver_pgroup, peceiver_p_ids = self.peceiver.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        return (
            img_encoder_pgroup + peceiver_pgroup,
            img_encoder_p_ids + peceiver_p_ids,
        )
