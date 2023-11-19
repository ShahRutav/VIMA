from __future__ import annotations

import os
from collections import OrderedDict
from functools import partial

import torch
import numpy as np
from torch import nn
import enlight.utils as U
from enlight.learn import (
    default_optimizer_groups,
    check_optimizer_groups,
    rank_zero_info,
    transformer_lr_decay_optimizer_groups,
)

from vima.nn.constant import VIMA_IMG_MEAN, VIMA_IMG_STD


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.vit = VisionTransformer(
            resolution=resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
        )

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)
        x = x.view(*leading_dim, self.output_dim)
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return self.vit.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )


class ViTEncoderRectangular(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        use_mvp_vitb: bool = False,
        mvp_vitb_ckpt: str | None = None,
    ):
        super().__init__()

        self.output_dim = output_dim
        self._use_mvp_vitb = use_mvp_vitb
        if not use_mvp_vitb:
            self.vit = VisionTransformerRectangular(
                img_size=img_size,
                patch_size=patch_size,
                width=width,
                layers=layers,
                heads=heads,
                output_dim=output_dim,
            )
        else:
            assert mvp_vitb_ckpt is not None
            assert os.path.exists(mvp_vitb_ckpt)
            from ...mvp import MVPVisionTransformer

            self.vit = MVPVisionTransformer(
                img_size=(64, 128),
                patch_size=32,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.vit.load_state_dict(
                torch.load(mvp_vitb_ckpt, map_location="cpu"), strict=True
            )
            if output_dim != 768:
                self._post_mvp_vit_adapt_layer = nn.Linear(768, output_dim, bias=False)
            else:
                self._post_mvp_vit_adapt_layer = None

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)
        if self._use_mvp_vitb:
            x = x[:, 0]
            x = (
                self._post_mvp_vit_adapt_layer(x)
                if self._post_mvp_vit_adapt_layer is not None
                else x
            )
        x = x.view(*leading_dim, self.output_dim)
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return self.vit.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )


class GatoViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        use_mvp_vitb: bool = False,
        mvp_vitb_ckpt: str | None = None,
    ):
        super().__init__()

        self.output_dim = output_dim
        self._use_mvp_vitb = use_mvp_vitb
        if not use_mvp_vitb:
            self.vit = GatoVisionTransformerRectangular(
                img_size=img_size,
                patch_size=patch_size,
                width=width,
                layers=layers,
                heads=heads,
                output_dim=output_dim,
            )
        else:
            assert mvp_vitb_ckpt is not None
            assert os.path.exists(mvp_vitb_ckpt)
            from ...mvp import MVPVisionTransformer

            self.vit = MVPVisionTransformer(
                img_size=(64, 128),
                patch_size=32,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.vit.load_state_dict(
                torch.load(mvp_vitb_ckpt, map_location="cpu"), strict=True
            )
            if output_dim != 768:
                self._post_mvp_vit_adapt_layer = nn.Linear(768, output_dim, bias=False)
            else:
                self._post_mvp_vit_adapt_layer = None

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)  # (B, L, E)
        if self._use_mvp_vitb:
            x = x[:, 1:, :]  # (B, L, E)
            x = (
                self._post_mvp_vit_adapt_layer(x)
                if self._post_mvp_vit_adapt_layer is not None
                else x
            )
        x = x.view(*leading_dim, *x.shape[-2:])  # (..., L, E)
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return self.vit.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        original_dtype = x.dtype
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        out = self.attn(
            x.to(torch.float32),
            x.to(torch.float32),
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]
        return out.to(original_dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn((resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "cls_token",
                "pos_embed",
                "ln_pre.*",
                "conv1.*",
            ],
            block_sequence_name="blocks",
            no_decay_filter=["cls_token", "pos_embed"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids


class VisionTransformerRectangular(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        self.pos_embed = nn.Parameter(
            scale * torch.randn(n_patches_height * n_patches_width + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "cls_token",
                "pos_embed",
                "ln_pre.*",
                "conv1.*",
            ],
            block_sequence_name="blocks",
            no_decay_filter=["cls_token", "pos_embed"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids


class GatoVisionTransformerRectangular(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        self.pos_embed = nn.Parameter(
            scale * torch.randn(n_patches_height * n_patches_width, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.img_patch_len = n_patches_height * n_patches_width

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, H_patch, W_patch]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, H_patch * W_patch]
        x = x.permute(0, 2, 1)  # shape = [*, H_patch * W_patch, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.projection
        return x

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "pos_embed",
                "ln_pre.*",
                "conv1.*",
            ],
            block_sequence_name="blocks",
            no_decay_filter=["pos_embed"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids


class MAEDecoder(nn.Module):
    def __init__(
        self,
        input_embd_dim: int,
        decoder_embd_dim: int,
        heads: int,
        layers: int,
        img_size: tuple[int, int],
        patch_size: int,
    ):
        super().__init__()
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        num_patches = n_patches_height * n_patches_width

        self.decoder_embed = nn.Linear(input_embd_dim, decoder_embd_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embd_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embd_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(decoder_embd_dim, heads) for _ in range(layers)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embd_dim)
        self.decoder_pred = nn.Linear(decoder_embd_dim, patch_size**2 * 3, bias=True)

        # initialize decoder position embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            decoder_embd_dim,
            grid_size_h=n_patches_height,
            grid_size_w=n_patches_width,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "mask_token",
                "decoder_pos_embed",
            ],
            block_sequence_name="blocks",
            no_decay_filter=["decoder_pos_embed"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids

    def forward(self, x: torch.Tensor, ids_restore):
        """
        x: (N, L_shorter + 1, D), with CLS token
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ViTMAE(nn.Module):
    def __init__(
        self,
        encoder: VisionTransformerRectangular,
        decoder: MAEDecoder,
        img_size: tuple[int, int],
        patch_size: int,
        encoder_embed_dim: int,
    ):
        super().__init__()
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        num_patches = n_patches_height * n_patches_width

        encoder.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            encoder_embed_dim,
            grid_size_h=n_patches_height,
            grid_size_w=n_patches_width,
            cls_token=True,
        )
        encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.encoder = encoder
        self.decoder = decoder

    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            encoder_pgroup,
            encoder_p_ids,
        ) = self.encoder.get_optimizer_groups(weight_decay, lr_layer_decay, lr_scale)
        decoder_pgroup, decoder_p_ids = self.decoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        return (
            encoder_pgroup + decoder_pgroup,
            encoder_p_ids + decoder_p_ids,
        )

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # add pos embed w/o cls token
        x = x + self.encoder.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.encoder.ln_post(x)

        if self.encoder.projection is not None:
            x = x @ self.encoder.projection

        return x, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio: float):
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.decoder(latent, ids_restore=ids_restore)
        return pred, mask

    @staticmethod
    def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pids = self._get_optimizer_groups(weight_decay, lr_layer_decay, lr_scale)
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay,
            lr_scale,
            exclude_filter=lambda name, p: id(p) in pids,
        )
        all_groups = pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups
