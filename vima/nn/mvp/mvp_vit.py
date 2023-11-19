import timm
import torch
import torch.nn as nn
import enlight.utils as U
from enlight.learn import (
    default_optimizer_groups,
    check_optimizer_groups,
    rank_zero_info,
    transformer_lr_decay_optimizer_groups,
)
from vima.nn.constant import VIMA_IMG_MEAN, VIMA_IMG_STD


class MVPVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer
    referene:
        - MAE:  https://github.com/facebookresearch/mae/blob/main/models_vit.py
        - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # remove the classifier
        if hasattr(self, "pre_logits"):
            del self.pre_logits
        del self.head

        self.img_patch_len = int(
            kwargs["img_size"][0]
            / kwargs["patch_size"]
            * kwargs["img_size"][1]
            / kwargs["patch_size"]
        )

    def finetune_forward(self, x, mask_ratio):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # x = x[:, 0].detach().float()
        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))

    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

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
        optim_groups, all_param_ids = transformer_lr_decay_optimizer_groups(
            self,
            layer_0_params=[
                "cls_token",
                "pos_embed",
                "patch_embed.*",
            ],
            block_sequence_name="blocks",
            no_decay_filter=["cls_token", "pos_embed"],
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        return optim_groups, all_param_ids


class MVPViTMAE(nn.Module):
    def __init__(
        self,
        encoder: MVPVisionTransformer,
        decoder,
    ):
        super().__init__()

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

    def forward(self, x: torch.Tensor, mask_ratio: float):
        x = U.basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        latent, mask, ids_restore = self.encoder.finetune_forward(x, mask_ratio)
        pred = self.decoder(latent, ids_restore=ids_restore)
        return pred, mask

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
