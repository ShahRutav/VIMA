from __future__ import annotations

from typing import Literal, Callable

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
import vima.utils as U
from enlight.learn import (
    default_optimizer_groups,
    check_optimizer_groups,
    rank_zero_info,
)

from einops import rearrange
from vima.nn.utils import build_mlp
from ..base import BasePolicy
from ... import nn as vnn


class XAttnGPTObjAsTokenPolicy(LightningModule, BasePolicy):
    def __init__(
        self,
        *,
        # ====== model ======
        embed_dim: int,
        # ------ transformer backbone ------
        dt_n_layers: int,
        dt_n_heads: int,
        dt_dropout: float,
        xattn_n_heads: int,
        xattn_ff_expanding: int,
        xattn_detach_qk: bool = False,
        xattn_n_positions: int,
        use_geglu: bool = False,
        # ------ observation encoder ------
        # ...... objects ......
        obj_transformer_emb_dim: int,
        img_views: list[str],
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
        # ...... end effector state ......
        end_effector_emb_dim: int,
        # ------ action encoder ------
        action_encoder_emb_dim: int,
        action_encoder_hidden_depth: int,
        use_continuous_action_encoder_despite_discrete_output: bool = False,
        # ------ prompt encoder ------
        prompt_emb_pretrained_lm: str,
        add_prompt_obj_adapter_mlp: bool = False,
        prompt_obj_adapter_hidden_depth: int | None = None,
        t5_prompt_encoder: str | None = None,
        unfreeze_t5_prompt_encoder_last_n_layers: int = 0,
        t5_prompt_encoder_adapter_type: str | None = None,
        t5_prompt_encoder_adapter_positions: int | list[int] | None = None,
        t5_prompt_encoder_adapter_n_layers: int | list[int] | None = None,
        # ------ action decoder ------
        action_decoder_hidden_dim: int,
        action_decoder_hidden_depth: int,
        action_decoder_activation: str | Callable = "relu",
        action_decoder_norm_type: Literal["batchnorm", "layernorm"] | None = None,
        action_decoder_last_layer_gain: float | None = 0.01,
        action_decoder_use_gmm: bool = False,
        action_decoder_gmm_n_components: int = 4,
        # ====== learning ======
        # ------ action type ------
        action_type: Literal["discrete", "continuous"],
        n_discrete_x_bins: int | None = None,  # bins for position x
        n_discrete_y_bins: int | None = None,  # bins for position y
        n_discrete_z_bins: int | None = None,  # bins for position z
        n_discrete_rot_bins: int | None = None,  # bins for rotation
        # ------ loss weights ------
        sub_action_loss_weights: dict | None = None,
    ):
        super().__init__()

        assert action_type in ["discrete", "continuous"]
        self.action_type = action_type
        self._n_discrete_x_bins = n_discrete_x_bins
        self._n_discrete_y_bins = n_discrete_y_bins
        self._n_discrete_z_bins = n_discrete_z_bins
        self._n_discrete_rot_bins = n_discrete_rot_bins
        if action_type == "discrete":
            assert (
                isinstance(n_discrete_x_bins, int)
                and isinstance(n_discrete_y_bins, int)
                and isinstance(n_discrete_z_bins, int)
                and isinstance(n_discrete_rot_bins, int)
            )

        self.embed_dim = embed_dim

        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=dt_n_layers,
            n_head=dt_n_heads,
            dropout=dt_dropout,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=xattn_ff_expanding,
            xattn_detach_qk=xattn_detach_qk,
            xattn_n_positions=xattn_n_positions,
            use_geglu=use_geglu,
        )

        self.obj_encoder = vnn.MultiViewObjectAsTokenEncoder(
            transformer_emb_dim=obj_transformer_emb_dim,
            views=img_views,
            img_encoder_output_dim=img_encoder_output_dim,
            img_encoder_type=img_encoder_type,
            resnet_name=resnet_name,
            use_pretrained_resnet=use_pretrained_resnet,
            freeze_resnet=freeze_resnet,
            resnet_add_top_mlp=resnet_add_top_mlp,
            resnet_top_mlp_hidden_depth=resnet_top_mlp_hidden_depth,
            vit_resolution=vit_resolution,
            vit_patch_size=vit_patch_size,
            vit_width=vit_width,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            add_bbox_mlp=add_bbox_mlp,
            bbox_mlp_hidden_dim=bbox_mlp_hidden_dim,
            bbox_mlp_hidden_depth=bbox_mlp_hidden_depth,
        )

        self.end_effector_encoder = vnn.Embedding(
            num_embeddings=2, embedding_dim=end_effector_emb_dim
        )

        obs_feat_dim = self.obj_encoder.output_dim + end_effector_emb_dim
        self.obs_fusion_layer = (
            nn.Identity()
            if obs_feat_dim == embed_dim
            else nn.Linear(obs_feat_dim, embed_dim)
        )

        self._use_continuous_action_encoder_despite_discrete_output = (
            use_continuous_action_encoder_despite_discrete_output
        )
        if action_type == "discrete":
            if use_continuous_action_encoder_despite_discrete_output:
                self.action_encoder = vnn.MultiHeadActionEmbeddingFusion(
                    output_dim=embed_dim,
                    embed_dict={
                        "pose0_position": vnn.ContinuousActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            input_dim=2,
                            hidden_dim=action_encoder_emb_dim,
                            hidden_depth=action_encoder_hidden_depth,
                        ),
                        "pose0_rotation": vnn.ContinuousActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            input_dim=4,
                            hidden_dim=action_encoder_emb_dim,
                            hidden_depth=action_encoder_hidden_depth,
                        ),
                        "pose1_position": vnn.ContinuousActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            input_dim=2,
                            hidden_dim=action_encoder_emb_dim,
                            hidden_depth=action_encoder_hidden_depth,
                        ),
                        "pose1_rotation": vnn.ContinuousActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            input_dim=4,
                            hidden_dim=action_encoder_emb_dim,
                            hidden_depth=action_encoder_hidden_depth,
                        ),
                    },
                )
            else:
                self.action_encoder = vnn.MultiHeadActionEmbeddingFusion(
                    output_dim=embed_dim,
                    embed_dict={
                        "pose0_position": vnn.MultiDiscreteActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            num_embs=[
                                n_discrete_x_bins,
                                n_discrete_y_bins,
                                n_discrete_z_bins,
                            ],
                            emb_dims=action_encoder_emb_dim,
                        ),
                        "pose0_rotation": vnn.MultiDiscreteActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            num_embs=[n_discrete_rot_bins] * 4,
                            emb_dims=action_encoder_emb_dim,
                        ),
                        "pose1_position": vnn.MultiDiscreteActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            num_embs=[
                                n_discrete_x_bins,
                                n_discrete_y_bins,
                                n_discrete_z_bins,
                            ],
                            emb_dims=action_encoder_emb_dim,
                        ),
                        "pose1_rotation": vnn.MultiDiscreteActionEmbedding(
                            output_dim=action_encoder_emb_dim,
                            num_embs=[n_discrete_rot_bins] * 4,
                            emb_dims=action_encoder_emb_dim,
                        ),
                    },
                )

            self.action_decoder = vnn.DiscreteActionDecoder(
                input_dim=embed_dim,
                action_dims={
                    "pose0_position": [
                        n_discrete_x_bins,
                        n_discrete_y_bins,
                    ],
                    "pose0_rotation": [n_discrete_rot_bins] * 4,
                    "pose1_position": [
                        n_discrete_x_bins,
                        n_discrete_y_bins,
                    ],
                    "pose1_rotation": [n_discrete_rot_bins] * 4,
                },
                hidden_dim=action_decoder_hidden_dim,
                hidden_depth=action_decoder_hidden_depth,
                activation=action_decoder_activation,
                norm_type=action_decoder_norm_type,
                last_layer_gain=action_decoder_last_layer_gain,
            )
        elif action_type == "continuous":
            self.action_encoder = vnn.MultiHeadActionEmbeddingFusion(
                output_dim=embed_dim,
                embed_dict={
                    "pose0_position": vnn.ContinuousActionEmbedding(
                        output_dim=action_encoder_emb_dim,
                        input_dim=3,
                        hidden_dim=action_encoder_emb_dim,
                        hidden_depth=action_encoder_hidden_depth,
                    ),
                    "pose0_rotation": vnn.ContinuousActionEmbedding(
                        output_dim=action_encoder_emb_dim,
                        input_dim=4,
                        hidden_dim=action_encoder_emb_dim,
                        hidden_depth=action_encoder_hidden_depth,
                    ),
                    "pose1_position": vnn.ContinuousActionEmbedding(
                        output_dim=action_encoder_emb_dim,
                        input_dim=3,
                        hidden_dim=action_encoder_emb_dim,
                        hidden_depth=action_encoder_hidden_depth,
                    ),
                    "pose1_rotation": vnn.ContinuousActionEmbedding(
                        output_dim=action_encoder_emb_dim,
                        input_dim=4,
                        hidden_dim=action_encoder_emb_dim,
                        hidden_depth=action_encoder_hidden_depth,
                    ),
                },
            )
            self.action_decoder = vnn.MixedActionDecoder(
                input_dim=embed_dim,
                action_types={
                    "pose0_position": "continuous",
                    "pose0_rotation": "continuous",
                    "pose1_position": "continuous",
                    "pose1_rotation": "continuous",
                },
                action_dims={
                    "pose0_position": 3,  # x, y, z
                    "pose0_rotation": 4,  # a, b, c, d (quaternion)
                    "pose1_position": 3,  # x, y, z
                    "pose1_rotation": 4,  # a, b, c, d (quaternion)
                },
                hidden_dim=action_decoder_hidden_dim,
                hidden_depth=action_decoder_hidden_depth,
                activation=action_decoder_activation,
                norm_type=action_decoder_norm_type,
                last_layer_gain=action_decoder_last_layer_gain,
                use_gmm=action_decoder_use_gmm,
                gmm_num_components=action_decoder_gmm_n_components,
            )
        else:
            raise ValueError(f"Unknown action type {action_type}")

        self.prompt_embedding = vnn.PromptTokenEmbedding(
            prompt_emb_pretrained_lm,
            freeze_pretrained=True,
        )
        self.t5_prompt_encoder = None
        if t5_prompt_encoder is not None:
            assert t5_prompt_encoder == prompt_emb_pretrained_lm
            self.t5_prompt_encoder = vnn.T5PromptEncoder(
                t5_prompt_encoder,
                unfreeze_last_n_layers=unfreeze_t5_prompt_encoder_last_n_layers,
                adapter_type=t5_prompt_encoder_adapter_type,
                adapter_positions=t5_prompt_encoder_adapter_positions,
                adapter_n_layers=t5_prompt_encoder_adapter_n_layers,
            )
            self.t5_prompt_encoder_post_layer = (
                nn.Identity()
                if embed_dim == self.t5_prompt_encoder.output_dim
                else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
            )

        dim = (
            embed_dim
            if self.t5_prompt_encoder is None
            else self.t5_prompt_encoder.input_dim
        )
        if add_prompt_obj_adapter_mlp:
            self.prompt_obj_post_layer = build_mlp(
                self.obj_encoder.output_dim,
                hidden_dim=dim,
                output_dim=dim,
                hidden_depth=prompt_obj_adapter_hidden_depth,
            )
        else:
            self.prompt_obj_post_layer = (
                nn.Identity()
                if dim == self.obj_encoder.output_dim
                else nn.Linear(self.obj_encoder.output_dim, dim)
            )

        if action_type == "discrete":
            if sub_action_loss_weights is None:
                sub_action_loss_weights = {
                    "pose0_position": 1,
                    "pose0_rotation": 1,
                    "pose1_position": 1,
                    "pose1_rotation": 1,
                }
            else:
                assert set(sub_action_loss_weights.keys()) == {
                    "pose0_position",
                    "pose0_rotation",
                    "pose1_position",
                    "pose1_rotation",
                }
        else:
            if sub_action_loss_weights is None:
                sub_action_loss_weights = {
                    "pose0_position": 1 / 2,
                    "pose0_rotation": 1 / 4,
                    "pose1_position": 1 / 2,
                    "pose1_rotation": 1 / 4,
                }
            else:
                assert set(sub_action_loss_weights.keys()) == {
                    "pose0_position",
                    "pose0_rotation",
                    "pose1_position",
                    "pose1_rotation",
                }
        self._sub_action_loss_weights = sub_action_loss_weights
        self._views = img_views
        self._inference_cache = {}

    def training_step(self, batch, batch_idx):
        obs, action, action_mask, prompts, task_name_to_batch_indices = batch
        L_obs, B = list(obs.values())[0].shape[:2]
        L_action = list(action.values())[0].shape[0]
        # last obs is the terminal obs
        assert L_obs == L_action + 1
        # remove terminal obs
        obs = U.any_slice(obs, np.s_[:-1])
        assert list(obs.values())[0].shape[0] == L_action

        # compute prompt tokens
        prompt_tokens, prompt_masks = self.forward_prompt_assembly(prompts)

        # compute obs tokens
        obs_tokens, obs_masks = self.forward_obs_token(obs)

        # compute action tokens
        # discretize if needed
        if self.action_type == "discrete":
            action = self.discretize_action(action)
        # cache target action
        tar_action = {k: v.clone() for k, v in action.items()}
        # slice action sequence up to the last one
        action = U.any_slice(action, np.s_[:-1])
        action_tokens = self.forward_action_token(action)

        # get predicted action tokens
        pred_action_tokens = self.forward(
            obs_token=obs_tokens,
            action_token=action_tokens,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_masks,
        )
        assert pred_action_tokens.shape[:2] == list(tar_action.values())[0].shape[:2]

        # forward action decoder
        dist_dict = self.forward_action_decoder(pred_action_tokens)
        assert (
            set(dist_dict.keys()) == set(tar_action.keys()) == set(action_mask.keys())
        )

        # compute imitation loss
        imitation_loss = {}
        if self.action_type == "discrete":
            for k, dist in dist_dict.items():
                mask = action_mask[k]
                raw_loss_list = dist.imitation_loss(
                    actions=tar_action[k], reduction="none"
                )
                raw_loss_list = [
                    raw_loss.reshape(sub_mask.shape)
                    for raw_loss, sub_mask in zip(
                        raw_loss_list, torch.unbind(mask, dim=-1)
                    )
                ]
                # reduce the loss according to the action mask
                # "True" indicates should calculate the loss
                loss_list = [
                    (raw_loss * sub_mask).sum() / B
                    for raw_loss, sub_mask in zip(
                        raw_loss_list, torch.unbind(mask, dim=-1)
                    )
                ]
                # average all components
                loss = sum(loss_list) / len(loss_list)
                imitation_loss[k] = loss
        elif self.action_type == "continuous":
            for k, dist in dist_dict.items():
                mask = action_mask[k]
                mask = mask[..., 0]
                raw_loss = dist.imitation_loss(
                    actions=tar_action[k], mask=mask, reduction="none"
                )  # (L, B)
                loss = raw_loss.sum() / B
                imitation_loss[k] = loss
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
        loss = sum(
            imitation_loss[k] * self._sub_action_loss_weights[k]
            for k in imitation_loss.keys()
        )

        # compute metrics
        metrics_ = {}
        if self.action_type == "discrete":
            acc = {}
            for k, dist in dist_dict.items():
                mask = action_mask[k]
                if mask.sum() == 0:
                    continue
                # same mask applies to all components
                mask = mask[..., 0]
                primitive_shape = mask.shape
                acc_list = dist.imitation_accuracy(
                    actions=tar_action[k], mask=mask, reduction="none"
                )  # list of (L * B)
                reduced_acc_list = [raw_acc.sum() / mask.sum() for raw_acc in acc_list]
                acc[k] = sum(reduced_acc_list) / len(reduced_acc_list)
                # compute breakdown acc for each task
                for task_name, batch_indices in task_name_to_batch_indices.items():
                    if mask[:, batch_indices].sum() == 0:
                        continue
                    task_acc_list = [
                        raw_acc.reshape(primitive_shape)[:, batch_indices].sum()
                        / mask[:, batch_indices].sum()
                        for raw_acc in acc_list
                    ]
                    task_acc = sum(task_acc_list) / len(task_acc_list)
                    acc[f"{task_name}_{k}"] = task_acc
            avg_position_acc = (acc["pose0_position"] + acc["pose1_position"]) / 2
            acc["position"] = avg_position_acc
            if "pose0_rotation" in acc and "pose1_rotation" in acc:
                avg_rotation_acc = (acc["pose0_rotation"] + acc["pose1_rotation"]) / 2
                acc["rotation"] = avg_rotation_acc
            for task_name in task_name_to_batch_indices.keys():
                avg_position_acc = (
                    acc[f"{task_name}_pose0_position"]
                    + acc[f"{task_name}_pose1_position"]
                ) / 2
                acc[f"{task_name}_position"] = avg_position_acc
                if (
                    f"{task_name}_pose0_rotation" in acc
                    and f"{task_name}_pose1_rotation" in acc
                ):
                    avg_rotation_acc = (
                        acc[f"{task_name}_pose0_rotation"]
                        + acc[f"{task_name}_pose1_rotation"]
                    ) / 2
                    acc[f"{task_name}_rotation"] = avg_rotation_acc
            metrics_["acc"] = acc
        elif self.action_type == "continuous":
            l1 = {}
            for k, dist in dist_dict.items():
                mask = action_mask[k]
                mask = mask[..., 0]
                raw_l1 = dist.imitation_accuracy(
                    actions=tar_action[k], mask=mask, reduction="none"
                )
                reduced_l1 = raw_l1.sum() / mask.sum()
                l1[k] = reduced_l1 / (3 if "position" in k else 4)
                # compute breakdown l1 for each task
                # raw_l1 is (L, B)
                for task_name, batch_indices in task_name_to_batch_indices.items():
                    task_l1 = (
                        raw_l1[:, batch_indices].sum() / mask[:, batch_indices].sum()
                    )
                    l1[f"{task_name}_{k}"] = task_l1 / (3 if "position" in k else 4)
            avg_position_l1 = (l1["pose0_position"] + l1["pose1_position"]) / 2
            l1["position"] = avg_position_l1
            avg_rotation_l1 = (l1["pose0_rotation"] + l1["pose1_rotation"]) / 2
            l1["rotation"] = avg_rotation_l1
            for task_name in task_name_to_batch_indices.keys():
                avg_position_l1 = (
                    l1[f"{task_name}_pose0_position"]
                    + l1[f"{task_name}_pose1_position"]
                ) / 2
                l1[f"{task_name}_position"] = avg_position_l1
                avg_rotation_l1 = (
                    l1[f"{task_name}_pose0_rotation"]
                    + l1[f"{task_name}_pose1_rotation"]
                ) / 2
                l1[f"{task_name}_rotation"] = avg_rotation_l1
            metrics_["l1"] = l1
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        metrics_["loss"] = imitation_loss
        metrics = {}
        for metric_name, metric_dict in metrics_.items():
            for k, v in metric_dict.items():
                metrics[f"{metric_name}_{k}"] = v
        # average metrics
        for metric_name, metric_dict in metrics_.items():
            metrics[metric_name] = sum(metric_dict.values()) / len(metric_dict)
        # calculate real batch size
        mask = action_mask["pose0_position"][:, :, 0]
        assert mask.shape == (L_action, B)
        real_batch_size = mask.sum()
        return loss, metrics, real_batch_size

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx)

    def forward(
        self,
        obs_token: torch.Tensor,
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ):
        """
        obs_token: (L, B, max_n_objs, E)
        obs_mask: (L, B, max_n_objs)
        action_token and prompt_token: (L, B, E)
        """
        if action_token is not None:
            assert obs_token.dim() - 1 == action_token.dim() == prompt_token.dim() == 3
            assert obs_token.shape[1] == action_token.shape[1] == prompt_token.shape[1]
            assert (
                obs_token.shape[3]
                == action_token.shape[2]
                == prompt_token.shape[2]
                == self.embed_dim
            )
        else:
            assert obs_token.dim() - 1 == prompt_token.dim() == 3
            assert obs_token.shape[1] == prompt_token.shape[1]
            assert obs_token.shape[3] == prompt_token.shape[2] == self.embed_dim
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        assert L_action == L_obs - 1
        n_max_objs = obs_token.shape[-2]
        L = L_obs * n_max_objs + L_action

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=self.device
        )
        masks = torch.ones(L, B, dtype=torch.bool, device=self.device)
        # interleave obs tokens and action tokens one by one
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")  # (L_obs * n_max_objs, B, E)
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")  # (L_obs * n_max_objs, B)
        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs :: n_max_objs + 1] = action_token

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        # since prompt_token_mask is (B, L), we cumsum along L
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1

        # forward sequence modeling
        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )
        assert tokens_out.shape == (L, B, self.embed_dim)

        # extract output tokens corresponding to predicted actions
        predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]
        # for each obs token we need to predict one action token, so there are L_obs predicted action tokens
        assert predicted_action_tokens.shape == (L_obs, B, self.embed_dim)
        return predicted_action_tokens

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        B = len(raw_prompts_token_type)

        # encode word batch
        n_words = word_batch.shape[0]
        batch_word_emb = self.prompt_embedding(word_batch)
        assert batch_word_emb.shape == (
            n_words,
            self.embed_dim
            if self.t5_prompt_encoder is None
            else self.t5_prompt_encoder.input_dim,
        )

        # encode image batch
        n_img = len(list(image_batch["name"].values())[0])
        del image_batch["name"]
        batch_image_emb = self.obj_encoder(
            **image_batch
        )  # (n_imgs, n_max_objs, embed_dim)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        n_max_objs = batch_image_emb.shape[-2]

        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:  # 0 is for word
                    L_this += 1
                elif item == 1:  # 1 is for image
                    L_this += n_max_objs
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)

        # start assembling
        prompt_tokens, prompt_masks = [], []
        word_ptr, img_ptr = 0, 0
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = []
            assembled_mask = []
            for item in raw_prompt:
                if item == 0:  # 0 is for word
                    assembled_prompt.append(batch_word_emb[word_ptr])
                    word_ptr += 1
                    assembled_mask.append(True)
                elif item == 1:  # 1 is for image
                    obj_mask = U.any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            assert len(assembled_prompt) <= L_max
            # calculate num of padding
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            # do padding
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            assert assembled_prompt.shape[0] == L_max
            prompt_tokens.append(assembled_prompt)

            # generate prompt mask
            prompt_masks.append(
                torch.cat(
                    [
                        U.any_to_torch_tensor(
                            assembled_mask, dtype=torch.bool, device=self.device
                        ),
                        torch.zeros(num_padding, dtype=torch.bool, device=self.device),
                    ],
                    dim=0,
                )
            )

        # check that all embedded features are used
        assert word_ptr == n_words, "INTERNAL"
        assert img_ptr == n_img, "INTERNAL"
        prompt_tokens = torch.stack(prompt_tokens, dim=0)  # (B, L, emb_dim)
        prompt_masks = torch.stack(prompt_masks, dim=0)  # (B, L)
        prompt_tokens = prompt_tokens.transpose(0, 1)  # (L, B, emb_dim)
        assert prompt_tokens.shape == (
            L_max,
            B,
            self.embed_dim
            if self.t5_prompt_encoder is None
            else self.t5_prompt_encoder.input_dim,
        )
        assert prompt_masks.shape == (B, L_max)
        if self.t5_prompt_encoder is not None:
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):
        """
        obs: dict with keys `objects`, `ee`

        optional:
            prompt_tokens: (L_prompt, B, emb_dim)
            prompt_masks: (B, L_prompt)
        """
        objects, ee = obs["objects"], obs["ee"]
        del objects["name"]
        leading_dims = ee.shape[:2]  # (L, B)

        # encode vision
        # flatten first two dimensions
        objects = objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[2:])
        )  # (L * B, ...)
        img_feats = self.obj_encoder(**objects)  # (L * B, n_max_objs, feat_dim)
        # recover first two dimensions
        img_feats = img_feats.reshape(
            *leading_dims, *img_feats.shape[1:]
        )  # (L, B, n_max_objs, feat_dim)
        obj_mask = {
            k: objects["mask"][k].reshape(*leading_dims, -1) for k in objects["mask"]
        }

        # encode end effector state
        ee_feats = self.end_effector_encoder(ee)  # (L, B, feat_dim)
        ee_feats = ee_feats.unsqueeze(2).repeat(
            1, 1, img_feats.shape[-2], 1
        )  # (L, B, n_max_objs, feat_dim)

        # concat vision and end effector features and fuse
        obs_feats = self.obs_fusion_layer(
            torch.cat(
                [img_feats, ee_feats], dim=-1
            )  # (L, B, n_max_objs, obj_encoder_feat_dim + ee_feat_dim)
        )  # (L, B, n_max_objs, E)

        obj_mask = U.any_concat(
            [obj_mask[view] for view in sorted(self._views)], dim=-1
        )
        return obs_feats, obj_mask

    def forward_action_token(self, action):
        if (
            self.action_type == "discrete"
            and self._use_continuous_action_encoder_despite_discrete_output
        ):
            # de-discretize action
            action = self._de_discretize_actions(action)
        return self.action_encoder(action)

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        """
        predicted_action_tokens: (L, B, emb_dim)
        """
        assert predicted_action_tokens.dim() == 3
        assert predicted_action_tokens.shape[-1] == self.embed_dim
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):
        """
        discretize dimension of position and rotation to `n_discrete_action_bins`
        """
        # position and rotation have been normalized to [0, 1]
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=self.device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=self.device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=self.device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )

        # convert dtype to long
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        assert self.action_type == "discrete"
        # convert dtype to float
        actions = {k: v.float() for k, v in actions.items()}
        # de-discretize position and rotation
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        xattn_pg, xattn_pids = self.xattn_gpt.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        obj_encoder_pg, obj_encoder_pids = self.obj_encoder.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        prompt_embd_pg, prompt_embd_pids = self.prompt_embedding.get_optimizer_groups(
            weight_decay, lr_layer_decay, lr_scale
        )
        t5_prompt_encoder_pg, t5_prompt_encoder_pids = [], []
        if self.t5_prompt_encoder is not None:
            (
                t5_prompt_encoder_pg,
                t5_prompt_encoder_pids,
            ) = self.t5_prompt_encoder.get_optimizer_groups(
                weight_decay, lr_layer_decay, lr_scale
            )
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay,
            lr_scale,
            no_decay_filter=[
                "end_effector_encoder.*",
                "prompt_obj_post_layer.*",
                "obs_fusion_layer.*",
                "action_encoder.*",
                "action_decoder.*",
                "t5_prompt_encoder_post_layer.*",
            ],
            exclude_filter=lambda name, p: id(p)
            in xattn_pids
            + obj_encoder_pids
            + prompt_embd_pids
            + t5_prompt_encoder_pids,
        )
        all_groups = xattn_pg + obj_encoder_pg + t5_prompt_encoder_pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups
    
    def act(self):
        pass
