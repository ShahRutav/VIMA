from __future__ import annotations

from typing import Literal, Callable

import torch
import torch.nn as nn
import numpy as np
from termcolor import colored
import peft
import transformers
from pytorch_lightning import LightningModule
from transformers import BitsAndBytesConfig

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


class VLMPolicy(LightningModule, BasePolicy):
    def __init__(
        self,
        *,
        # ====== model ======
        embed_dim: int,
        # ....... vlm backbone .......
        model_name: str,
        revision: str,
        torch_dtype: str,
        vlm_last_n_feats: int,
        load_in_4bit: bool,
        bnb_4bit_use_double_quant: bool,
        bnb_4bit_quant_type: str,
        # vlm head that projects the vlm hidden state to the embed_dim
        vlm_head_type: Literal["linear", "mlp_block", "default"],
        # ......... finetuning configs .........
        # LORA configs
        use_lora: bool,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] = None,
        lora_only_last_n_layers: int = -1,
        # ...... objects ......
        img_views: list[str],
        # ...... end effector state ......
        end_effector_emb_dim: int,
        # ------ action encoder ------
        action_encoder_emb_dim: int,
        action_encoder_hidden_depth: int,
        use_continuous_action_encoder_despite_discrete_output: bool = False,
        # ------ prompt encoder ------
        prompt_emb_pretrained_lm: str,
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

        torch_dtype = U.get_torch_dtype(torch_dtype)
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

        # self.end_effector_encoder = vnn.Embedding(
        #     num_embeddings=2, embedding_dim=end_effector_emb_dim
        # )

        # obs_feat_dim = embed_dim + end_effector_emb_dim # output dim of vlm + end effector embedding
        # self.obs_fusion_layer = (
        #     nn.Identity()
        #     if obs_feat_dim == embed_dim
        #     else nn.Linear(obs_feat_dim, embed_dim)
        # )

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

        '''
        # the below prompt_embedding is not necessary since we do not want to finetune the embeddings.
        # TODO: If used, directly pass the model embeddings to avoid loading the model twice. very expensive and time consuming.
        self.prompt_embedding = vnn.PromptVLMTokenEmbedding(
                pretrained_lm_str=prompt_emb_pretrained_lm,
                freeze_pretrained=True,)
        '''

        # # ------- VLM Model loading ----
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
        )
        vlm_model = vnn.LlavaPromptEncoder.from_pretrained(
                model_name,
                revision=revision,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                last_n_feats=vlm_last_n_feats,
        )
        vlm_model = peft.prepare_model_for_kbit_training(vlm_model)
        # we keep this before the model is wrapped with trainable additional parameters
        if use_lora:
            lora_config = peft.LoraConfig(
                r=lora_rank,
                lora_alpha=2*lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=lora_target_modules,
                layers_to_transform=vlm_model.get_lora_layers_to_transform(lora_only_last_n_layers) if lora_only_last_n_layers > 0 else None,
                task_type=peft.TaskType.CAUSAL_LM,
            )
            vlm_model = peft.get_peft_model(vlm_model, lora_config)
        self.vlm_model = vlm_model
        del vlm_model

        if vlm_head_type == "mlp_block":
            self.vlm_head = vnn.MLP(
                    input_dim=self.vlm_model.vlm_hidden_size,
                    hidden_dim=self.vlm_model.vlm_hidden_size,
                    output_dim=embed_dim,
                    hidden_depth=0,
                    norm_type='layernorm',
            )
        else:
            raise ValueError(f"Unknown vlm_head_type {vlm_head_type}")

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
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        obs, action, action_mask, prompt_dict, task_name_to_batch_indices = batch
        # calculate the DRAM usage

        B = list(prompt_dict.values())[0].shape[0]
        # L_obs, B = list(obs.values())[0].shape[:2]
        L_action = list(action.values())[0].shape[0]
        assert B == list(action.values())[0].shape[1]

        # compute action tokens
        # discretize if needed
        if self.action_type == "discrete":
            action = self.discretize_action(action)
        # cache target action
        tar_action = {k: v.clone() for k, v in action.items()}
        # slice action sequence up to the last one
        action = U.any_slice(action, np.s_[:-1]) # remove the last element from L_max. Empty tensor when traj_length=1
        action_tokens = self.forward_action_token(action)

        # get predicted action tokens
        pred_action_tokens = self.forward(
            action_token=action_tokens,
            prompt_dict=prompt_dict,
        )
        assert pred_action_tokens.shape[:2] == list(tar_action.values())[0].shape[:2], \
                f"pred_action_tokens.shape[:2]={pred_action_tokens.shape[:2]}, tar_action.shape[:2]={list(tar_action.values())[0].shape[:2]}"

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
        action_token: torch.Tensor | None,
        prompt_dict: dict,
    ):
        """
        prompt_token: batch first dictionary
        action_token and prompt_token: (L, B, E)
        """
        # output all the hidden states is very memory inefficient.
        # TODO: Alternative: Replace lm_head with identiy and the use the last hidden state by accessing logits.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # import ipdb; ipdb.set_trace()
            # converts to bfloat16. note the output with be float32 since lm_head is float32 handeled by the prepare_model_for_kbit_training
            vlm_feats = self.vlm_model(
                    **prompt_dict,
                    output_last_hidden_state=True,
                    return_dict=True,
            ).last_hidden_state
            # # make a dummy feats for now
            # vlm_feats = torch.zeros(action_token.shape[1], 1, self.embed_dim).to(self.device)

            if vlm_feats.shape[1] > 1:
                vlm_feats = vlm_feats.mean(dim=1, keepdim=True)
            # B, 1, E -> B, 1, e
            vlm_feats = self.vlm_head(vlm_feats)
            # B, 1, e -> 1, B, e -> L, B, e
            vlm_feats = U.any_transpose_first_two_axes(vlm_feats).expand(action_token.shape[0]+1, -1, -1)
        return vlm_feats

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
        vlm_pg, vlm_pids = self.vlm_model.get_optimizer_groups(
                weight_decay, lr_layer_decay, lr_scale
        )
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay,
            lr_scale,
            no_decay_filter=[
                "action_encoder.*",
                "action_decoder.*",
                "vlm_head.*",
            ],
            exclude_filter=lambda name, p: id(p) in vlm_pids,
        )
        all_groups = vlm_pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups

    def act(self):
        pass
