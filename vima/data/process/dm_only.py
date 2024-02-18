from __future__ import annotations
from typing import Literal
from copy import deepcopy
from functools import partial

import torch
import numpy as np
from tokenizers import Tokenizer
from transformers import ProcessorMixin
import vima.utils as U

from .common import (
    prepare_prompt_with_bbox,
    prepare_prompt_with_bbox_from_detection,
    collate_prompt_with_bbox,
    prepare_obs_with_bbox,
    prepare_obs_with_bbox_from_detection,
    collate_obs_with_bbox,
    prepare_prompt_rgb_only,
    prepare_vlm_prompt_rgb_only,
    prepare_obs_rgb_only,
    prepare_vlm_obs_rgb_only,
    collate_prompt_rgb_only,
    collate_vlm_prompt_rgb_only,
    collate_obs_rgb_only,
)
from ..constants import OBJ_NAME_TO_ID, PLACEHOLDERS


def prepare_sample(
    obs: dict,
    action: dict,
    action_position_bounds: dict[str, np.ndarray],
    prompt: str | None,
    prompt_assets: dict | None,
    meta: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
):
    """
    Prepare trajectory data. Used in both training data module and rollout eval.
    This function does the following things.
    1. Use tokenizer to tokenize prompt texts.
    2. Stack images from `V` views to (..., V, C, H, W) for rgb or (..., V, H, W) for segmentation if used.
    3. If segm is used, replace local obj ids in segm with global ids from `OBJ_NAME_TO_ID`.
    4. Fill placeholders in prompt with corresponding image assets.
    5. Normalize sub-actions position and rotation to [0, 1]

    obs: nested obs dict, each value has the shape of (L + 1, ...)
    action: nested action dict, each value has the shape of (L, ...)
    action_position_bounds: dict with keys "low" and "high" in (x, y, z) format
    prompt: str, prompt text
    prompt_assets: dict of prompt assets, should contain all placeholders
    meta: dict from env.meta_info
    tokenizer: HF tokenizer with placeholder tokens properly set
    add_special_tokens: whether to add special tokens such as [CLS] and [SEP].
        Should be True if use pretrained LM backbone.

    Return obs, action, filled_prompt
    """
    # get image H and W. We assume that rgb must exist
    H, W = list(obs["rgb"].values())[0].shape[-2:]
    # get number of time-steps
    L = U.get_batch_size(obs) - 1
    assert L == U.get_batch_size(action) == meta["steps"]

    # stack images from `V` views to (L + 1, V, C, H, W) for rgb or (L + 1, V, H, W) for segmentation if used.
    rgb_views = sorted(obs["rgb"].keys())
    stacked_rgb = np.stack([obs["rgb"][view] for view in rgb_views], axis=0).swapaxes(
        0, 1
    )
    U.check_shape(stacked_rgb, [L + 1, len(rgb_views), 3, H, W])
    obs["rgb"] = stacked_rgb
    if "segm" in obs:
        segm_views = sorted(obs["segm"].keys())
        stacked_segm = np.stack(
            [obs["segm"][view] for view in segm_views], axis=0
        ).swapaxes(0, 1)
        U.check_shape(stacked_segm, [L + 1, len(segm_views), H, W])
        obs["segm"] = stacked_segm.astype(np.int64)

    # if use segm, replace local obj ids in segm obs with global ids from `OBJ_NAME_TO_ID`
    if "segm" in obs:
        local_id_to_global_id = {
            component_id: OBJ_NAME_TO_ID["robot components"]
            for component_id in meta["robot_components"]
        }
        assert meta["n_objects"] == len(meta["obj_id_to_info"])
        local_id_to_global_id.update(
            {
                k: OBJ_NAME_TO_ID[f"{v['texture_name']} {v['obj_name']}"]
                for k, v in meta["obj_id_to_info"].items()
            }
        )
        masks = [
            (global_id, obs["segm"] == local_id)
            for local_id, global_id in local_id_to_global_id.items()
        ]
        padding_mask = ~np.any(np.stack([mask for _, mask in masks], axis=0), axis=0)
        for global_id, mask in masks:
            obs["segm"][mask] = global_id
        obs["segm"][padding_mask] = OBJ_NAME_TO_ID["padding"]

    # normalize action to [0, 1]
    # normalize position with action_position_bounds
    action["position"] = (action["position"] - action_position_bounds["low"]) / (
        action_position_bounds["high"] - action_position_bounds["low"]
    )
    # check all normalized actions are in [0, 1]
    assert np.all(action["position"] >= 0) and np.all(action["position"] <= 1)
    # normalize rotation
    # rotation is represented in quaternion in [-1, 1]
    action["rotation"] = (action["rotation"] + 1) / 2
    # check all normalized actions are in [0, 1]
    assert np.all(action["rotation"] >= 0) and np.all(action["rotation"] <= 1)
    # change subactions release and push to int64
    action["release"] = action["release"].astype(np.int64)
    action["push"] = action["push"].astype(np.int64)

    # tokenize prompt
    # bypass None case because this function is also used in eval rollout, where we prepare prompt once and cache it
    if prompt is None:
        filled_prompt = None
    else:
        encoding = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
        assert set(prompt_assets.keys()) == set(
            [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
        )
        filled_prompt = []
        for id, token in zip(prompt_ids, prompt_tokens):
            if token not in PLACEHOLDERS:
                # an indexed word
                assert "{" not in token and "}" not in token
                filled_prompt.append(id)
            else:
                # a multimodal placeholder
                assert token.startswith("{") and token.endswith("}")
                asset_name = token[1:-1]
                assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
                asset = prompt_assets[asset_name]
                # stack images from `V` views to (V, C, H, W) for rgb or (V, H, W) for segmentation if used.
                rgb_views = sorted(asset["rgb"].keys())
                stacked_rgb = np.stack(
                    [asset["rgb"][view] for view in rgb_views], axis=0
                )
                U.check_shape(stacked_rgb, [len(rgb_views), 3, H, W])
                asset["rgb"] = stacked_rgb
                if "segm" in asset:
                    obj_info = asset["segm"].pop("obj_info")
                    segm_views = sorted(asset["segm"].keys())
                    stacked_segm = np.stack(
                        [asset["segm"][view] for view in segm_views], axis=0
                    ).astype(np.int64)
                    U.check_shape(stacked_segm, [len(segm_views), H, W])

                    # replace local obj ids in segm obs with global ids from `OBJ_NAME_TO_ID`
                    placeholder_type = asset.pop("placeholder_type")
                    if placeholder_type == "object":
                        assert isinstance(obj_info, dict)
                        assert (
                            "obj_id" in obj_info
                            and "obj_color" in obj_info
                            and "obj_name" in obj_info
                        )
                        local_id_to_global_id = {
                            obj_info["obj_id"]: OBJ_NAME_TO_ID[
                                f"{obj_info['obj_color']} {obj_info['obj_name']}"
                            ]
                        }
                    elif placeholder_type == "scene":
                        assert isinstance(obj_info, list)
                        local_id_to_global_id = {
                            each_info["obj_id"]: OBJ_NAME_TO_ID[
                                f"{each_info['obj_color']} {each_info['obj_name']}"
                            ]
                            for each_info in obj_info
                        }
                    else:
                        raise ValueError(
                            f"unknown placeholder type {asset['placeholder_type']}"
                        )
                    masks = [
                        (global_id, stacked_segm == local_id)
                        for local_id, global_id in local_id_to_global_id.items()
                    ]
                    padding_mask = ~np.any(
                        np.stack([mask for _, mask in masks], axis=0), axis=0
                    )
                    for global_id, mask in masks:
                        stacked_segm[mask] = global_id
                    stacked_segm[padding_mask] = OBJ_NAME_TO_ID["padding"]
                    asset["segm"] = stacked_segm
                filled_prompt.append(asset)
    return obs, action, filled_prompt


def prepare_sample_bbox(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    action: dict,
    action_position_bounds: dict[str, np.ndarray],
    prompt: str | None,
    prompt_assets: dict | None,
    meta: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
    cropped_img_size: int = 224,
    add_obj_aug: bool = False,
    obj_aug_prob_map: dict[int, float] | None = None,
    from_detection_model: bool = False,
    detection_model=None,
):
    # get number of time-steps
    L = U.get_batch_size(obs) - 1
    assert L == U.get_batch_size(action) == meta["steps"]

    if from_detection_model:
        obs_rtn = prepare_obs_with_bbox_from_detection(
            obs=obs,
            rgb_dict=rgb_dict,
            meta=meta,
            cropped_img_size=cropped_img_size,
            add_obj_aug=add_obj_aug,
            obj_aug_prob_map=obj_aug_prob_map,
            detection_model=detection_model,
        )
    else:
        obs_rtn = prepare_obs_with_bbox(
            obs=obs,
            rgb_dict=rgb_dict,
            meta=meta,
            cropped_img_size=cropped_img_size,
            add_obj_aug=add_obj_aug,
            obj_aug_prob_map=obj_aug_prob_map,
        )

    # normalize action to [0, 1]
    # normalize position with action_position_bounds
    action["pose0_position"] = (
        action["pose0_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    action["pose1_position"] = (
        action["pose1_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    # check all normalized positions are in [0, 1]
    assert np.all(action["pose0_position"] >= 0) and np.all(
        action["pose0_position"] <= 1
    )
    assert np.all(action["pose1_position"] >= 0) and np.all(
        action["pose1_position"] <= 1
    )
    # normalize rotation
    # rotation is represented in quaternion in [-1, 1]
    action["pose0_rotation"] = (action["pose0_rotation"] + 1) / 2
    action["pose1_rotation"] = (action["pose1_rotation"] + 1) / 2
    # check all normalized rotations are in [0, 1]
    assert np.all(action["pose0_rotation"] >= 0) and np.all(
        action["pose0_rotation"] <= 1
    )
    assert np.all(action["pose1_rotation"] >= 0) and np.all(
        action["pose1_rotation"] <= 1
    )

    # tokenize prompt
    # bypass None case because this function is also used in eval rollout, where we prepare prompt once and cache it
    if prompt is None:
        filled_prompt = None
    else:
        if from_detection_model:
            filled_prompt = prepare_prompt_with_bbox_from_detection(
                prompt=prompt,
                prompt_assets=prompt_assets,
                tokenizer=tokenizer,
                add_special_tokens=add_special_tokens,
                cropped_img_size=cropped_img_size,
                add_obj_aug=add_obj_aug,
                obj_aug_prob_map=obj_aug_prob_map,
                detection_model=detection_model,
            )
        else:
            filled_prompt = prepare_prompt_with_bbox(
                prompt=prompt,
                prompt_assets=prompt_assets,
                tokenizer=tokenizer,
                add_special_tokens=add_special_tokens,
                cropped_img_size=cropped_img_size,
                add_obj_aug=add_obj_aug,
                obj_aug_prob_map=obj_aug_prob_map,
            )
    return obs_rtn, action, filled_prompt

def prepare_sample_vlm_rgb_only(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    action: dict,
    action_position_bounds: dict[str, np.ndarray],
    prompt: str | None,
    prompt_assets: dict | None,
    meta: dict,
    tokenizer: Tokenizer,
    processor: ProcessorMixin,
    add_special_tokens: bool,
):
    # get number of time-steps
    L = U.get_batch_size(obs) - 1
    assert L == U.get_batch_size(action) == meta["steps"]

    obs_rtn = prepare_vlm_obs_rgb_only(
        obs=obs,
        rgb_dict=rgb_dict,
        img_h=256, img_w=256,
    )

    # normalize action to [0, 1]
    # normalize position with action_position_bounds
    action["pose0_position"] = (
        action["pose0_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    action["pose1_position"] = (
        action["pose1_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    # check all normalized positions are in [0, 1]
    assert np.all(action["pose0_position"] >= 0) and np.all(
        action["pose0_position"] <= 1
    )
    assert np.all(action["pose1_position"] >= 0) and np.all(
        action["pose1_position"] <= 1
    )
    # normalize rotation
    # rotation is represented in quaternion in [-1, 1]
    action["pose0_rotation"] = (action["pose0_rotation"] + 1) / 2
    action["pose1_rotation"] = (action["pose1_rotation"] + 1) / 2
    # check all normalized rotations are in [0, 1]
    assert np.all(action["pose0_rotation"] >= 0) and np.all(
        action["pose0_rotation"] <= 1
    )
    assert np.all(action["pose1_rotation"] >= 0) and np.all(
        action["pose1_rotation"] <= 1
    )

    # tokenize prompt
    # bypass None case because this function is also used in eval rollout, where we prepare prompt once and cache it
    if prompt is None:
        filled_prompt = None
    else:
        filled_prompt = prepare_vlm_prompt_rgb_only(
            rgb=deepcopy(obs_rtn['rgb']),
            text_prompt=prompt,
            prompt_assets=prompt_assets,
            processor=processor,
            add_special_tokens=add_special_tokens,
        )
    return obs_rtn, action, filled_prompt

def prepare_sample_rgb_only(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    action: dict,
    action_position_bounds: dict[str, np.ndarray],
    prompt: str | None,
    prompt_assets: dict | None,
    meta: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
):
    # get number of time-steps
    L = U.get_batch_size(obs) - 1
    assert L == U.get_batch_size(action) == meta["steps"]

    obs_rtn = prepare_obs_rgb_only(
        obs=obs,
        rgb_dict=rgb_dict,
    )

    # normalize action to [0, 1]
    # normalize position with action_position_bounds
    action["pose0_position"] = (
        action["pose0_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    action["pose1_position"] = (
        action["pose1_position"] - action_position_bounds["low"]
    ) / (action_position_bounds["high"] - action_position_bounds["low"])
    # check all normalized positions are in [0, 1]
    assert np.all(action["pose0_position"] >= 0) and np.all(
        action["pose0_position"] <= 1
    )
    assert np.all(action["pose1_position"] >= 0) and np.all(
        action["pose1_position"] <= 1
    )
    # normalize rotation
    # rotation is represented in quaternion in [-1, 1]
    action["pose0_rotation"] = (action["pose0_rotation"] + 1) / 2
    action["pose1_rotation"] = (action["pose1_rotation"] + 1) / 2
    # check all normalized rotations are in [0, 1]
    assert np.all(action["pose0_rotation"] >= 0) and np.all(
        action["pose0_rotation"] <= 1
    )
    assert np.all(action["pose1_rotation"] >= 0) and np.all(
        action["pose1_rotation"] <= 1
    )

    # tokenize prompt
    # bypass None case because this function is also used in eval rollout, where we prepare prompt once and cache it
    if prompt is None:
        filled_prompt = None
    else:
        filled_prompt = prepare_prompt_rgb_only(
            prompt=prompt,
            prompt_assets=prompt_assets,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
        )
    return obs_rtn, action, filled_prompt


def collate_fn(
    samples_list: list[
        tuple[dict, dict | None, dict | None, Literal["suction", "spatula"]]
    ]
):
    """
    Collate a list of samples into a batch with L the leading dimension (i.e., `batch_first = False`).

    samples_list: A list of tuple(obs, action, prompt, ee_type), each with leading dimension L except `ee_type` is str.

    Will pad to max len in this batch.
    obs: (L + 1, ...) -> (L_max + 1, B, ...)
    action: (L, ...) -> (L_max, B, ...)
    action_mask: (L, ...) -> (L_max, B, ...)
    prompt padding is performed in model forward because we need to preserve the gradient.
    Returned `prompts` will be `tuple(raw_prompts, word_batch, image_batch)`, where `raw_prompts` preserves
    structures and orders of prompts, `word_batch` and `image_batch` are tensors that can be passed into models.
    Expected usage is passing `word_batch` and `image_batch` to corresponding encoders
    and then following `raw_prompts` to assemble prompt tokens.
    """
    B = len(samples_list)
    Lp1_max = max([U.get_batch_size(obs) for obs, _, _, _ in samples_list])
    L_max = max([U.get_batch_size(action) for _, action, _, _ in samples_list])
    assert L_max + 1 == Lp1_max

    # first pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    obs_structure = deepcopy(U.any_slice(samples_list[0][0], np.s_[0:1]))
    if samples_list[0][1] is not None:
        action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    else:
        action_structure = None
    padded_obs = U.any_to_datadict(
        U.any_stack(
            [
                U.any_concat(
                    [sample[0]]
                    + [U.any_zeros_like(obs_structure)]
                    * (L_max - U.get_batch_size(sample[1])),
                    dim=0,
                )
                for sample in samples_list
            ],
            dim=0,
        )
    )
    # bypass None case because this function will be used in rollout eval, where the first obs has no action yet
    if samples_list[0][1] is not None:
        padded_action = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [sample[1]]
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1])),
                        dim=0,
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
    else:
        padded_action = None

    # construct action_mask
    if samples_list[0][1] is not None:
        padded_action_mask = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [U.any_ones_like(action_structure)]
                        * U.get_batch_size(sample[1])
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1]))
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
        suction_ee = U.any_to_numpy(
            [sample[3] == "suction" for sample in samples_list]
        )[..., None]
        spatula_ee = U.any_to_numpy(
            [sample[3] == "spatula" for sample in samples_list]
        )[..., None]
        # action "push" is for spatula only, we don't want to calculate loss for suction
        padded_action_mask["push"] = padded_action_mask["push"] * spatula_ee
        # action "release" is for suction only, we don't want to calculate loss for spatula
        padded_action_mask["release"] = padded_action_mask["release"] * suction_ee
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
    else:
        padded_action_mask = None

    # collect prompt
    # bypass None case because prompt only need to be prepared once
    if samples_list[0][2] is not None:
        raw_prompts, word_batch, image_batch = [], [], []
        for _, _, prompt, _ in samples_list:
            raw_prompts.append(prompt)
            for token in prompt:
                if isinstance(token, int):
                    word_batch.append(token)
                elif isinstance(token, dict):
                    for v in token.values():
                        assert isinstance(v, np.ndarray)
                        assert v.ndim >= 3
                    image_batch.append(token)
        assert sum([len(prompt) for prompt in raw_prompts]) == len(word_batch) + len(
            image_batch
        )
        word_batch = U.any_stack(word_batch, dim=0)
        if len(image_batch) > 0:
            image_batch = U.any_to_datadict(U.stack_sequence_fields(image_batch))
    else:
        raw_prompts, word_batch, image_batch = None, None, None

    # convert to tensor and make L the first dim
    padded_obs = padded_obs.to_torch_tensor()
    padded_obs = U.any_transpose_first_two_axes(padded_obs)
    if padded_action is not None:
        padded_action = padded_action.to_torch_tensor()
        padded_action = U.any_transpose_first_two_axes(padded_action)
    if padded_action_mask is not None:
        padded_action_mask = padded_action_mask.to_torch_tensor()
        padded_action_mask = U.any_transpose_first_two_axes(padded_action_mask)
    if word_batch is not None:
        word_batch = U.any_to_torch_tensor(word_batch)
    if image_batch is not None and image_batch != []:
        image_batch = image_batch.to_torch_tensor()

    assert U.get_batch_size(padded_obs, strict=True) == L_max + 1
    assert U.get_batch_size(U.any_slice(padded_obs, np.s_[0]), strict=True) == B
    if padded_action is not None:
        assert U.get_batch_size(padded_action, strict=True) == L_max
        assert U.get_batch_size(U.any_slice(padded_action, np.s_[0]), strict=True) == B
    if padded_action_mask is not None:
        assert U.get_batch_size(padded_action_mask, strict=True) == L_max
        assert (
            U.get_batch_size(U.any_slice(padded_action_mask, np.s_[0]), strict=True)
            == B
        )
    return (
        padded_obs,
        padded_action,
        padded_action_mask,
        (raw_prompts, word_batch, image_batch),
    )


def collate_fn_bbox(samples_list):
    B = len(samples_list)
    Lp1_max = max([U.get_batch_size(obs, strict=True) for obs, _, _, _ in samples_list])
    L_max = max(
        [U.get_batch_size(action, strict=True) for _, action, _, _ in samples_list]
    )
    assert L_max + 1 == Lp1_max
    views = sorted(list(samples_list[0][0]["objects"]["cropped_img"].keys()))
    cropped_img_size = samples_list[0][0]["objects"]["cropped_img"][views[0]].shape[-1]

    obs_list = [sample[0] for sample in samples_list]
    padded_obs = collate_obs_with_bbox(obs_list=obs_list)

    # pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    if samples_list[0][1] is not None:
        action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    else:
        action_structure = None
    # bypass None case because this function will be used in rollout eval, where the first obs has no action yet
    if samples_list[0][1] is not None:
        padded_action = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [sample[1]]
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1])),
                        dim=0,
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
    else:
        padded_action = None

    # construct action_mask
    if samples_list[0][1] is not None:
        padded_action_mask = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [U.any_ones_like(action_structure)]
                        * U.get_batch_size(sample[1])
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1]))
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
    else:
        padded_action_mask = None

    # collect prompt
    # bypass None case because prompt only need to be prepared once
    if samples_list[0][2] is not None:
        raw_prompt_token_type, word_batch, image_batch = collate_prompt_with_bbox(
            views=views,
            raw_prompt_list=[sample[2] for sample in samples_list],
            cropped_img_size=cropped_img_size,
        )
    else:
        raw_prompt_token_type, word_batch, image_batch = None, None, None

    # convert to tensor and make L the first dim
    if padded_action is not None:
        padded_action = padded_action.to_torch_tensor()
        padded_action = U.any_transpose_first_two_axes(padded_action)
    if padded_action_mask is not None:
        padded_action_mask = padded_action_mask.to_torch_tensor()
        padded_action_mask = U.any_transpose_first_two_axes(padded_action_mask)

    if padded_action is not None:
        assert U.get_batch_size(padded_action, strict=True) == L_max
        assert U.get_batch_size(U.any_slice(padded_action, np.s_[0]), strict=True) == B
    if padded_action_mask is not None:
        assert U.get_batch_size(padded_action_mask, strict=True) == L_max
        assert (
            U.get_batch_size(U.any_slice(padded_action_mask, np.s_[0]), strict=True)
            == B
        )

    for k in padded_obs["objects"]["mask"].keys():
        padded_obs["objects"]["mask"][k][:-1].masked_fill_(
            torch.logical_and(
                padded_obs["objects"]["mask"][k][:-1].sum(axis=2) == 0,
                padded_action_mask["pose0_position"][:, :, 0] == 0,
            ).unsqueeze(-1),
            True,
        )

    task_names = [sample[3] for sample in samples_list]
    # dedupe task names and collect batch indices
    task_names_dedupe = list(set(task_names))
    task_names_dedupe.sort()
    task_name_to_batch_idx = {
        task_name: [i for i, t in enumerate(task_names) if t == task_name]
        for task_name in task_names_dedupe
    }

    # mask rotation for position only tasks
    rotation_mask = torch.zeros(B, dtype=bool)
    for task_name, batch_indices in task_name_to_batch_idx.items():
        # if task_name in ["rotate", "twist"]:
            rotation_mask[batch_indices] = True
    rotation_mask = rotation_mask.unsqueeze(0).unsqueeze(-1)
    padded_action_mask["pose0_rotation"] = (
        padded_action_mask["pose0_rotation"] * rotation_mask
    )
    padded_action_mask["pose1_rotation"] = (
        padded_action_mask["pose1_rotation"] * rotation_mask
    )

    # discard z coordinate
    padded_action["pose0_position"] = U.any_slice(
        padded_action["pose0_position"], np.s_[:, :, :2]
    )
    padded_action["pose1_position"] = U.any_slice(
        padded_action["pose1_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose0_position"] = U.any_slice(
        padded_action_mask["pose0_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose1_position"] = U.any_slice(
        padded_action_mask["pose1_position"], np.s_[:, :, :2]
    )

    return (
        padded_obs,
        padded_action,
        padded_action_mask,
        (raw_prompt_token_type, word_batch, image_batch),
        task_name_to_batch_idx,
    )


def collate_fn_rgb_only(samples_list):
    B = len(samples_list)
    Lp1_max = max([U.get_batch_size(obs, strict=True) for obs, _, _, _ in samples_list])
    L_max = max(
        [U.get_batch_size(action, strict=True) for _, action, _, _ in samples_list]
    )
    assert L_max + 1 == Lp1_max

    obs_list = [sample[0] for sample in samples_list]
    padded_obs = collate_obs_rgb_only(obs_list=obs_list)

    # pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    if samples_list[0][1] is not None:
        action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    else:
        action_structure = None
    # bypass None case because this function will be used in rollout eval, where the first obs has no action yet
    if samples_list[0][1] is not None:
        padded_action = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [sample[1]]
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1])),
                        dim=0,
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
    else:
        padded_action = None

    # construct action_mask
    if samples_list[0][1] is not None:
        padded_action_mask = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [U.any_ones_like(action_structure)]
                        * U.get_batch_size(sample[1])
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1]))
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
    else:
        padded_action_mask = None

    # collect prompt
    # bypass None case because prompt only need to be prepared once
    if samples_list[0][2] is not None:
        raw_prompt_token_type, word_batch, image_batch = collate_prompt_rgb_only(
            raw_prompt_list=[sample[2] for sample in samples_list],
        )
    else:
        raw_prompt_token_type, word_batch, image_batch = None, None, None

    # convert to tensor and make L the first dim
    if padded_action is not None:
        padded_action = padded_action.to_torch_tensor()
        padded_action = U.any_transpose_first_two_axes(padded_action)
    if padded_action_mask is not None:
        padded_action_mask = padded_action_mask.to_torch_tensor()
        padded_action_mask = U.any_transpose_first_two_axes(padded_action_mask)

    if padded_action is not None:
        assert U.get_batch_size(padded_action, strict=True) == L_max
        assert U.get_batch_size(U.any_slice(padded_action, np.s_[0]), strict=True) == B
    if padded_action_mask is not None:
        assert U.get_batch_size(padded_action_mask, strict=True) == L_max
        assert (
            U.get_batch_size(U.any_slice(padded_action_mask, np.s_[0]), strict=True)
            == B
        )

    task_names = [sample[3] for sample in samples_list]
    # dedupe task names and collect batch indices
    task_names_dedupe = list(set(task_names))
    task_names_dedupe.sort()
    task_name_to_batch_idx = {
        task_name: [i for i, t in enumerate(task_names) if t == task_name]
        for task_name in task_names_dedupe
    }

    # mask rotation for position only tasks
    rotation_mask = torch.zeros(B, dtype=bool)
    for task_name, batch_indices in task_name_to_batch_idx.items():
        # if task_name in ["rotate", "twist"]:
            rotation_mask[batch_indices] = True
    rotation_mask = rotation_mask.unsqueeze(0).unsqueeze(-1)
    padded_action_mask["pose0_rotation"] = (
        padded_action_mask["pose0_rotation"] * rotation_mask
    )
    padded_action_mask["pose1_rotation"] = (
        padded_action_mask["pose1_rotation"] * rotation_mask
    )

    # discard z coordinate
    padded_action["pose0_position"] = U.any_slice(
        padded_action["pose0_position"], np.s_[:, :, :2]
    )
    padded_action["pose1_position"] = U.any_slice(
        padded_action["pose1_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose0_position"] = U.any_slice(
        padded_action_mask["pose0_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose1_position"] = U.any_slice(
        padded_action_mask["pose1_position"], np.s_[:, :, :2]
    )

    return (
        padded_obs,
        padded_action,
        padded_action_mask,
        (raw_prompt_token_type, word_batch, image_batch),
        task_name_to_batch_idx,
    )

def collate_fn_vlm_rgb_only(samples_list, tokenizer):
    B = len(samples_list)
    # Lp1_max = max([U.get_batch_size(obs, strict=True) for obs, _, _, _ in samples_list])
    L_max = max(
        [U.get_batch_size(action, strict=True) for _, action, _, _ in samples_list]
    )
    # assert L_max + 1 == Lp1_max

    # obs_list = [sample[0] for sample in samples_list]
    # padded_obs = collate_obs_rgb_only(obs_list=obs_list)

    # pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    # Add the code below to handle multiple actions in the same batch. Currently we using only predicting one action.
    '''
    if samples_list[0][1] is not None:
        action_structure = deepcopy(U.any_slice(samples_list[0][1], np.s_[0:1]))
    else:
        action_structure = None
    # bypass None case because this function will be used in rollout eval, where the first obs has no action yet
    if samples_list[0][1] is not None:
        padded_action = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [sample[1]]
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1])),
                        dim=0,
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
    else:
        padded_action = None

    # construct action_mask
    if samples_list[0][1] is not None:
        padded_action_mask = U.any_to_datadict(
            U.any_stack(
                [
                    U.any_concat(
                        [U.any_ones_like(action_structure)]
                        * U.get_batch_size(sample[1])
                        + [U.any_zeros_like(action_structure)]
                        * (L_max - U.get_batch_size(sample[1]))
                    )
                    for sample in samples_list
                ],
                dim=0,
            )
        )
        padded_action_mask.map_structure(lambda x: x.astype(bool), inplace=True)
    else:
        padded_action_mask = None
    '''
    # this will always be 0 in for trajectory of length 1
    sample_indices = [np.random.choice(U.get_batch_size(action, strict=True)) for _, action, _, _ in samples_list]
    # actions are of shape [L, dim]
    # here instead of padding, we slice the action to sample only one action at a time
    padded_action = U.any_to_datadict(
        U.any_stack(
            [
                 U.any_slice(sample[1], np.s_[i:i+1]) \
                    for sample, i in zip(samples_list, sample_indices)
            ],
            dim=0,
        )
    )
    padded_action_mask = U.any_to_datadict(U.any_ones_like(padded_action))
    L_max = 1
    # collect prompt
    # bypass None case because prompt only need to be prepared once
    if samples_list[0][2] is not None:
        prompt_batch = collate_vlm_prompt_rgb_only(
            raw_prompt_list=[sample[2] for sample in samples_list],
            tokenizer=tokenizer,
            sample_indices=sample_indices,
        )
    else:
        prompt_batch = None

    # convert to tensor and make L the first dim
    if padded_action is not None:
        padded_action = padded_action.to_torch_tensor()
        padded_action = U.any_transpose_first_two_axes(padded_action)
    if padded_action_mask is not None:
        padded_action_mask = padded_action_mask.to_torch_tensor()
        padded_action_mask = U.any_transpose_first_two_axes(padded_action_mask)

    if padded_action is not None:
        assert U.get_batch_size(padded_action, strict=True) == L_max
        assert U.get_batch_size(U.any_slice(padded_action, np.s_[0]), strict=True) == B
    if padded_action_mask is not None:
        assert U.get_batch_size(padded_action_mask, strict=True) == L_max
        assert (
            U.get_batch_size(U.any_slice(padded_action_mask, np.s_[0]), strict=True)
            == B
        )

    task_names = [sample[3] for sample in samples_list]
    # dedupe task names and collect batch indices
    task_names_dedupe = list(set(task_names))
    task_names_dedupe.sort()
    task_name_to_batch_idx = {
        task_name: [i for i, t in enumerate(task_names) if t == task_name]
        for task_name in task_names_dedupe
    }

    # mask rotation for position only tasks
    rotation_mask = torch.zeros(B, dtype=bool)
    for task_name, batch_indices in task_name_to_batch_idx.items():
        # if task_name in ["rotate", "twist"]:
            rotation_mask[batch_indices] = True
    rotation_mask = rotation_mask.unsqueeze(0).unsqueeze(-1)
    padded_action_mask["pose0_rotation"] = (
        padded_action_mask["pose0_rotation"] * rotation_mask
    )
    padded_action_mask["pose1_rotation"] = (
        padded_action_mask["pose1_rotation"] * rotation_mask
    )

    # discard z coordinate
    padded_action["pose0_position"] = U.any_slice(
        padded_action["pose0_position"], np.s_[:, :, :2]
    )
    padded_action["pose1_position"] = U.any_slice(
        padded_action["pose1_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose0_position"] = U.any_slice(
        padded_action_mask["pose0_position"], np.s_[:, :, :2]
    )
    padded_action_mask["pose1_position"] = U.any_slice(
        padded_action_mask["pose1_position"], np.s_[:, :, :2]
    )

    return (
        None, # padded_obs
        padded_action,
        padded_action_mask,
        prompt_batch,
        task_name_to_batch_idx,
    )
