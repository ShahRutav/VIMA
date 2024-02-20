from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from tokenizers import Tokenizer
from einops import rearrange
import cv2
from PIL import Image
import vima.utils as U

from ..constants import OBJ_NAME_TO_ID, PLACEHOLDERS


def prepare_prompt_with_bbox(
    *,
    prompt: str,
    prompt_assets: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
    cropped_img_size: int = 224,
    add_obj_aug: bool = False,
    obj_aug_prob_map: dict[int, float] | None = None,
):
    if add_obj_aug:
        assert obj_aug_prob_map is not None
        assert isinstance(obj_aug_prob_map, dict)
        assert 0 in obj_aug_prob_map
        assert sum(obj_aug_prob_map.values()) == 1.0

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
            views = sorted(asset["rgb"].keys())
            rgb_h, rgb_w = asset["rgb"][views[0]].shape[-2:]
            # process to get bbox and cropped images of objects
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                assert isinstance(obj_info, dict)
                assert (
                    "obj_id" in obj_info
                    and "obj_color" in obj_info
                    and "obj_name" in obj_info
                )
                obj_id_to_name = {obj_info["obj_id"]: obj_info["obj_name"]}
            elif placeholder_type == "scene":
                assert isinstance(obj_info, list)
                obj_id_to_name = {
                    each_info["obj_id"]: each_info["obj_name"] for each_info in obj_info
                }
            else:
                raise ValueError(
                    f"unknown placeholder type {asset['placeholder_type']}"
                )
            objects = list(obj_id_to_name.keys())
            obj_id_to_global_id = {
                obj_id: OBJ_NAME_TO_ID[obj_name]
                for obj_id, obj_name in obj_id_to_name.items()
            }
            obj_repr = {
                "name": {view: [] for view in views},
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                # iterate over objects to get bbox and cropped_img for each object
                names = []
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    # get pixel indices corresponding to the object from segm
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    # bypass if no regions are found
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    names.append(obj_id_to_global_id[obj_id])
                    # find xmin, xmax, ymin, ymax
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    # bbox is in [x_center, y_center, h, w] format
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    # crop image
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    # pad the cropped image to square if necessary
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        # pad the shorter side to square
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    # resize the square cropped image to the desired size
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (cropped_img_size, cropped_img_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                names = np.asarray(names)  # (n_obj)
                bboxes = np.asarray(bboxes)  # (n_obj, 4)
                cropped_imgs = np.asarray(cropped_imgs)  # (n_obj, 3, H, W)
                assert bboxes.shape[0] == cropped_imgs.shape[0] == names.shape[0]
                # add to obj_repr
                obj_repr["name"][view] = names
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs

            if add_obj_aug:
                n_aug = np.random.choice(
                    list(obj_aug_prob_map.keys()), p=list(obj_aug_prob_map.values())
                )
                if n_aug > 0:
                    for view in views:
                        # randomly select object names
                        aug_obj_names = np.random.choice(
                            list(OBJ_NAME_TO_ID.values())[682:716], size=(n_aug,)
                        )
                        # randomly generate bboxes
                        while True:
                            # x_min and x_max
                            aug_obj_bbox_x = np.random.randint(
                                0, rgb_w, size=(n_aug, 2)
                            )
                            aug_obj_bbox_x = np.sort(aug_obj_bbox_x, axis=-1)
                            aug_obj_bbox_x_min, aug_obj_bbox_x_max = (
                                aug_obj_bbox_x[:, 0],
                                aug_obj_bbox_x[:, 1],
                            )
                            # y_min and y_max
                            aug_obj_bbox_y = np.random.randint(
                                0, rgb_h, size=(n_aug, 2)
                            )
                            aug_obj_bbox_y = np.sort(aug_obj_bbox_y, axis=-1)
                            aug_obj_bbox_y_min, aug_obj_bbox_y_max = (
                                aug_obj_bbox_y[:, 0],
                                aug_obj_bbox_y[:, 1],
                            )
                            # bbox is in [x_center, y_center, h, w] format
                            x_center, y_center = (
                                aug_obj_bbox_x_min + aug_obj_bbox_x_max
                            ) / 2, (aug_obj_bbox_y_min + aug_obj_bbox_y_max) / 2
                            x_center = x_center.astype(int)  # (n_aug,)
                            y_center = y_center.astype(int)  # (n_aug,)
                            h = (aug_obj_bbox_y_max - aug_obj_bbox_y_min).astype(
                                int
                            )  # (n_aug,)
                            w = (aug_obj_bbox_x_max - aug_obj_bbox_x_min).astype(
                                int
                            )  # (n_aug,)
                            # check if the bbox is valid, h and w should >= 2
                            if np.all(h >= 2) and np.all(w >= 2):
                                break
                        aug_obj_bboxes = np.stack(
                            [x_center, y_center, h, w], axis=-1
                        )  # (n_aug, 4)
                        # crop images
                        aug_obj_cropped_imgs = []
                        for idx in range(n_aug):
                            x_min, x_max = (
                                aug_obj_bbox_x_min[idx],
                                aug_obj_bbox_x_max[idx],
                            )
                            y_min, y_max = (
                                aug_obj_bbox_y_min[idx],
                                aug_obj_bbox_y_max[idx],
                            )
                            aug_crop_img_raw = asset["rgb"][view][
                                :, y_min:y_max, x_min:x_max
                            ]  # (3, h, w)
                            # pad the raw cropped image to square with half probability
                            if (
                                aug_crop_img_raw.shape[1] != aug_crop_img_raw.shape[2]
                                and np.random.rand() < 0.5
                            ):
                                # pad the shorter side to square
                                diff = abs(
                                    aug_crop_img_raw.shape[1]
                                    - aug_crop_img_raw.shape[2]
                                )
                                pad_before, pad_after = int(diff / 2), diff - int(
                                    diff / 2
                                )
                                if (
                                    aug_crop_img_raw.shape[1]
                                    > aug_crop_img_raw.shape[2]
                                ):
                                    pad_width = (
                                        (0, 0),
                                        (0, 0),
                                        (pad_before, pad_after),
                                    )
                                else:
                                    pad_width = (
                                        (0, 0),
                                        (pad_before, pad_after),
                                        (0, 0),
                                    )
                                aug_crop_img_raw = np.pad(
                                    aug_crop_img_raw,
                                    pad_width,
                                    mode="constant",
                                    constant_values=0,
                                )
                                assert (
                                    aug_crop_img_raw.shape[1]
                                    == aug_crop_img_raw.shape[2]
                                ), "INTERNAL"
                            # resize the square cropped image to the desired size
                            aug_crop_img = rearrange(aug_crop_img_raw, "c h w -> h w c")
                            aug_crop_img = np.asarray(aug_crop_img)
                            aug_crop_img = cv2.resize(
                                aug_crop_img,
                                (cropped_img_size, cropped_img_size),
                                interpolation=cv2.INTER_AREA,
                            )
                            aug_crop_img = rearrange(aug_crop_img, "h w c -> c h w")
                            aug_obj_cropped_imgs.append(aug_crop_img)
                        aug_obj_cropped_imgs = np.stack(
                            aug_obj_cropped_imgs, axis=0
                        )  # (n_aug, 3, H, W)

                        # generation finished, update obj_repr
                        obj_repr["name"][view] = np.concatenate(
                            [obj_repr["name"][view], aug_obj_names], axis=0
                        )
                        obj_repr["bbox"][view] = np.concatenate(
                            [obj_repr["bbox"][view], aug_obj_bboxes], axis=0
                        )
                        obj_repr["cropped_img"][view] = np.concatenate(
                            [obj_repr["cropped_img"][view], aug_obj_cropped_imgs],
                            axis=0,
                        )
            filled_prompt.append(obj_repr)
    return filled_prompt

def prepare_vlm_prompt_rgb_only(
    *,
    view: str = 'top',
    rgb: dict,
    text_prompt: str,
    prompt_assets: dict,
    processor: ProcessorMixin,
    add_special_tokens: bool,
):
    images = [Image.fromarray(rearrange(img, "c h w -> h w c")) for img in rgb[view]]
    # we do not maintain any history. Each image will be randomly sampled in the collate_fn
    filled_prompt = processor(text=[text_prompt]*len(images), images=images, return_tensors="pt")
    # prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    # assert set(prompt_assets.keys()) == set(
    #     [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    # )
    # filled_prompt = []
    # for id, token in zip(prompt_ids, prompt_tokens):
    #     if token not in PLACEHOLDERS:
    #         # an indexed word
    #         assert "{" not in token and "}" not in token
    #         filled_prompt.append(id)
    #     else:
    #         # a multimodal placeholder
    #         assert token.startswith("{") and token.endswith("}")
    #         asset_name = token[1:-1]
    #         assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
    #         asset = prompt_assets[asset_name]
    #         # resize to (64, 128)
    #         rgb = {
    #             k: rearrange(
    #                 cv2.resize(
    #                     np.asarray(rearrange(v, "c h w -> h w c")),
    #                     (128, 64),
    #                     interpolation=cv2.INTER_AREA,
    #                 ),
    #                 "h w c -> c h w",
    #             )
    #             for k, v in asset["rgb"].items()
    #         }
    #         obj_repr = {"rgb": rgb}
    #         filled_prompt.append(obj_repr)
    return filled_prompt

def prepare_prompt_rgb_only(
    *,
    prompt: str,
    prompt_assets: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
):
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
            # resize to (64, 128)
            rgb = {
                k: rearrange(
                    cv2.resize(
                        np.asarray(rearrange(v, "c h w -> h w c")),
                        (128, 64),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "h w c -> c h w",
                )
                for k, v in asset["rgb"].items()
            }
            obj_repr = {"rgb": rgb}
            filled_prompt.append(obj_repr)
    return filled_prompt


def prepare_obs_with_bbox(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    meta: dict,
    cropped_img_size: int = 224,
    add_obj_aug: bool = False,
    obj_aug_prob_map: dict[int, float] | None = None,
):
    if add_obj_aug:
        assert obj_aug_prob_map is not None
        assert isinstance(obj_aug_prob_map, dict)
        assert 0 in obj_aug_prob_map
        assert sum(obj_aug_prob_map.values()) == 1.0

    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    rgb_h, rgb_w = rgb_dict[views[0]].shape[-2:]
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys())

    L_obs = U.get_batch_size(obs)

    # process obs
    obs_rtn = {
        "ee": obs["ee"],
        "objects": {
            "name": {view: [] for view in views},
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = U.any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = U.any_slice(segm_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]
            # iterate over objects to get bbox and cropped_img for each object
            names = []
            bboxes = []
            cropped_imgs = []
            n_pad = 0
            for obj_id in objects:
                # get pixel indices corresponding to the object from segm
                ys, xs = np.nonzero(segm_this_view == obj_id)
                # bypass if no regions are found
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue
                # append name
                names.append(OBJ_NAME_TO_ID[meta["obj_id_to_info"][obj_id]["obj_name"]])
                # find xmin, xmax, ymin, ymax
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                # bbox is in [x_center, y_center, h, w] format
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                # crop image
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                # pad the cropped image to square if necessary
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    # pad the shorter side to square
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                # resize the square cropped image to the desired size
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (cropped_img_size, cropped_img_size),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            names = np.asarray(names)  # (n_obj)
            bboxes = np.asarray(bboxes)  # (n_obj, 4)
            cropped_imgs = np.asarray(cropped_imgs)  # (n_obj, 3, H, W)
            mask = np.ones(len(bboxes), dtype=bool)  # (n_obj)
            # pad bboxes and cropped_imgs to match the number of objects
            if n_pad > 0:
                names = np.concatenate(
                    [names, np.zeros(n_pad, dtype=names.dtype)], axis=0
                )  # (n_obj + n_pad)
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )  # (n_obj + n_pad, 4)
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, cropped_img_size, cropped_img_size),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )  # (n_obj + n_pad, 3, H, W)
                mask = np.concatenate(
                    [mask, np.zeros(n_pad, dtype=bool)], axis=0
                )  # (n_obj + n_pad)
            assert (
                names.shape[0]
                == bboxes.shape[0]
                == cropped_imgs.shape[0]
                == mask.shape[0]
                == len(objects)
            )
            # add to obs_rtn
            obs_rtn["objects"]["name"][view].append(names)
            obs_rtn["objects"]["bbox"][view].append(bboxes)
            obs_rtn["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_rtn["objects"]["mask"][view].append(mask)
    # stack obs
    for view in views:
        obs_rtn["objects"]["name"][view] = np.stack(
            obs_rtn["objects"]["name"][view], axis=0
        )
        obs_rtn["objects"]["bbox"][view] = np.stack(
            obs_rtn["objects"]["bbox"][view], axis=0
        )
        obs_rtn["objects"]["cropped_img"][view] = np.stack(
            obs_rtn["objects"]["cropped_img"][view], axis=0
        )
        obs_rtn["objects"]["mask"][view] = np.stack(
            obs_rtn["objects"]["mask"][view], axis=0
        )
        assert obs_rtn["objects"]["bbox"][view].shape == (L_obs, len(objects), 4)
        assert obs_rtn["objects"]["cropped_img"][view].shape == (
            L_obs,
            len(objects),
            3,
            cropped_img_size,
            cropped_img_size,
        )
        assert obs_rtn["objects"]["mask"][view].shape == (L_obs, len(objects))
        assert obs_rtn["objects"]["name"][view].shape == (L_obs, len(objects))
    assert U.get_batch_size(obs_rtn, strict=True) == L_obs

    if add_obj_aug:
        n_aug = np.random.choice(
            list(obj_aug_prob_map.keys()), p=list(obj_aug_prob_map.values())
        )
        if n_aug > 0:
            for view in views:
                # randomly select object names
                aug_obj_names = np.random.choice(
                    list(OBJ_NAME_TO_ID.values())[682:716], size=(L_obs, n_aug)
                )
                # valid object masks
                aug_obj_masks = np.ones((L_obs, n_aug), dtype=bool)
                # randomly generate bboxes
                while True:
                    # x_min and x_max
                    aug_obj_bbox_x = np.random.randint(0, rgb_w, size=(L_obs, n_aug, 2))
                    aug_obj_bbox_x = np.sort(aug_obj_bbox_x, axis=-1)
                    aug_obj_bbox_x_min, aug_obj_bbox_x_max = (
                        aug_obj_bbox_x[:, :, 0],
                        aug_obj_bbox_x[:, :, 1],
                    )
                    # y_min and y_max
                    aug_obj_bbox_y = np.random.randint(0, rgb_h, size=(L_obs, n_aug, 2))
                    aug_obj_bbox_y = np.sort(aug_obj_bbox_y, axis=-1)
                    aug_obj_bbox_y_min, aug_obj_bbox_y_max = (
                        aug_obj_bbox_y[:, :, 0],
                        aug_obj_bbox_y[:, :, 1],
                    )
                    # bbox is in [x_center, y_center, h, w] format
                    x_center, y_center = (
                        aug_obj_bbox_x_min + aug_obj_bbox_x_max
                    ) / 2, (aug_obj_bbox_y_min + aug_obj_bbox_y_max) / 2
                    x_center = x_center.astype(int)  # (L_obs, n_aug)
                    y_center = y_center.astype(int)  # (L_obs, n_aug)
                    h = (aug_obj_bbox_y_max - aug_obj_bbox_y_min).astype(
                        int
                    )  # (L_obs, n_aug)
                    w = (aug_obj_bbox_x_max - aug_obj_bbox_x_min).astype(
                        int
                    )  # (L_obs, n_aug)
                    # check if the bbox is valid, h and w should >= 2
                    if np.all(h >= 2) and np.all(w >= 2):
                        break
                aug_obj_bboxes = np.stack(
                    [x_center, y_center, h, w], axis=-1
                )  # (L_obs, n_aug, 4)
                # crop images
                aug_obj_cropped_imgs = []
                for l in range(L_obs):
                    aug_obj_cropped_imgs_l = []
                    rgb_this_step = rgb_dict[view][l]  # (3, H, W)
                    for idx in range(n_aug):
                        x_min, x_max = (
                            aug_obj_bbox_x_min[l, idx],
                            aug_obj_bbox_x_max[l, idx],
                        )
                        y_min, y_max = (
                            aug_obj_bbox_y_min[l, idx],
                            aug_obj_bbox_y_max[l, idx],
                        )
                        aug_crop_img_raw = rgb_this_step[
                            :, y_min:y_max, x_min:x_max
                        ]  # (3, h, w)
                        # pad the raw cropped image to square with half probability
                        if (
                            aug_crop_img_raw.shape[1] != aug_crop_img_raw.shape[2]
                            and np.random.rand() < 0.5
                        ):
                            # pad the shorter side to square
                            diff = abs(
                                aug_crop_img_raw.shape[1] - aug_crop_img_raw.shape[2]
                            )
                            pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                            if aug_crop_img_raw.shape[1] > aug_crop_img_raw.shape[2]:
                                pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                            else:
                                pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                            aug_crop_img_raw = np.pad(
                                aug_crop_img_raw,
                                pad_width,
                                mode="constant",
                                constant_values=0,
                            )
                            assert (
                                aug_crop_img_raw.shape[1] == aug_crop_img_raw.shape[2]
                            ), "INTERNAL"
                        # resize the square cropped image to the desired size
                        aug_crop_img = rearrange(aug_crop_img_raw, "c h w -> h w c")
                        aug_crop_img = np.asarray(aug_crop_img)
                        aug_crop_img = cv2.resize(
                            aug_crop_img,
                            (cropped_img_size, cropped_img_size),
                            interpolation=cv2.INTER_AREA,
                        )
                        aug_crop_img = rearrange(aug_crop_img, "h w c -> c h w")
                        aug_obj_cropped_imgs_l.append(aug_crop_img)
                    aug_obj_cropped_imgs_l = np.stack(
                        aug_obj_cropped_imgs_l, axis=0
                    )  # (n_aug, 3, cropped_img_size, cropped_img_size)
                    aug_obj_cropped_imgs.append(aug_obj_cropped_imgs_l)
                aug_obj_cropped_imgs = np.stack(
                    aug_obj_cropped_imgs, axis=0
                )  # (L_obs, n_aug, 3, cropped_img_size, cropped_img_size)

                # generation finished, update the obs_rtn
                obs_rtn["objects"]["name"][view] = np.concatenate(
                    [obs_rtn["objects"]["name"][view], aug_obj_names], axis=1
                )
                obs_rtn["objects"]["bbox"][view] = np.concatenate(
                    [obs_rtn["objects"]["bbox"][view], aug_obj_bboxes], axis=1
                )
                obs_rtn["objects"]["cropped_img"][view] = np.concatenate(
                    [obs_rtn["objects"]["cropped_img"][view], aug_obj_cropped_imgs],
                    axis=1,
                )
                obs_rtn["objects"]["mask"][view] = np.concatenate(
                    [obs_rtn["objects"]["mask"][view], aug_obj_masks], axis=1
                )

                assert obs_rtn["objects"]["bbox"][view].shape == (
                    L_obs,
                    len(objects) + n_aug,
                    4,
                )
                assert obs_rtn["objects"]["cropped_img"][view].shape == (
                    L_obs,
                    len(objects) + n_aug,
                    3,
                    cropped_img_size,
                    cropped_img_size,
                )
                assert obs_rtn["objects"]["mask"][view].shape == (
                    L_obs,
                    len(objects) + n_aug,
                )
                assert obs_rtn["objects"]["name"][view].shape == (
                    L_obs,
                    len(objects) + n_aug,
                )

    return obs_rtn

def prepare_vlm_obs_rgb_only(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    img_h: int, img_w: int,
):
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    L_ee = U.get_batch_size(obs["ee"], strict=True)
    L_rgb = U.get_batch_size(rgb_dict, strict=True)
    if L_ee > L_rgb:
        obs["ee"] = U.any_slice(obs["ee"], np.s_[:L_rgb])
    elif L_ee < L_rgb:
        rgb_dict = U.any_slice(rgb_dict, np.s_[:L_ee])
    rgb_dict = {
        k: U.any_stack(
            [
                np.pad(
                    x,
                    (
                        (0, 0),
                        (max(0, (img_h - x.shape[1]) // 2), max(0, (img_h - x.shape[1]) // 2)),
                        (max(0, (img_w - x.shape[2]) // 2), max(0, (img_w - x.shape[2]) // 2)),
                    ),
                    mode="constant", constant_values=255,
                ) for x in v
            ],
            dim=0,
        )
        for k, v in rgb_dict.items()
    }
    # process obs
    obs_rtn = {
        "ee": obs["ee"],
        "rgb": rgb_dict,
    }
    return obs_rtn

def prepare_obs_rgb_only(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
):
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    L_ee = U.get_batch_size(obs["ee"], strict=True)
    L_rgb = U.get_batch_size(rgb_dict, strict=True)
    if L_ee > L_rgb:
        obs["ee"] = U.any_slice(obs["ee"], np.s_[:L_rgb])
    elif L_ee < L_rgb:
        rgb_dict = U.any_slice(rgb_dict, np.s_[:L_ee])
    rgb_dict = {
        k: U.any_stack(
            [
                rearrange(
                    cv2.resize(
                        np.asarray(rearrange(x, "c h w -> h w c")),
                        (128, 64),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "h w c -> c h w",
                )
                for x in v
            ],
            dim=0,
        )
        for k, v in rgb_dict.items()
    }
    # process obs
    obs_rtn = {
        "ee": obs["ee"],
        "rgb": rgb_dict,
    }
    return obs_rtn


def collate_prompt_with_bbox(*, views, raw_prompt_list, cropped_img_size):
    # find max number of objects each view
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt_list:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["name"][view])
                    )

    # now process
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt_list:
        token_type = []  # 0 for word, 1 for image
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {view: len(token["name"][view]) for view in views}
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (
                                n_objs_to_pad[view],
                                3,
                                cropped_img_size,
                                cropped_img_size,
                            ),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "name": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=np.int64)
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                        for view in views
                    },
                }
                token = U.any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = U.any_stack(word_batch, dim=0)
    word_batch = U.any_to_torch_tensor(word_batch)

    if len(image_batch) > 0:
        image_batch = U.any_to_datadict(U.stack_sequence_fields(image_batch))
        image_batch = image_batch.to_torch_tensor()
    else:
        image_batch = None
    return raw_prompt_token_type, word_batch, image_batch

def pad_sequence(sequences, batch_first=False, padding_value=0, padding_side="right"):
    max_size = max([seq.size(0) for seq in sequences])
    out_dims = (len(sequences), max_size) + sequences[0].size()[1:]
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    if padding_side == "right":
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, ...] = tensor
    elif padding_side == "left":
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[-length:, ...] = tensor
    return out_tensor

def collate_vlm_prompt_rgb_only(*, tokenizer, raw_prompt_list, sample_indices):
    # there are input_ids, attention_mask, pixel_values
    # pad input_ids and make corresponding attention_mask
    assert len(raw_prompt_list) == len(sample_indices)
    raw_prompt = [U.any_slice(prompt, np.s_[i]) for prompt, i in zip(raw_prompt_list, sample_indices)]

    input_ids = [prompt["input_ids"] for prompt in raw_prompt]
    attention_mask = [prompt["attention_mask"] for prompt in raw_prompt]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side=tokenizer.padding_side)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side=tokenizer.padding_side)
    print('padded_sequence, attention_mask')
    print(input_ids, attention_mask)

    for ind in range(len(raw_prompt)):
        raw_prompt[ind]["input_ids"] = input_ids[ind]
        raw_prompt[ind]["attention_mask"] = attention_mask[ind]

    batch = U.any_to_datadict(
        U.any_stack(raw_prompt, dim=0)
    )
    return batch

def collate_prompt_rgb_only(*, raw_prompt_list):
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt_list:
        token_type = []  # 0 for word, 1 for image
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = U.any_stack(word_batch, dim=0)
    word_batch = U.any_to_torch_tensor(word_batch)

    if len(image_batch) > 0:
        image_batch = U.any_to_datadict(U.stack_sequence_fields(image_batch))
        image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch


def collate_obs_with_bbox(*, obs_list):
    B = len(obs_list)
    Lp1_max = max([U.get_batch_size(obs, strict=True) for obs in obs_list])
    n_objs_max = max(
        U.get_batch_size(U.any_slice(obs["objects"], np.s_[0]), strict=True)
        for obs in obs_list
    )
    views = sorted(list(obs_list[0]["objects"]["cropped_img"].keys()))
    cropped_img_size = obs_list[0]["objects"]["cropped_img"][views[0]].shape[-1]

    # fist pad obj set in each trajectory to max number of objs in this batch
    for b in range(B):
        obs = obs_list[b]
        n_objs = U.get_batch_size(U.any_slice(obs["objects"], np.s_[0]), strict=True)
        n_objs_pad = n_objs_max - n_objs
        L_obs = U.get_batch_size(obs, strict=True)
        objs_pad = {
            "bbox": {
                view: np.zeros((L_obs, n_objs_pad, 4), dtype=np.int64) for view in views
            },
            "cropped_img": {
                view: np.zeros(
                    (L_obs, n_objs_pad, 3, cropped_img_size, cropped_img_size),
                    dtype=np.uint8,
                )
                for view in views
            },
            "mask": {
                view: np.zeros((L_obs, n_objs_pad), dtype=bool) for view in views
            },
            "name": {
                view: np.zeros((L_obs, n_objs_pad), dtype=np.int64) for view in views
            },
        }
        obs["objects"] = U.any_concat([obs["objects"], objs_pad], dim=1)
        obs_list[b] = obs

    # pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    obs_structure = deepcopy(U.any_slice(obs_list[0], np.s_[0:1]))
    padded_obs = U.any_to_datadict(
        U.any_stack(
            [
                U.any_concat(
                    [obs]
                    + [U.any_zeros_like(obs_structure)]
                    * (Lp1_max - U.get_batch_size(obs)),
                    dim=0,
                )
                for obs in obs_list
            ],
            dim=0,
        )
    )

    # convert to tensor and make L the first dim
    padded_obs = padded_obs.to_torch_tensor()
    padded_obs = U.any_transpose_first_two_axes(padded_obs)

    assert U.get_batch_size(padded_obs, strict=True) == Lp1_max
    assert U.get_batch_size(U.any_slice(padded_obs, np.s_[0]), strict=True) == B
    assert (
        U.get_batch_size(
            U.any_slice(U.any_slice(padded_obs["objects"], np.s_[0]), np.s_[0]),
            strict=True,
        )
        == n_objs_max
    )
    return padded_obs


def collate_obs_rgb_only(*, obs_list):
    B = len(obs_list)
    Lp1_max = max([U.get_batch_size(obs, strict=True) for obs in obs_list])

    # pad each trajectory to L_max in this batch
    # note that we slice instead of index to keep the first dim
    obs_structure = deepcopy(U.any_slice(obs_list[0], np.s_[0:1]))
    padded_obs = U.any_to_datadict(
        U.any_stack(
            [
                U.any_concat(
                    [obs]
                    + [U.any_zeros_like(obs_structure)]
                    * (Lp1_max - U.get_batch_size(obs)),
                    dim=0,
                )
                for obs in obs_list
            ],
            dim=0,
        )
    )

    # convert to tensor and make L the first dim
    padded_obs = padded_obs.to_torch_tensor()
    padded_obs = U.any_transpose_first_two_axes(padded_obs)

    assert U.get_batch_size(padded_obs, strict=True) == Lp1_max
    assert U.get_batch_size(U.any_slice(padded_obs, np.s_[0]), strict=True) == B
    return padded_obs


def prepare_obs_with_bbox_from_detection(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    meta: dict,
    cropped_img_size: int = 224,
    add_obj_aug: bool = False,
    obj_aug_prob_map: dict[int, float] | None = None,
    detection_model,
):
    """
    Only used during inference.
    """
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    views = sorted(rgb_dict.keys())

    L_obs = U.get_batch_size(obs)

    # use detection model to process rgbs
    detect_results = {view: detection_model(v)[0] for view, v in rgb_dict.items()}
    max_objs = {
        view: max(len(x["name"]) for x in detect_results[view]) for view in views
    }
    max_objs = max(v for v in max_objs.values())

    # process obs
    obs_rtn = {
        "ee": obs["ee"],
        "objects": {
            "name": {view: [] for view in views},
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = U.any_slice(rgb_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            detect_result_this_view = detect_results[view][
                l
            ]  # a dict with keys: name, bbox, segm
            # iterate over objects to get bbox and cropped_img for each object
            names = []
            bboxes = []
            cropped_imgs = []
            for obj_id, obj_bbox in zip(
                detect_result_this_view["name"], detect_result_this_view["bbox"]
            ):
                # append name
                names.append(obj_id)
                # bbox is in [x_center, y_center, h, w] format
                bboxes.append(obj_bbox)
                # crop image
                x_center, y_center, h, w = obj_bbox
                xmin = int(x_center - w / 2)
                xmax = int(x_center + w / 2)
                ymin = int(y_center - h / 2)
                ymax = int(y_center + h / 2)
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                # pad the cropped image to square if necessary
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    # pad the shorter side to square
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                # resize the square cropped image to the desired size
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (cropped_img_size, cropped_img_size),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            if len(names) == 0:
                names = np.zeros((0,), dtype=np.int64)
            else:
                names = np.asarray(names)  # (n_obj)
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 4), dtype=np.int64)
            else:
                bboxes = np.asarray(bboxes)  # (n_obj, 4)
            if len(cropped_imgs) == 0:
                cropped_imgs = np.zeros(
                    (0, 3, cropped_img_size, cropped_img_size), dtype=np.uint8
                )
            else:
                cropped_imgs = np.asarray(cropped_imgs)  # (n_obj, 3, H, W)
            names = np.asarray(names)  # (n_obj)
            bboxes = np.asarray(bboxes)  # (n_obj, 4)
            cropped_imgs = np.asarray(cropped_imgs)  # (n_obj, 3, H, W)
            mask = np.ones(len(bboxes), dtype=bool)  # (n_obj)
            # pad bboxes and cropped_imgs to match the number of objects
            n_pad = max_objs - len(detect_result_this_view["name"])
            if n_pad > 0:
                names = np.concatenate(
                    [names, np.zeros(n_pad, dtype=names.dtype)], axis=0
                )  # (n_obj + n_pad)
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )  # (n_obj + n_pad, 4)
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, cropped_img_size, cropped_img_size),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )  # (n_obj + n_pad, 3, H, W)
                mask = np.concatenate(
                    [mask, np.zeros(n_pad, dtype=bool)], axis=0
                )  # (n_obj + n_pad)
            assert (
                names.shape[0]
                == bboxes.shape[0]
                == cropped_imgs.shape[0]
                == mask.shape[0]
                == max_objs
            )
            # add to obs_rtn
            obs_rtn["objects"]["name"][view].append(names)
            obs_rtn["objects"]["bbox"][view].append(bboxes)
            obs_rtn["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_rtn["objects"]["mask"][view].append(mask)
    # stack obs
    for view in views:
        obs_rtn["objects"]["name"][view] = np.stack(
            obs_rtn["objects"]["name"][view], axis=0
        )
        obs_rtn["objects"]["bbox"][view] = np.stack(
            obs_rtn["objects"]["bbox"][view], axis=0
        )
        obs_rtn["objects"]["cropped_img"][view] = np.stack(
            obs_rtn["objects"]["cropped_img"][view], axis=0
        )
        obs_rtn["objects"]["mask"][view] = np.stack(
            obs_rtn["objects"]["mask"][view], axis=0
        )
        assert obs_rtn["objects"]["bbox"][view].shape == (L_obs, max_objs, 4)
        assert obs_rtn["objects"]["cropped_img"][view].shape == (
            L_obs,
            max_objs,
            3,
            cropped_img_size,
            cropped_img_size,
        )
        assert obs_rtn["objects"]["mask"][view].shape == (L_obs, max_objs)
        assert obs_rtn["objects"]["name"][view].shape == (L_obs, max_objs)
    assert U.get_batch_size(obs_rtn, strict=True) == L_obs
    return obs_rtn


def prepare_prompt_with_bbox_from_detection(
    *,
    prompt: str,
    prompt_assets: dict,
    tokenizer: Tokenizer,
    add_special_tokens: bool,
    cropped_img_size: int = 224,
    add_obj_aug: bool = False,
    obj_aug_prob_map: dict[int, float] | None = None,
    detection_model,
):
    """
    Only used during inference.
    """
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
            views = sorted(asset["rgb"].keys())
            # process to get bbox and cropped images of objects
            obj_repr = {
                "name": {view: [] for view in views},
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]

                # use detection model to process rgb
                detect_result = detection_model(rgb_this_view)[0][
                    0
                ]  # a dict of name, bbox, segm

                # iterate over objects to get bbox and cropped_img for each object
                names = []
                bboxes = []
                cropped_imgs = []
                for obj_id, obj_bbox in zip(
                    detect_result["name"], detect_result["bbox"]
                ):
                    names.append(obj_id)
                    # bbox is in [x_center, y_center, h, w] format
                    bboxes.append(obj_bbox)
                    # crop image
                    x_center, y_center, h, w = obj_bbox
                    xmin = int(x_center - w / 2)
                    xmax = int(x_center + w / 2)
                    ymin = int(y_center - h / 2)
                    ymax = int(y_center + h / 2)
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    # pad the cropped image to square if necessary
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        # pad the shorter side to square
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    # resize the square cropped image to the desired size
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (cropped_img_size, cropped_img_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                if len(names) == 0:
                    names = np.zeros((0,), dtype=np.int64)
                else:
                    names = np.asarray(names)  # (n_obj)
                if len(bboxes) == 0:
                    bboxes = np.zeros((0, 4), dtype=np.int64)
                else:
                    bboxes = np.asarray(bboxes)  # (n_obj, 4)
                if len(cropped_imgs) == 0:
                    cropped_imgs = np.zeros(
                        (0, 3, cropped_img_size, cropped_img_size), dtype=np.uint8
                    )
                else:
                    cropped_imgs = np.asarray(cropped_imgs)  # (n_obj, 3, H, W)
                assert bboxes.shape[0] == cropped_imgs.shape[0] == names.shape[0]
                # add to obj_repr
                obj_repr["name"][view] = names
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs

            filled_prompt.append(obj_repr)
    return filled_prompt
