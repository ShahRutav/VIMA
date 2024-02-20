from __future__ import annotations

import os
import copy
import yaml

import numpy as np
from PIL import Image
from termcolor import colored
from easydict import EasyDict
from tokenizers import Tokenizer
from tokenizers import AddedToken
import transformers
from einops import rearrange
from omegaconf import OmegaConf
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse

import vima_bench
from vima.data.process.dm_only import prepare_sample_vlm_rgb_only, collate_fn_vlm_rgb_only

os.environ["TOKENIZERS_PARALLELISM"] = "true"


_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{base_obj}", **_kwargs),
    AddedToken("{base_obj_1}", **_kwargs),
    AddedToken("{base_obj_2}", **_kwargs),
    AddedToken("{dragged_obj}", **_kwargs),
    AddedToken("{dragged_obj_1}", **_kwargs),
    AddedToken("{dragged_obj_2}", **_kwargs),
    AddedToken("{dragged_obj_3}", **_kwargs),
    AddedToken("{dragged_obj_4}", **_kwargs),
    AddedToken("{dragged_obj_5}", **_kwargs),
    AddedToken("{swept_obj}", **_kwargs),
    AddedToken("{bounds}", **_kwargs),
    AddedToken("{constraint}", **_kwargs),
    AddedToken("{scene}", **_kwargs),
    AddedToken("{demo_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_blicker_obj_3}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
    AddedToken("{start_scene}", **_kwargs),
    AddedToken("{end_scene}", **_kwargs),
    AddedToken("{before_twist_1}", **_kwargs),
    AddedToken("{after_twist_1}", **_kwargs),
    AddedToken("{before_twist_2}", **_kwargs),
    AddedToken("{after_twist_2}", **_kwargs),
    AddedToken("{before_twist_3}", **_kwargs),
    AddedToken("{after_twist_3}", **_kwargs),
    AddedToken("{frame_0}", **_kwargs),
    AddedToken("{frame_1}", **_kwargs),
    AddedToken("{frame_2}", **_kwargs),
    AddedToken("{frame_3}", **_kwargs),
    AddedToken("{frame_4}", **_kwargs),
    AddedToken("{frame_5}", **_kwargs),
    AddedToken("{frame_6}", **_kwargs),
    AddedToken("{ring}", **_kwargs),
    AddedToken("{hanoi_stand}", **_kwargs),
    AddedToken("{start_scene_1}", **_kwargs),
    AddedToken("{end_scene_1}", **_kwargs),
    AddedToken("{start_scene_2}", **_kwargs),
    AddedToken("{end_scene_2}", **_kwargs),
    AddedToken("{start_scene_3}", **_kwargs),
    AddedToken("{end_scene_3}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)

def plot_actions(action_to_plot, file_name):
    '''
        Plot the action with label pick and place in the image and save the image
        pick is the one with pose0_position and place is the one with pose1_position
    '''
    video = []
    for index, rgb in enumerate(action_to_plot['rgb']):
        # convert chw to hwc
        rgb = np.transpose(rgb, (1, 2, 0))
        h, w, _ = rgb.shape
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pos0 = action_to_plot['pose0_position'][index]
        pos1 = action_to_plot['pose1_position'][index]

        pos0 = np.asarray(pos0.squeeze()) * np.array([h, w])
        pos1 = np.asarray(pos1.squeeze()) * np.array([h, w])

        rgb = cv2.circle(rgb, tuple(pos0.astype(np.int32)[::-1]), 5, (0, 0, 255), 2)
        rgb = cv2.circle(rgb, tuple(pos1.astype(np.int32)[::-1]), 5, (0, 255, 0), 2)
        rgb = cv2.putText(
            rgb,
            " pick",
            org=tuple(pos0.astype(np.int32)[::-1]),
            fontScale=0.5,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        rgb = cv2.putText(
            rgb,
            " place",
            org=tuple(pos1.astype(np.int32)[::-1]),
            fontScale=0.5,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        video.append(rgb)
    # stack the images in the video list along the height axis
    video = np.vstack(video)
    # save it as an image
    video = Image.fromarray(video)
    video.save(file_name)
    print(f"Saved the video at {file_name}")
    return

@torch.no_grad()
def main(eval_cfg):
    assert eval_cfg.partition in ALL_PARTITIONS, f"invalid partition {eval_cfg.partition}. Allowed: {ALL_PARTITIONS}"
    assert eval_cfg.task in PARTITION_TO_SPECS["test"][eval_cfg.partition]

    seed = 42
    NUM_EVAL_TRAJ = eval_cfg.num_eval_traj
    num_eval_traj = NUM_EVAL_TRAJ
    num_successes = 0
    exp_dir = os.path.dirname(os.path.dirname(eval_cfg.ckpt))
    video_dir = os.path.join(exp_dir, "videos")
    if eval_cfg.save_video:
        os.makedirs(video_dir, exist_ok=True)

    policy = create_policy_from_ckpt(eval_cfg.ckpt, eval_cfg.device).to(eval_cfg.device)
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                eval_cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][eval_cfg.partition][eval_cfg.task] if eval_cfg.dist == "test" else PARTITION_TO_SPECS[eval_cfg.dist][eval_cfg.task],
                seed=seed,
                # render_prompt=False,
                # display_debug_window=False,
                # hide_arm_rgb=True,
            )
        ),
        bonus_steps=2,
    )
    task = env.task
    oracle_fn = task.oracle(env)

    cfg = None
    folder_path = eval_cfg.ckpt.split("/ckpt")[0]
    cfg_path = folder_path + "/conf.yaml"
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    cfg = OmegaConf.create(cfg)
    policy_cfg = cfg.module.policy
    data_cfg = cfg.data_module
    trainer_cfg = cfg.trainer

    _views = sorted(["top", "front"])
    prompt_template = open(data_cfg.prompt_file, "r").read()
    print(colored(f"Prompt:\n{prompt_template}", "green"))
    processor = transformers.AutoProcessor.from_pretrained(policy_cfg.model_name, padding_side="left")
    tokenizer = processor.tokenizer
    while num_eval_traj:
        print("num_eval_traj: ", num_eval_traj)
        env.global_seed = seed

        obs = env.reset()
        rgb_plot = copy.deepcopy(obs['rgb']['top'])
        # env.render()

        meta_info = env.meta_info
        env_prompt = env.prompt
        prompt_assets = env.prompt_assets
        elapsed_steps = 0
        inference_cache = {}
        action_to_plot = {'pose0_position': [], 'pose1_position': [], 'rgb': []}
        task_name = eval_cfg.task

        while True:
            print("elapsed_steps: ", elapsed_steps)
            oracle_action = oracle_fn.act(obs)
            if oracle_action is None:
                print("WARNING: no oracle action, skip!")
                oracle_failed = True
                break
            oracle_action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in oracle_action.items()
            }
            oracle_action_prepare = stack_sequence_fields([oracle_action])
            print(obs.keys())
            obs_list = [obs]
            # imported from vima_bench
            obs_prepare = stack_sequence_fields(obs_list)
            rgb_dict = obs_prepare.pop("rgb")
            rgb_dict = {view: np.stack(rgb_dict[view], axis=0) for view in _views}

            print(env_prompt)
            prompt = prompt_template.format(env_prompt)
            print(colored(f"{prompt}", "blue"))
            # sample_list is a tuple of obs, action, prompt_dict
            sample_list = prepare_sample_vlm_rgb_only(
                obs=obs_prepare,
                rgb_dict=rgb_dict,
                action=oracle_action_prepare,
                action_position_bounds=meta_info["action_bounds"],
                prompt=prompt,
                prompt_assets=prompt_assets,
                meta=meta_info,
                tokenizer=tokenizer,
                processor=processor,
                add_special_tokens=data_cfg.tokenizer_add_special_tokens,
            )
            # append the task_name to the tuple
            sample_list += (task_name,)
            # get all the values for forwarding it through the model
            obs_tokens_to_forward, action_f, action_mask_f, prompt_tokens_to_forward, task_name_to_batch_indices_f= \
                    collate_fn_vlm_rgb_only(samples_list=[sample_list], tokenizer=tokenizer)
            if action_f is not None:
                action_f_to_forward = {k: v.to(eval_cfg.device) for k, v in action_f.items()}
                action_f_to_forward = any_to_datadict(action_f_to_forward)
            if action_mask_f is not None:
                action_mask_f_to_forward = {k: v.to(eval_cfg.device) for k, v in action_mask_f.items()}
                action_mask_f_to_forward = any_to_datadict(action_mask_f_to_forward)
            if obs_tokens_to_forward is not None:
                obs_tokens_to_forward = {k: v.to(eval_cfg.device) for k, v in obs_tokens_to_forward.items()}

            prompt_tokens_to_forward = {k: v.to(eval_cfg.device) for k, v in prompt_tokens_to_forward.items()}

            with torch.autocast(device_type="cuda", dtype=policy.dtype):
                loss, metrics, bs = policy.training_step((obs_tokens_to_forward, action_f_to_forward, action_mask_f_to_forward, prompt_tokens_to_forward, task_name_to_batch_indices_f), 0)
                print(metrics['loss_pose0_position'])
                print("loss: ", loss)
                print("metrics: ", metrics)
                print("bs: ", bs)
            import ipdb; ipdb.set_trace()

            # TODO: discretize the action if present
            predicted_action_tokens = policy.forward(
                    action_token=None,
                    prompt_dict=prompt_tokens_to_forward,
            )  # (1, B, E)
            predicted_action_tokens = predicted_action_tokens.to(policy.dtype)

            dist_dict = policy.forward_action_decoder(predicted_action_tokens)
            actions = {k: v.mode() for k, v in dist_dict.items()}

            action_tokens = policy.forward_action_token(actions)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            actions = policy._de_discretize_actions(actions)

            action_to_plot['pose0_position'].append(copy.deepcopy(actions['pose0_position']).detach().cpu().numpy())
            action_to_plot['pose1_position'].append(copy.deepcopy(actions['pose1_position']).detach().cpu().numpy())
            action_to_plot['rgb'].append(copy.deepcopy(rgb_plot))

            action_bounds = [meta_info["action_bounds"]]
            action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
            action_bounds_high = [
                action_bound["high"] for action_bound in action_bounds
            ]
            action_bounds_low = np.asarray(action_bounds_low)
            action_bounds_high = np.asarray(action_bounds_high)
            action_bounds_low = torch.tensor(
                action_bounds_low, dtype=torch.float32, device=eval_cfg.device
            )
            action_bounds_high = torch.tensor(
                action_bounds_high, dtype=torch.float32, device=eval_cfg.device
            )
            actions["pose0_position"] = (
                actions["pose0_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose1_position"] = (
                actions["pose1_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose0_position"] = torch.clamp(
                actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose1_position"] = torch.clamp(
                actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
            )

            actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
            actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
            actions["pose0_rotation"] = torch.clamp(
                actions["pose0_rotation"], min=-1, max=1
            )
            actions["pose1_rotation"] = torch.clamp(
                actions["pose1_rotation"], min=-1, max=1
            )
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            actions = any_slice(actions, np.s_[0, 0])
            # print(actions)
            obs, _, done, info = env.step(actions)
            rgb_plot = copy.deepcopy(obs['rgb']['top'])
            elapsed_steps += 1
            if done:
                if eval_cfg.save_video:
                    plot_actions(action_to_plot, os.path.join(video_dir, f"video_{num_eval_traj:03d}.png"))
                print(env.env.env.task.check_success().success)
                num_successes += env.env.env.task.check_success().success
                num_eval_traj -= 1
                break
    print("************************************************************************")
    print("Success: ")
    print("{}".format((100.00*num_successes)/NUM_EVAL_TRAJ))
    print("************************************************************************")

class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--dist", type=str, default="test", choices=["test", "train_l", "train"])
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cpu")
    arg.add_argument("--num_eval_traj", default=25, type=int)
    arg.add_argument('--save_video', action="store_true", default=False)

    arg = arg.parse_args()
    main(arg)
