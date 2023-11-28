from __future__ import annotations

import os

import numpy as np
from torch.utils.data import Dataset
from vimasim import PARTITION_TO_SPECS
from tokenizers import Tokenizer
from einops import rearrange
from PIL import Image
from tqdm import tqdm

import vima.utils as U

from .constants import PLACEHOLDER_TOKENS
from .process.dm_only import prepare_sample_bbox, prepare_sample_rgb_only

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class VIMADataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer_add_special_tokens: bool,
        tokenizer: str = "distilbert-base-uncased",
        t5_prompt_prefix: str | None = None,
        use_bbox_repr: bool = True,
        cropped_img_size: int = 224,
        num_trajs: int | None = None,
        num_trajs_dict: dict | None = None,
        seed: int | None = None,
        task_selection: str | list[str] | None = None,
        add_obj_aug: bool = False,
        obj_aug_prob_map: dict[int, float] | None = None,
        train_on_unseen_task_for_finetune: bool = False,
        setup_data: bool = True,
        bbox_from_detection_model: bool = False,
        detection_model=None,
    ):
        if "t5" not in tokenizer:
            t5_prompt_prefix = None
        tokenizer = Tokenizer.from_pretrained(tokenizer)
        _n_added_tokens = tokenizer.add_tokens(PLACEHOLDER_TOKENS)
        assert _n_added_tokens == len(PLACEHOLDER_TOKENS), "INTERNAL"
        self.tokenizer = tokenizer
        self._t5_prompt_prefix = t5_prompt_prefix

        random_state = np.random.RandomState(seed)

        if setup_data:
            assert os.path.exists(path)
            if train_on_unseen_task_for_finetune:
                path = U.f_join(path, "unseen_task_fine_tune")
                assert os.path.exists(path)
                tasks = sorted(list(PARTITION_TO_SPECS["unseen_task_finetune"].keys()))
            else:
                tasks = sorted(list(PARTITION_TO_SPECS["train"].keys()))
            if task_selection is not None:
                if isinstance(task_selection, str):
                    task_selection = [task_selection]
                tasks = [task for task in tasks if task in task_selection]
            for task in tasks:
                assert os.path.exists(U.f_join(path, task)), f"{task} not found"

            max_traj_steps = None
            max_traj_seed = None
            traj_paths = {task: None for task in tasks}
            for task in tqdm(tasks, desc="Loading task dataset..."):
                task_path = U.f_join(path, task)
                if max_traj_steps is None:
                    max_traj_steps = U.load_pickle(U.f_join(task_path, "metadata.pkl"))[
                        "n_steps_max"
                    ]
                else:
                    max_traj_steps = max(
                        max_traj_steps,
                        U.load_pickle(U.f_join(task_path, "metadata.pkl"))[
                            "n_steps_max"
                        ],
                    )
                if max_traj_seed is None:
                    max_traj_seed = U.load_pickle(U.f_join(task_path, "metadata.pkl"))[
                        "seed_max"
                    ]
                else:
                    max_traj_seed = max(
                        max_traj_seed,
                        U.load_pickle(U.f_join(task_path, "metadata.pkl"))["seed_max"],
                    )
                # get all folders in task_path, note that valid trajectories are in folders with names that are integers
                folders = [f for f in os.listdir(task_path) if f.isdigit()]
                random_state.shuffle(folders)
                if num_trajs_dict is not None:
                    num_trajs = num_trajs_dict[task]
                    assert num_trajs <= len(folders), f"{num_trajs} > {len(folders)}"
                    folders = folders[:num_trajs]
                elif num_trajs is not None:
                    assert num_trajs <= len(folders), f"{num_trajs} > {len(folders)}"
                    folders = folders[:num_trajs]
                traj_paths[task] = [U.f_join(task_path, f) for f in folders]
            self.max_action_len = max_traj_steps
            self.max_traj_seed = max_traj_seed
            self.max_obs_len = max_traj_steps + 1
            self._flat_traj_paths = [p for ps in traj_paths.values() for p in ps]
            self.n_total_demos = len(self._flat_traj_paths)
            self._ptrs = random_state.permutation(self.n_total_demos)
        self._tokenizer_add_special_tokens = tokenizer_add_special_tokens

        self._use_bbox_repr = use_bbox_repr
        self._cropped_img_size = cropped_img_size
        self._views = sorted(["top", "front"])
        self._add_obj_aug = add_obj_aug
        self._obj_aug_prob_map = obj_aug_prob_map

        self._bbox_from_detection_model = bbox_from_detection_model
        self._detection_model = detection_model

    def __len__(self):
        return self.n_total_demos

    def __getitem__(self, index):
        traj_path = self._flat_traj_paths[self._ptrs[index]]
        task_name = traj_path.split("/")[-2]
        obs = U.load_pickle(U.f_join(traj_path, "obs.pkl"))
        # check same number of rgb images from different views
        assert (
            len(
                set(
                    len(os.listdir(U.f_join(traj_path, f"rgb_{view}")))
                    for view in self._views
                )
            )
            == 1
        )
        rgb_dict = {view: [] for view in self._views}
        n_rgb_frames = len(os.listdir(U.f_join(traj_path, f"rgb_{self._views[0]}")))
        for view in self._views:
            for idx in range(n_rgb_frames):
                # load {idx}.jpg using PIL
                rgb_dict[view].append(
                    rearrange(
                        np.array(
                            Image.open(
                                U.f_join(traj_path, f"rgb_{view}", f"{idx}.jpg")
                            ),
                            copy=True,
                            dtype=np.uint8,
                        ),
                        "h w c -> c h w",
                    )
                )
        rgb_dict = {view: np.stack(rgb_dict[view], axis=0) for view in self._views}
        action = U.load_pickle(U.f_join(traj_path, "action.pkl"))
        traj_meta = U.load_pickle(U.f_join(traj_path, "trajectory.pkl"))
        action_bounds = traj_meta["action_bounds"]
        prompt = traj_meta.pop("prompt")
        if self._t5_prompt_prefix is not None:
            prompt = self._t5_prompt_prefix + " " + prompt
        prompt_assets = traj_meta.pop("prompt_assets")
        if self._use_bbox_repr:
            obs, action, filled_prompt = prepare_sample_bbox(
                obs=obs,
                rgb_dict=rgb_dict,
                action=action,
                action_position_bounds=action_bounds,
                prompt=prompt,
                prompt_assets=prompt_assets,
                meta=traj_meta,
                tokenizer=self.tokenizer,
                add_special_tokens=self._tokenizer_add_special_tokens,
                cropped_img_size=self._cropped_img_size,
                add_obj_aug=self._add_obj_aug,
                obj_aug_prob_map=self._obj_aug_prob_map,
                from_detection_model=self._bbox_from_detection_model,
                detection_model=self._detection_model,
            )
        else:
            obs, action, filled_prompt = prepare_sample_rgb_only(
                obs=obs,
                rgb_dict=rgb_dict,
                action=action,
                action_position_bounds=action_bounds,
                prompt=prompt,
                prompt_assets=prompt_assets,
                meta=traj_meta,
                tokenizer=self.tokenizer,
                add_special_tokens=self._tokenizer_add_special_tokens,
            )
        return obs, action, filled_prompt, task_name
