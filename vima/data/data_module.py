from __future__ import annotations
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from vima import utils as U

from .dataset import VIMADataset
from .dataset_vlm import VLMDataset
from .process.dm_only import collate_fn_bbox, collate_fn_rgb_only, collate_fn_vlm_rgb_only
import json

class VIMADataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        tokenizer_add_special_tokens: bool,
        batch_size: int,
        val_batch_size: int,
        dataloader_num_workers: int,
        train_portion: float = 0.8,
        tokenizer: str = "distilbert-base-uncased",
        t5_prompt_prefix: str | None = None,
        use_bbox_repr: bool = True,
        cropped_img_size: int = 224,
        num_trajs: int | None = None,
        seed: int | None = None,
        task_selection: str | list[str] | None = None,
        add_obj_aug: bool = False,
        obj_aug_prob_map: dict[int, float] | None = None,
        train_on_unseen_task_for_finetune: bool = False,
        bbox_from_detection_model: bool = False,
    ):
        super().__init__()
        self.path = path
        self._train_on_unseen_task_for_finetune = train_on_unseen_task_for_finetune
        self.tokenizer_add_special_tokens = tokenizer_add_special_tokens
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = dataloader_num_workers
        self._train_portion = train_portion
        self._tokenizer = tokenizer
        self.use_bbox_repr = use_bbox_repr
        self.cropped_img_size = cropped_img_size
        self._add_obj_aug = add_obj_aug
        self._obj_aug_prob_map = obj_aug_prob_map

        self.num_trajs = num_trajs

        self.seed = seed
        if isinstance(task_selection, str):
            task_selection = [task_selection]
        self.task_selection = task_selection
        self._t5_prompt_prefix = t5_prompt_prefix

        self._dataset, self._train_set, self._val_set = None, None, None
        self._max_prompt_token_len = None

        self._bbox_from_detection_model = bbox_from_detection_model
        self.detection_model = None  # set outside

    def setup(self, stage: str | None = None, setup_data: bool = True) -> None:
        if stage == "fit" or stage is None:
            self._dataset = VIMADataset(
                path=self.path,
                tokenizer_add_special_tokens=self.tokenizer_add_special_tokens,
                tokenizer=self._tokenizer,
                t5_prompt_prefix=self._t5_prompt_prefix,
                use_bbox_repr=self.use_bbox_repr,
                cropped_img_size=self.cropped_img_size,
                num_trajs=self.num_trajs,
                seed=self.seed,
                task_selection=self.task_selection,
                add_obj_aug=self._add_obj_aug,
                obj_aug_prob_map=self._obj_aug_prob_map,
                train_on_unseen_task_for_finetune=self._train_on_unseen_task_for_finetune,
                setup_data=setup_data,
                bbox_from_detection_model=self._bbox_from_detection_model,
                detection_model=self.detection_model,
            )
            if setup_data:
                self._train_set, self._val_set = U.sequential_split_dataset(
                    self._dataset,
                    split_portions=[self._train_portion, 1 - self._train_portion],
                )

    def train_dataloader(self):
        # shuffle=True will be added by distributed sampler,
        # set `replace_sampler_ddp=True`  for Lightning trainer
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            collate_fn=collate_fn_bbox if self.use_bbox_repr else collate_fn_rgb_only,
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
            persistent_workers=True if self._num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._val_batch_size,
            collate_fn=collate_fn_bbox if self.use_bbox_repr else collate_fn_rgb_only,
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
            persistent_workers=True if self._num_workers > 0 else False,
        )

    def test_dataloader(self):
        """
        For test_step(), simply returns None N times.
        test_step() can have arbitrary logic
        """
        return DummyDataset(batch_size=1).get_dataloader()

    @property
    def max_obs_len(self):
        return self._dataset.max_obs_len

    @property
    def max_action_len(self):
        return self._dataset.max_action_len

    @property
    def max_prompt_token_len(self):
        if self._max_prompt_token_len is None:
            max_prompt_token_len = 0
            for idx in range(len(self._dataset)):
                _, _, prompt, _ = self._dataset[idx]
                max_prompt_token_len = max(max_prompt_token_len, len(prompt))
            self._max_prompt_token_len = max_prompt_token_len
        return self._max_prompt_token_len

    @property
    def max_traj_seed(self):
        return self._dataset.max_traj_seed

    @property
    def tokenizer(self):
        return self._dataset.tokenizer

class VLMDataModule(VIMADataModule):
    def __init__(self, *args, **kwargs):
        # if tokenizer not in the kwargs
        self._model_name = kwargs.pop("model_name")
        self._prompt_file = kwargs.pop("prompt_file")
        super().__init__(*args, **kwargs)

    def setup(self, stage: str | None = None, setup_data: bool = True) -> None:
        if stage == "fit" or stage is None:
            self._dataset = VLMDataset(
                path=self.path,
                tokenizer_add_special_tokens=self.tokenizer_add_special_tokens,
                model_name=self._model_name,
                use_bbox_repr=self.use_bbox_repr,
                cropped_img_size=self.cropped_img_size,
                num_trajs=self.num_trajs,
                seed=self.seed,
                task_selection=self.task_selection,
                train_on_unseen_task_for_finetune=self._train_on_unseen_task_for_finetune,
                setup_data=setup_data,
                bbox_from_detection_model=self._bbox_from_detection_model,
                detection_model=self.detection_model,
                prompt_file=self._prompt_file,
            )
            if setup_data:
                self._train_set, self._val_set = U.sequential_split_dataset(
                    self._dataset,
                    split_portions=[self._train_portion, 1 - self._train_portion],
                )

    def train_dataloader(self):
        # shuffle=True will be added by distributed sampler,
        # set `replace_sampler_ddp=True`  for Lightning trainer
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            collate_fn=partial(collate_fn_vlm_rgb_only, tokenizer=self._dataset.processor.tokenizer),
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
            persistent_workers=True if self._num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._val_batch_size,
            collate_fn=partial(collate_fn_vlm_rgb_only, tokenizer=self._dataset.processor.tokenizer),
            num_workers=min(self._batch_size, self._num_workers),
            pin_memory=True,
            persistent_workers=True if self._num_workers > 0 else False,
        )

class DummyDataset(Dataset):
    """
    For test_step(), simply returns None N times.
    test_step() can have arbitrary logic
    """

    def __init__(self, batch_size, epoch_len=1):
        """
        Still set batch_size because pytorch_lightning tracks it
        """
        self.n = epoch_len
        self._batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.zeros((self._batch_size,), dtype=bool)

    def get_dataloader(self) -> DataLoader:

        """
        Our dataset directly returns batched tensors instead of single samples,
        so for DataLoader we don't need a real collate_fn and set batch_size=1
        """
        return DataLoader(
            self,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            collate_fn=_singleton_collate_fn,
        )


def _singleton_collate_fn(tensor_list):
    """
    Our dataset directly returns batched tensors instead of single samples,
    so for DataLoader we don't need a real collate_fn.
    """
    assert len(tensor_list) == 1, "INTERNAL: collate_fn only allows a single item"
    return tensor_list[0]
