import os
import numpy as np
from PIL import Image

from vima.data.data_module import VIMADataModule, VLMDataModule
model_name = 'llava-hf/llava-1.5-7b-hf'
cfg = {
    "path": "/home/rutavms/data/vima/dataset_50k_lang_left_pick/",
    "tokenizer_add_special_tokens": True,
    "batch_size": 5,
    "val_batch_size": 1,
    "dataloader_num_workers": 0,
    "train_portion": 0.8,
    "tokenizer": model_name,
    "model_name": model_name,
    "use_bbox_repr": False,
    "cropped_img_size": 224,
    "num_trajs": None,
    "num_trajs_dict_path": "/home/rutavms/research/preference/VIMA/main/num_trajs_dict.json",
    "seed": 0,
    "add_obj_aug": True,
    "obj_aug_prob_map": {0: 0.95, 1: 0.05},
    "prompt_file": "vima/prompts/llava_lang_task.txt",
    "task_selection": "visual_manipulation",}
# set seed in numpy
np.random.seed(cfg["seed"])
data_module = VLMDataModule(**cfg)
data_module.setup()
train_loader = data_module.train_dataloader()
for batch in train_loader:
    print(len(batch))
    # 5: obs, action, action_mask, task_prompt, task_name
    obs = batch[0]

    for k, v in batch[-2].items():
        print(k, v.shape)

    for k, v in batch[1].items():
        print(k, v.shape)
    # # concatenate along an axis and save the image
    # img = np.concatenate([traj_obs[i] for i in range(traj_obs.shape[0])], axis=1)
    # print(img.shape)
    # img = Image.fromarray(img)
    # img.save("test_images/traj_obs.jpg")
    break
