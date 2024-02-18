import os
from omegaconf import OmegaConf
from vima.policy.xattn.vlm_as_policy import VLMPolicy
import os
import numpy as np
from PIL import Image
from peft import prepare_model_for_kbit_training
from vima.data.data_module import VIMADataModule, VLMDataModule
model_name = 'llava-hf/llava-1.5-7b-hf'
cfg = {
    "path": "/home/rutavms/data/vima/dataset_50k_lang_left_pick/",
    "tokenizer_add_special_tokens": True,
    "batch_size": 8,
    "val_batch_size": 8,
    "dataloader_num_workers": 0,
    "train_portion": 0.8,
    "tokenizer": model_name,
    "model_name": model_name,
    "use_bbox_repr": False,
    "cropped_img_size": 224,
    "num_trajs": None,
    "num_trajs_dict_path": "/home/rutavms/research/preference/VIMA/main/num_trajs_dict.json",
    "seed": 0,
    "prompt_file": "vima/prompts/llava_lang_task.txt",
    "task_selection": "visual_manipulation",}
# set seed in numpy
np.random.seed(cfg["seed"])
data_module = VLMDataModule(**cfg)
data_module.setup()
train_loader = data_module.train_dataloader()

config_file = 'main/algo/llava.yaml'
cfg = OmegaConf.load(config_file)
# cfg.module.policy.vit_resolution = 256
cfg.module.policy.pop('cls')
model = VLMPolicy(**cfg.module.policy)
print(model)
groups = model.get_optimizer_groups(weight_decay=0.1, lr_layer_decay=0.1, lr_scale=0.1)
model.train()
print('*'*100)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('*'*100)


for batch in train_loader:
    loss, metrics, real_batch_size = model.training_step(batch, batch_idx=1)
    print("loss: ", loss)
    print("metrics: ", metrics)
    print("real_batch_size: ", real_batch_size)
    # break
