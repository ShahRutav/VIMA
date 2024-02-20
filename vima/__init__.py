import os
import torch
import yaml
import peft
import copy

#from .policy import Policy
from .policy.xattn.xattn_obj_as_token import XAttnGPTObjAsTokenPolicy
from .policy.xattn.vlm_as_policy import VLMPolicy

def create_policy_from_ckpt(ckpt_path, device, rank=0):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location='cpu')

    folder_path = ckpt_path.split("/ckpt")[0]
    cfg_path = folder_path + "/conf.yaml"

    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)
        cfg = cfg['module']['policy']
        cls = cfg["cls"].split(".")[-1]
        del cfg["cls"]

    print(cfg)
    if "rank" in cfg.keys():
        cfg["rank"] = rank
    print(cfg["rank"])
    print(f"Creating policy from {cls} with config")
    policy = eval(cls)(**cfg)
    model_state_dict = copy.deepcopy(policy.state_dict())

    policy.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy.vlm_model = peft.prepare_model_for_kbit_training(policy.vlm_model)

    policy.eval()
    return policy
