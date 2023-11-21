import os
import torch
import yaml

#from .policy import Policy
from .policy.xattn.xattn_obj_as_token import XAttnGPTObjAsTokenPolicy

def create_policy_from_ckpt(ckpt_path, device):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location=device)

    folder_path = ckpt_path.split("/ckpt")[0]
    cfg_path = folder_path + "/conf.yaml"

    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)
        cfg = cfg['module']['policy']
        del cfg["cls"]

    policy = XAttnGPTObjAsTokenPolicy(**cfg)
    policy.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy.eval()
    return policy
