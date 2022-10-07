import os
import torch

from .policy import Policy


def create_policy_from_ckpt(ckpt_path, device):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location=device)
    policy = Policy(**ckpt["cfg"])
    policy.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy.eval()
    return policy
