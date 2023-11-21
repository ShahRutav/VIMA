import json
import subprocess
import sys
import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse

eval_script = "scripts/example.py"

model_path = sys.argv[1]
n_rollouts = sys.argv[2]

def run_eval(partition, task):
    result = subprocess.run(['python', eval_script, 
                             "--ckpt", model_path, 
                             "--num_eval_traj", n_rollouts,
                             "--partition", partition,
                             "--task", task,
                             ], capture_output=True, text=True)
    print(result)
        
    output_lines = result.stdout.splitlines()
    try:
        start_index = output_lines.index("Success: ")
        desired_output = output_lines[start_index+1]
        return desired_output
    except:
        return ""

for partition in PARTITION_TO_SPECS["test"]:
    tasks_for_partition = PARTITION_TO_SPECS["test"][partition]

    for task in tasks_for_partition:
        r = run_eval(partition, task)
        with open(model_path.split(".ckpt")[0] + ".txt", 'a') as file:
            file.write(partition + "\n")
            file.write(task + "\n")
            file.write(str(r) + "\n\n\n")