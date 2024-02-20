import os
import yaml
import hydra
from omegaconf import OmegaConf
from vima.trainer import *
import enlight.utils as U
from vima import create_policy_from_ckpt

exp_dir = "experiments/xattn_obj_token_2M/lang_5k_llava_sweep_xattn_obj_token_2M_lr1.0e-4_wd1e-2_cos.epochs3-100_b2.vb2__100ep_20240219-122601"
ckpt_path = os.path.join(exp_dir, 'ckpt', 'epoch71-val_acc0.99700.pth')
cfg_path = os.path.join(exp_dir, 'conf.yaml')

with open(cfg_path, "r") as file:
    cfg = yaml.safe_load(file)
cfg = OmegaConf.create(cfg)
cfg.module.ckpt_path = ckpt_path
cfg.module.policy.rank = 0

data_module = U.instantiate(cfg.data_module)
data_module.setup(stage="fit", setup_data=True)
train_dataloader = data_module.train_dataloader()

policy = create_policy_from_ckpt(cfg.module.ckpt_path, 'cuda') #.to('cuda')
# module = U.instantiate(cfg.module)
# policy = module.policy

ind = 0.0
run_avg_loss = 0.0
run_avg_metric = None
for data in train_dataloader:
    loss, metric, real_batch_size = policy.validation_step(data, batch_idx=ind)
    run_avg_loss = (run_avg_loss*ind + loss.item())/(ind+1)
    if run_avg_metric is None:
        run_avg_metric = metric
    else:
        for k,v in run_avg_metric.items():
            run_avg_metric[k] = (run_avg_metric[k]*ind + metric[k])/(ind+1)

    print(f"Running avg loss: {run_avg_loss:.4f}")
    for k,v in run_avg_metric.items():
        print(f"Running {k}: {v:.4f}")

    ind += 1.0
    # break

