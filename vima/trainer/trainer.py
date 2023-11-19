import time
from math import ceil
import enlight.utils as U
import pytorch_lightning as pl
from enlight.learn import LightingTrainer
import pytorch_lightning.loggers as pl_loggers


from vima.data import *
from vima.module import *
from vima.policy import *


class VIMATrainer(LightingTrainer):
    def __init__(self, *args, inference: bool = False, **kwargs):
        self._inference = inference
        super().__init__(*args, **kwargs)
        self.data_module.detection_model = self.module.policy.detection_model

    def create_module(self, cfg) -> pl.LightningModule:
        module = U.instantiate(cfg.module)
        # compute steps per epoch for LR scheduler
        num_gpus = self.trainer.num_gpus
        if num_gpus == 0:
            num_gpus = 1
        effective_bs = cfg.bs * num_gpus * cfg.trainer.accumulate_grad_batches
        num_tasks = 13
        steps_per_epoch = ceil(
            num_tasks * cfg.num_trajs * cfg.data_module.train_portion / effective_bs
        )
        U.rank_zero_info(U.color_text(f"Steps per epoch: {steps_per_epoch}", "green"))
        module.steps_per_epoch = steps_per_epoch
        module.batch_size = effective_bs
        # attach dm to policy in favor of inference
        module.policy.data_module = self.data_module
        # attach vis save path
        module.vis_save_path = U.f_join(self.run_dir, "visualization")
        module.profiler_save_path = U.f_join(self.run_dir, "profile.txt")
        return module

    def create_data_module(self, cfg) -> pl.LightningDataModule:
        return U.instantiate(cfg.data_module)

    def create_loggers(self, cfg) -> list[pl.loggers.LightningLoggerBase]:
        loggers = super().create_loggers(cfg)
        if cfg.use_wandb:
            loggers.append(
                pl_loggers.WandbLogger(
                    name=cfg.wandb_run_name, project=cfg.wandb_project, id=self.run_name
                )
            )
        return loggers

    def generate_run_name(self, cfg):
        name = cfg.run_name + "_" + time.strftime("%Y%m%d-%H%M%S")
        if self._inference:
            name = "eval_" + cfg.module.eval_partition + name
        return name
