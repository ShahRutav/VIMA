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
        if hasattr(self.module.policy, "detection_model"):
            self.data_module.detection_model = self.module.policy.detection_model

    def create_module(self, cfg) -> pl.LightningModule:
        for attr in [
            "global_rank",
            "local_rank",
            "world_size",
            "num_nodes",
            # "num_processes",
            "node_rank",
            # "num_gpus",
            # "data_parallel_device_ids",
        ]:
            print(attr, getattr(self.trainer, attr))
        cfg.module.policy.rank = self.trainer.local_rank
        module = U.instantiate(cfg.module)
        # compute steps per epoch for LR scheduler
        num_gpus = self.trainer.num_devices
        if num_gpus == 0:
            num_gpus = 1
        effective_bs = cfg.bs * num_gpus * cfg.trainer.accumulate_grad_batches
        assert (self.data_module.task_selection is None) or isinstance(self.data_module.task_selection, list), \
                f"Data module task selection should be stored as either None or list of tasks. \
                Currently, {cfg.data_module.task_selection}"
        num_tasks = len(self.data_module.task_selection) if self.data_module.task_selection is not None else 13
        U.rank_zero_info(U.color_text(f"num_tasks {num_tasks}, num_trajs: {cfg.num_trajs}, \
                train_portion: {cfg.data_module.train_portion}, effective_bs: {effective_bs}", "green"))
        steps_per_epoch = ceil(
            num_tasks * cfg.num_trajs * cfg.data_module.train_portion / effective_bs
        )
        U.rank_zero_info(U.color_text(f"Steps per epoch: {steps_per_epoch}", "green"))
        # all the scheduler iterations are counted using steps_per_epoch.
        module.steps_per_epoch = steps_per_epoch
        # all learning rate are scaled according to effective_bs
        module.batch_size = effective_bs
        # attach dm to policy in favor of inference
        module.policy.data_module = self.data_module
        # attach vis save path
        module.vis_save_path = U.f_join(self.run_dir, "visualization")
        module.profiler_save_path = U.f_join(self.run_dir, "profile.txt")
        return module

    def create_data_module(self, cfg) -> pl.LightningDataModule:
        return U.instantiate(cfg.data_module)

    def create_loggers(self, cfg) -> list[pl.loggers.Logger]:
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
