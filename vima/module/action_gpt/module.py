from __future__ import annotations
from typing import Literal

import torch

from enlight.learn import CosineScheduleFunction
from enlight.utils import register_class
import enlight.utils as U
from enlight.learn import transformer_freeze_except_last_layers

from ..base import ImitationBaseModule
from ...policy.base import BasePolicy
from ...policy.xattn.xattn_obj_as_token import XAttnGPTObjAsTokenPolicy


@register_class
class ActionGPTModule(ImitationBaseModule):
    def __init__(
        self,
        policy: BasePolicy,
        *,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = True,
        lr_warmup_epochs: int,
        lr_cosine_epochs: int,
        lr_cosine_min: float,
        weight_decay: float = 0.0,
        lr_layer_decay: float = 1.0,
        bs_wrt_lr: int = 256,
        # ====== online eval ======
        eval_partition: Literal[
            "within_distribution", "novel_combo", "unseen_object", "unseen_task"
        ],
        eval_vec_env_size: int | None = None,
        n_eval_per_task: int = 200,
        deterministic_eval: bool = True,
        eval_bonus_steps: int,
        enable_visualization: bool,
        bbox_from_detection_model: bool = False,
        detection_model_ckpt_path: str | None = None,
        detection_model_threshold: float | None = None,
        # ====== unseen task finetuning ======
        finetune_unseen_task: bool = False,
        unfreeze_dt_last_n_layers: int | None = None,
        # ====== finetune from ckpt ======
        ckpt_path: str | None = None,
        # ====== randomness ======
        seed: int | None = None,
        # ====== profile flops ======
        profile_flops: bool = False,
        profile_epochs: int = 1,
    ):
        self._finetune_unseen_task = finetune_unseen_task
        if finetune_unseen_task:
            assert ckpt_path is not None
            assert unfreeze_dt_last_n_layers is not None
        super().__init__()
        self.policy = policy

        # ====== learning ======
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_cosine_epochs = lr_cosine_epochs
        self.steps_per_epoch = None
        self.lr_cosine_min = lr_cosine_min
        self.weight_decay = weight_decay
        self.lr_layer_decay = lr_layer_decay
        self.batch_size = None  # used to scale LR
        self.bs_wrt_lr = bs_wrt_lr
        # ====== online eval ======
        assert eval_partition in [
            "within_distribution",
            "novel_combo",
            "unseen_object",
            "unseen_task",
        ]
        self.eval_partition = eval_partition
        self.eval_vec_env_size = eval_vec_env_size
        self.n_eval_per_task = n_eval_per_task
        self.deterministic_eval = deterministic_eval
        self.eval_bonus_steps = eval_bonus_steps
        self.enable_visualization = enable_visualization
        self.vis_save_path = None  # set by trainer
        # ====== randomness ======
        self.seed = seed

        if bbox_from_detection_model:
            assert detection_model_ckpt_path is not None
            assert detection_model_threshold is not None
            from vima_detectron import VIMADetect

            detection_model = VIMADetect(
                ckpt_path=detection_model_ckpt_path,
                score_thresh_test=detection_model_threshold,
            )
            self.policy.detection_model = detection_model

        if ckpt_path is not None:
            print(f"Loading from ckpt path: {ckpt_path}")
            self.load_state_dict(
                torch.load(
                    ckpt_path,
                    map_location="cpu" if not torch.cuda.is_available() else None,
                )["state_dict"]
            )

        if finetune_unseen_task:
            U.freeze_params(self)
            U.unfreeze_params(self.policy.xattn_gpt)
            unfreeze_dt_last_n_layers = min(
                unfreeze_dt_last_n_layers,
                len(self.policy.xattn_gpt.h),
                len(self.policy.xattn_gpt.xattns),
            )
            transformer_freeze_except_last_layers(
                model=self.policy.xattn_gpt,
                layer_0_params=[
                    "positions_embed.*",
                    "xattn_positions_embed.*",
                ],
                block_sequence_name="h",
                num_last_layers=unfreeze_dt_last_n_layers,
            )
            transformer_freeze_except_last_layers(
                model=self.policy.xattn_gpt,
                layer_0_params=[
                    "positions_embed.*",
                    "xattn_positions_embed.*",
                ],
                block_sequence_name="xattns",
                num_last_layers=unfreeze_dt_last_n_layers,
            )

        if profile_flops:
            assert profile_epochs > 0
            from deepspeed.profiling.flops_profiler import FlopsProfiler

            self._profiler = FlopsProfiler(self.policy)
            self._profiler_epochs = profile_epochs
            self._elapsed_steps = 0
            self._profiler_start = False
        self._profile_flops = profile_flops
        # set outside by trainer
        self.profiler_save_path = None

    def training_step(self, *args, **kwargs):
        if self._profile_flops:
            if not self._profiler_start:
                self._profiler.start_profile()
                self._profiler_start = True
            self._elapsed_steps += 1
        rtn = super().training_step(*args, **kwargs)
        if (
            self._profile_flops
            and self._elapsed_steps > self._profiler_epochs * self.steps_per_epoch
        ):
            flops = self._profiler.get_total_flops()
            params = self._profiler.get_total_params()
            self._profiler.print_model_profile(
                profile_step=self._elapsed_steps,
                output_file=self.profiler_save_path,
            )
            self._profiler.end_profile()
            self.loggers[-1].log_hyperparams(
                {"profiled_flops": flops, "profiled_params": params}
            )
            # one time profiling
            exit()
        return rtn

    def configure_optimizers(self):
        lr, lr_cosine_min = self.lr, self.lr_cosine_min
        assert self.batch_size is not None
        lr = lr * self.batch_size / self.bs_wrt_lr
        lr_cosine_min = lr_cosine_min * self.batch_size / self.bs_wrt_lr

        optimizer_groups = self.policy.get_optimizer_groups(
            weight_decay=self.weight_decay,
            lr_layer_decay=self.lr_layer_decay,
            lr_scale=1.0,
        )
        optimizer = torch.optim.AdamW(optimizer_groups, lr=lr)
        if self.use_cosine_lr:
            # calculate cosine scheduler based on gradient steps
            # so we don't need to know number of batches apriori
            assert self.steps_per_epoch is not None
            scheduler_kwargs = dict(
                base_value=1.0,  # anneal from the original LR value
                final_value=lr_cosine_min / lr,
                epochs=self.lr_cosine_epochs,
                warmup_start_value=lr_cosine_min / lr,
                warmup_epochs=self.lr_warmup_epochs,
                steps_per_epoch=self.steps_per_epoch,
            )
            U.rank_zero_info(U.color_text(f"Scaled with batch size learning rate: {lr}"), "green")
            U.rank_zero_info(U.color_text(f"Scaled with batch size minimum learning rate: {lr_cosine_min}"), "green")
            U.rank_zero_info(U.color_text(f"Effective batch size: {self.batch_size}"), "green")
            U.rank_zero_info(U.color_text(f"Effect steps per epoch: {self.steps_per_epoch}"), "green")

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )
        else:
            return optimizer

    def imitation_training_step(
        self, batch, batch_idx
    ) -> (torch.Tensor, dict[str, torch.Tensor]):
        return self.policy.training_step(batch, batch_idx)

    def imitation_validation_step(
        self, batch, batch_idx
    ) -> (torch.Tensor, dict[str, torch.Tensor]):
        return self.policy.validation_step(batch, batch_idx)

    def imitation_evaluation_step(self):
        self.evaluator.start(self.policy)
        results = self.evaluator.get_results()
        results = {f"eval/{k}": v for k, v in results.items()}
        self.log_dict(
            results, prog_bar=False, on_step=False, on_epoch=True, batch_size=1
        )
        return results

    def create_evaluator(self):
        pass
