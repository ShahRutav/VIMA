import hydra
from vima.trainer import *
import enlight.utils as U

def check_cfg(cfg):
    print(cfg.keys())
    policy_cls = cfg.module.policy.cls
    if policy_cls == "vima.policy.xattn.vlm_as_policy.VLMPolicy":
        assert cfg.trainer.precision == "bf16-mixed", "LLAVA only supports mixed precision"
    elif policy_cls == "vima.policy.xattn.xattn_obj_as_token.XAttnGPTObjAsTokenPolicy":
        assert cfg.trainer.precision == "32" or cfg.trainer.precision == "32-true", "VIMA only supports 32-bit precision"
    else:
        raise NotImplementedError(f"Checking for policy Policy {policy_cls} is not supported")

@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    print(cfg)
    check_cfg(cfg)
    trainer_ = VIMATrainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(U.omegaconf_to_dict(cfg))
    trainer_.fit()
    # trainer_.data_module.setup()
    # trainer_.trainer.validate(
    #     model=trainer_.module,
    #     datamodule=trainer_.data_module,
    #     ckpt_path="best" if cfg.eval_best_model else None,
    # )
    # trainer_.trainer.test(
    #     model=trainer_.module,
    #     datamodule=trainer_.data_module,
    #     ckpt_path="best" if cfg.eval_best_model else None,
    # )


if __name__ == "__main__":
    main()
