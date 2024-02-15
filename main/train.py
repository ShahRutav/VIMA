import hydra
from vima.trainer import *
import enlight.utils as U


@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    trainer_ = VIMATrainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(U.omegaconf_to_dict(cfg))
    trainer_.fit()
    trainer_.data_module.setup()
    trainer_.trainer.validate(
        model=trainer_.module,
        datamodule=trainer_.data_module,
        ckpt_path="best" if cfg.eval_best_model else None,
    )
    trainer_.trainer.test(
        model=trainer_.module,
        datamodule=trainer_.data_module,
        ckpt_path="best" if cfg.eval_best_model else None,
    )


if __name__ == "__main__":
    main()
