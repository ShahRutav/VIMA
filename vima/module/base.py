import torch
from pytorch_lightning import LightningModule
import enlight.utils as U


class ImitationBaseModule(LightningModule):
    """
    Base class for imitation algorithms that use datasets for training and validation
    but requires interacting with environments for test (online eval).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # require env for online eval
        self.evaluator = None
        # on_test_start() should be called only once to create online evaluator
        self._on_test_once = U.Once()

    def training_step(self, batch, batch_idx):
        loss, log_dict, real_batch_size = self.imitation_training_step(batch, batch_idx)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=real_batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict, real_batch_size = self.imitation_validation_step(
            batch, batch_idx
        )
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
        )
        return loss

    def on_test_start(self):
        if self._on_test_once():
            self.evaluator = self.create_evaluator()

    def test_step(self, batch, batch_idx):
        """
        `batch` and `batch_idx` are dummies.
        We conduct imitation evaluation by interacting with environments.
        """
        return self.imitation_evaluation_step()

    def configure_optimizers(self):
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def imitation_training_step(
        self, batch, batch_idx
    ) -> (torch.Tensor, dict[str, torch.Tensor], int):
        """
        One imitation training step taking inputs of `batch` and `batch_idx` (essentially supervised learning)
        and return a loss tensor, a metrics dict, and the real batch size.
        """
        raise NotImplementedError

    def imitation_validation_step(
        self, batch, batch_idx
    ) -> (torch.Tensor, dict[str, torch.Tensor], int):
        """
        One imitation validation step taking inputs of `batch` and `batch_idx`
        and return a loss tensor, a metrics dict, and the real batch size.
        """
        raise NotImplementedError

    def imitation_evaluation_step(self):
        """
        Imitation evaluation involves interacting with environments.
        """
        raise NotImplementedError

    def create_evaluator(self):
        """
        Create a evaluator containing vectorized distributed envs.
        """
        raise NotImplementedError
