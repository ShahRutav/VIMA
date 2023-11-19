from abc import ABC, abstractmethod


class BasePolicy(ABC):
    detection_model = None

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Given obs, return loss and other metrics for training.
        """
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        """
        Given obs, return loss and other metrics for validation.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward the NN.
        """
        pass

    @abstractmethod
    def act(
        self,
        obs,
        prompt,
        prompt_assets,
        meta_info,
        on_reset: bool,
        ready_env_ids,
        deterministic: bool,
        **kwargs
    ):
        """
        Given obs, return action.

        on_reset: indicate if the input obs is initial obs
        """
        pass

    def get_optimizer_groups(self, *args, **kwargs):
        """
        Return a list of optimizer groups.
        """
        pass
