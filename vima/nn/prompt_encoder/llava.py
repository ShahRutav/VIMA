import torch
import torch.nn as nn
from enlight.learn import transformer_lr_decay_optimizer_groups, default_optimizer_groups

from transformers import LlavaForConditionalGeneration

class LlavaPromptEncoder(LlavaForConditionalGeneration):
    def __init__(self, config, head_type='linear', last_n_feats=1):
        self._head_type = head_type
        self._last_n_feats = last_n_feats
        self._setup_is_once = False
        self._hidden_size = 4096
        super().__init__(config)

    @property
    def vlm_hidden_size(self):
        return self._hidden_size

    def _setup_head(self, hidden_size, head_output_dim):
        if self._setup_is_once:
            raise ValueError("Setup can only be called once")
        if self._head_type == "linear":
            self.language_model.lm_head = nn.Linear(hidden_size, head_output_dim, bias=True)
        elif self._head_type == "mlp_block":
            self.language_model.lm_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, head_output_dim)
            )
        else:
            raise ValueError(f"Head type {self._head_type} not supported")
        self.language_model.lm_head.to(self.device)
        self._setup_is_once = True

    def generate(self, *args, **kwargs):
        # override the generate method to return the regression output
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        output = self.forward(*args, **kwargs)
        return output

    def get_ft_layer_names(self):
        return ["language_model.lm_head"]

    def forward(self, labels=None, *args, **kwargs):
        assert self._setup_is_once, "Setup must be called before forward"
        output = super().forward(*args, **kwargs)
        # output is of shape [b, t, e]
        output.logits = output.logits[:, :-self._last_n_feats, :]
        output.logits = torch.mean(output.logits, dim=1, keepdim=False) # TODO: maybe remove this or optional.
        # if labels is not None:
        #     pred = output.logits
        #     # calculate the loss with the labels which are of shape b,4
        #     # assert shape of labels and output is same
        #     assert labels.shape == pred.shape
        #     loss = nn.MSELoss()(pred, labels)
        #     output.loss = loss
        return output

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale):
        llava_pg, llava_pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "language_model.lm_head.*",
            ],
        )
        return llava_pg, llava_pids
