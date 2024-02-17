from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import peft
from enlight.learn import transformer_lr_decay_optimizer_groups, default_optimizer_groups

from transformers.modeling_outputs import ModelOutput
from transformers import LlavaForConditionalGeneration

@dataclass
class LlavaPromptEncoderCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None

class LlavaPromptEncoder(LlavaForConditionalGeneration):
    def __init__(self, config, last_n_feats=1):
        self._last_n_feats = last_n_feats
        self._setup_is_once = False
        self._hidden_size = 4096
        super().__init__(config)

    @property
    def vlm_hidden_size(self):
        return self._hidden_size

    def forward(self, *args, **kwargs):
        last_hidden_state = None
        output_last_hidden_state = kwargs.pop("output_last_hidden_state", False)
        kwargs["output_hidden_states"] = True if output_last_hidden_state else kwargs["output_hidden_states"]
        output = super().forward(*args, **kwargs)
        last_hidden_state = None
        if output_last_hidden_state:
            last_hidden_state = output.hidden_states[-1][:, -self._last_n_feats:, :]

        return LlavaPromptEncoderCausalLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            image_hidden_states=output.image_hidden_states,
            last_hidden_state=last_hidden_state
        )

    def get_lora_layers_to_transform(self, last_n_layers):
        if last_n_layers < 0:
            return None
        lora_layer_names = [i for i in range(32-last_n_layers, 32)]
        return lora_layer_names

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale):
        llava_pg, llava_pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "*lora_*"
            ],
        )
        return llava_pg, llava_pids

