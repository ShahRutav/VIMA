from __future__ import annotations
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import enlight.utils as U
from transformers.models.openai.modeling_openai import (
    OpenAIGPTPreTrainedModel,
    Block,
    BaseModelOutput,
)

from ....adapters import ALL_ADAPTERS


class GPTModelWithAdapter(OpenAIGPTPreTrainedModel):
    def __init__(
        self,
        config,
        *,
        adapter_type: str,
        adapter_positions: int | list[int] = -1,
        adapter_n_layers: int | list[int] = 2,
        freeze_backbone: bool,
    ):
        assert adapter_type in ALL_ADAPTERS

        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                Block(config.n_positions, config, scale=True)
                for _ in range(config.n_layer)
            ]
        )

        self.register_buffer("position_ids", torch.arange(config.n_positions))

        # create adapters
        if isinstance(adapter_positions, int):
            assert adapter_positions == -1 or adapter_positions > 0
            if adapter_positions == -1:
                adapter_positions = list(range(config.n_layer))
            else:
                assert adapter_positions <= config.n_layer
                adapter_positions = list(range(adapter_positions))
        elif isinstance(adapter_positions, list):
            assert all([0 <= pos < config.n_layer for pos in adapter_positions])
        else:
            raise ValueError(
                f"adapter_positions should be int or list[int], but got {adapter_positions}"
            )

        if isinstance(adapter_n_layers, int):
            assert adapter_n_layers > 0
            adapter_n_layers = [adapter_n_layers] * len(adapter_positions)
        elif isinstance(adapter_n_layers, list):
            assert len(adapter_n_layers) == len(adapter_positions)
            assert all([n > 0 for n in adapter_n_layers])
        else:
            raise ValueError(
                f"adapter_n_layers must be int or list[int], got {type(adapter_n_layers)}"
            )

        self.adapters = nn.ModuleDict(
            {
                f"adapter_{position_idx}": ALL_ADAPTERS[adapter_type](
                    emb_dim=config.n_embd, n_layers=n_layers
                )
                for position_idx, n_layers in zip(adapter_positions, adapter_n_layers)
            }
        )

        # Initialize weights and apply final processing
        self.post_init()

        if freeze_backbone:
            U.freeze_params(self.tokens_embed)
            U.freeze_params(self.positions_embed)
            U.freeze_params(self.drop)
            U.freeze_params(self.h)

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

            # apply adapter if necessary
            if f"adapter_{i}" in self.adapters:
                hidden_states = self.adapters[f"adapter_{i}"](hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
