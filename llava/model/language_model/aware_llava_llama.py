#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict, Any
import json
from abc import ABC, abstractmethod
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AwareLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, freeze_pos):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.frozen_pos = sorted(freeze_pos)
        self.active_pos = [
            i for i in range(self.out_features) if i not in self.frozen_pos
        ]
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(
                f"Target neuron indices must be within [0, {self.out_features - 1}]"
            )
        if len(self.frozen_pos) != len(set(self.frozen_pos)):
            raise ValueError("Frozen neuron indices contain duplicate values")
        self.active = nn.Linear(self.in_features, len(self.active_pos), bias=False)
        self.frozen = nn.Linear(self.in_features, len(self.frozen_pos), bias=False)
        W = original_linear.weight.data
        self.active.weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.frozen.weight = nn.Parameter(W[self.frozen_pos].clone(), requires_grad=False)
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(b[self.active_pos].clone(), requires_grad=True)
            self.frozen_bias = nn.Parameter(b[self.frozen_pos].clone(), requires_grad=False)
        else:
            self.register_parameter("active_bias", None)
            self.register_parameter("frozen_bias", None)
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.frozen_pos] = torch.arange(len(self.frozen_pos)) + len(self.active_pos)
        self.register_buffer("index_map", index_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_out = self.active(x)
        frozen_out = self.frozen(x)
        output = torch.cat([active_out, frozen_out], dim=-1)
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.frozen_bias], dim=0)
            output += bias.expand_as(output)
        return output.gather(
            dim=-1,
            index=self.index_map.expand_as(output),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"active_neurons={len(self.active_pos)} "
            f"[{100 * len(self.active_pos) / self.out_features:.1f}%]"
        )


class BaseAwareModel(ABC):
    def _apply_aware(self, config):
        if not hasattr(config, "target_pos") or config.target_pos is None:
            raise ValueError(
                f"Config must include `target_pos` attribute, but got: {config}"
            )
        self.target_pos = config.target_pos

        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Replace target layers up front so weight loading matches AwareLinear
        self.apply_aware_linear()

        # Unfreeze the Aware active heads only
        for module in self.modules():
            if isinstance(module, AwareLinear):
                module.active.weight.requires_grad = True
                if module.active_bias is not None:
                    module.active_bias.requires_grad = True

    def apply_aware_linear(self) -> None:
        if not hasattr(self, "config"):
            raise ValueError("Model must have 'config' before applying AwareLinear")

        for module in self.target_pos:
            freeze_pos = self.target_pos[module]
            if len(freeze_pos) == 0:
                logger.warning(f"No neurons specified for {module}, skipping.")
                continue

            _original_linear = self.get_submodule(module) 
            if len(freeze_pos) >= _original_linear.out_features:
                _original_linear.weight.requires_grad = True
                if _original_linear.bias is not None:
                    _original_linear.bias.requires_grad = True
                continue
            parent, child = module.rsplit('.', 1)
            _module_parent = self.get_submodule(parent)
            if not isinstance(_original_linear, nn.Linear):
                raise TypeError(f"Expected nn.Linear, got {type(_original_linear)}")
            aware_linear = AwareLinear(original_linear=_original_linear, freeze_pos=freeze_pos)
            if child.isdigit():
                child = int(child)
                _module_parent[child] = aware_linear
            else:
                setattr(_module_parent, child, aware_linear)
            logger.info(
                f"Replaced {module} "
                f"({len(freeze_pos)}/{_original_linear.out_features} neurons freezed)"
            )


class AwareLlavaConfig(LlamaConfig):
    model_type = "aware_llava_llama"
    def __init__(self, **kwargs):
        target_pos = None
        pos_path = kwargs.pop("pos_path", None)
        if pos_path:
            with open(pos_path, "r") as f:
                target_pos = json.load(f)
            for key in target_pos:
                target_pos[key] = eval(target_pos[key])
        super().__init__(**kwargs)
        self.target_pos = target_pos if target_pos is not None else getattr(self, "target_pos", None)
    
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict, kwargs = LlamaConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        pos_path = kwargs.pop("pos_path", None)
        if pos_path is not None:
            config_dict["pos_path"] = pos_path
        return config_dict, kwargs


class AwareLlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = AwareLlavaConfig

    def __init__(self, config: AwareLlavaConfig):
        super(AwareLlavaLlamaModel, self).__init__(config)


class AwareLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM, BaseAwareModel):
    config_class = AwareLlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = AwareLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("aware_llava_llama", AwareLlavaConfig)
AutoModelForCausalLM.register(AwareLlavaConfig, AwareLlavaLlamaForCausalLM)