"""
Base class for Activation analysis models.

Provides common functionality for computing activation and save it.
"""

from abc import ABC
from typing import Any, Dict, Tuple, Union
import os
import re

from transformers import AutoConfig
from llava.model import LlavaLlamaForCausalLM, LlavaConfig

import torch
import torch.nn as nn

def wildcard_to_regex(pattern_str):
    """Convert a wildcard pattern (with '*') to a compiled regex pattern."""
    escaped_str = re.escape(pattern_str)
    regex_pattern = escaped_str.replace(r'\*', '.*')

    return re.compile(regex_pattern)

class BaseActivationModel(ABC):
    def __init__(self, config):
        target = config.target if hasattr(config, "target") else None
        self.target_patterns = []
        if target:
            self.target_patterns = [wildcard_to_regex(sub_target) for sub_target in target.split(",") if sub_target]
        self.hooks = self._register_hooks()

        # Store per-layer activations
        self.activations: dict[str, torch.Tensor] = {}

        # In evaluation mode, as we only need forward passes
        self.model.eval()

    def _check(self, name: str) -> bool:
        """Check if a given name matches any of the target patterns."""
        if not self.target_patterns:
            return True
        for pattern in self.target_patterns:
            if pattern.match(name):
                return True
        return False

    def _register_hooks(self) -> list[torch.utils.hooks.RemovableHandle]:
        module_visited = set()
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(module: str):
            def hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    return
                # Mean over sequence length -> [batch, hidden_dim]
                token_mean = torch.norm(output.detach(), p=2, dim=1, dtype=torch.float32)
                token_mean = token_mean / torch.norm(token_mean, p=2, dim=-1, keepdim=True)
                self.activations[module] = token_mean

            return hook

        for module in self.state_dict():
            module = module.replace('.weight', '').replace('.bias', '')
            if self._check(module) and module not in module_visited:
                try:
                    hook = make_hook(module)
                    _module = self.get_submodule(module) 
                    hooks.append(_module.register_forward_hook(hook))
                    module_visited.add(module)
                except Exception as e:
                    print(f"Warning: could not register hook for module {module}: {e}")
                
        return hooks
    
    def _remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def compute_activation(
        self, input_ids: torch.Tensor, image_tensor: torch.Tensor, image_sizes, args
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _ = self.forward(
                input_ids=input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                use_cache=True)

        return self.activations

    def clean(self) -> None:
        """Clean up stored attribution data to free memory."""
        self.activations = {}
        if hasattr(self, "_context"):
            del self._context

class ActivationLlavaConfig(LlavaConfig):
    model_type = "activation_llava"

    def __init__(self, **kwargs):
        self.target = kwargs.pop("target", None)
        super().__init__(**kwargs)
    
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict, kwargs = LlavaConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        target = kwargs.pop("target", None)
        if target is not None:
            config_dict["target"] = target
        return config_dict, kwargs

class ActivationLlavaLlamaWrapper(BaseActivationModel, LlavaLlamaForCausalLM):  # type: ignore[misc,name-defined]
    __name__ = "ActivationLlavaLlamaWrapper"
    __qualname__ = "ActivationLlavaLlamaWrapper"
    __module__ = LlavaLlamaForCausalLM.__module__
    config_class = ActivationLlavaConfig
    def __init__(self, config: ActivationLlavaConfig):
        LlavaLlamaForCausalLM.__init__(self, config)  # type: ignore[attr-defined]
        BaseActivationModel.__init__(self, config)

AutoConfig.register("activation_llava", ActivationLlavaConfig)