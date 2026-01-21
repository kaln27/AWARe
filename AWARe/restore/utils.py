import json
from pathlib import Path
from glob import glob
import os
import re

from llava.model import *
from llava.model.language_model.aware_llava_llama import AwareLinear

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def _merge_aware_weights(
    aware_layer: AwareLinear,
    original_linear: torch.nn.Linear,
) -> None:
    """
    Merge weights from AwareLinear back into standard Linear layer.

    Args:
        aware_layer: AwareLinear with split active/frozen weights
        original_linear: Target Linear layer to receive merged weights
    """
    # Allocate space for the merged weight matrix
    merged_weight = torch.zeros_like(original_linear.weight.data)

    # Insert active and frozen weights into correct positions
    merged_weight[aware_layer.active_pos] = aware_layer.active.weight.data
    merged_weight[aware_layer.frozen_pos] = aware_layer.frozen.weight.data

    # If bias exists, merge it similarly
    if original_linear.bias is not None:
        merged_bias = torch.zeros_like(original_linear.bias.data)
        merged_bias[aware_layer.active_pos] = aware_layer.active_bias.data
        merged_bias[aware_layer.frozen_pos] = aware_layer.frozen_bias.data

    # Overwrite the original layer's parameters
    original_linear.weight.data.copy_(merged_weight)
    if original_linear.bias is not None:
        original_linear.bias.data.copy_(merged_bias)

def restore_aware_model(
    target_pos_path: str | Path,
    model_path: str | Path,
    model_base: str | Path,
    output_path: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Restore AWARe model to standard Transformers format.

    Args:
        target_pos_path: JSON file with target neuron positions
        model_path: Directory with AWARe model weights (safetensors)
        model_base: Base model name or path
        output_path: Output directory for restored model
        torch_dtype: Data type for model loading

    Note:
        Handles both single and sharded safetensors files.
    """
    print(f"Restoring AWARe model from {model_path} to standard format")

    # Load target position indices from JSON file
    with open(target_pos_path, encoding="utf-8") as f:
        target_pos = json.load(f)
    for key in target_pos:
        target_pos[key] = eval(target_pos[key])

    config =  json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    # Load base transformer model
    print("Loading base model architecture...")
    original_model = LlavaLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch_dtype)
    original_model.config.mm_vision_tower = config['mm_vision_tower']

    # Initialize tensor dictionary for modified weights
    tensor_dict = {}

    # Handle single safetensors file case
    safe_tensor_path = Path(model_path) / "model.safetensors"
    if safe_tensor_path.is_file():
        print("Loading from single safetensors file")
        with safe_open(safe_tensor_path, framework="pt") as f:
            tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})
    else:
        # Load from sharded safetensors files
        shard_files = sorted(glob(f"{model_path}/model-*-of-*.safetensors"))
        print(f"Loading from {len(shard_files)} sharded safetensors files")
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt") as f:
                tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})

    # Process each transformer layer
    for module in target_pos:
        freeze_pos = target_pos[module]

        # Get original linear
        try:
            original_proj = original_model.get_submodule(module) 
        except AttributeError:
            original_proj = None
        if original_proj is None:
            print(f"Warning: {module} not found, skipping...")
            continue

        # Initialize AWARe modified layer
        aware_layer = AwareLinear(original_proj, freeze_pos=freeze_pos)

        # Prepare state dictionary for weight loading
        state_dict = {}

        # Load weight parameters
        weight_keys = {
            "active.weight": f"{module}.active.weight",
            "frozen.weight": f"{module}.frozen.weight",
        }
        for param_key, tensor_key in weight_keys.items():
            if tensor_key in tensor_dict:
                state_dict[param_key] = tensor_dict[tensor_key]

        # Load bias parameters if present in original layer
        if original_proj.bias is not None:
            bias_keys = {
                "active_bias": f"{module}.active_bias",
                "frozen_bias": f"{module}.frozen_bias",
            }
            for param_key, tensor_key in bias_keys.items():
                if tensor_key in tensor_dict:
                    state_dict[param_key] = tensor_dict[tensor_key]

        # Load parameters into AWARe layer
        aware_layer.load_state_dict(state_dict, strict=False)

        # Integrate AWARe parameters back into base model
        _merge_aware_weights(aware_layer, original_proj)

        print(f"Restored {module}")

    # Save reconstructed model and tokenizer
    print(f"Saving restored model to {output_path}")
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.save_pretrained(output_path)

    print(f"Model successfully restored and saved to {output_path}")