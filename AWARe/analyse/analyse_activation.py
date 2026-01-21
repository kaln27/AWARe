import argparse
import os
import random
import time
from PIL import Image
from collections import defaultdict
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


from activation_model import ActivationLlavaLlamaWrapper
from utils import load_pretrained_model


class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, image_processor, model_config, conv_mode='llava_v1'):
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def setup_model_parallel(model, num_gpus: int):
    """
    Setup model parallelism by distributing layers across GPUs.

    Args:
        model: The model to parallelize
        num_gpus: Number of GPUs to use

    Returns:
        Modified model with layers on different devices
    """
    if num_gpus <= 1:
        return model

    num_layers = model.config.num_hidden_layers
    layers_per_gpu = num_layers // num_gpus

    print(f"Setting up model parallelism across {num_gpus} GPUs")
    print(f"Layers per GPU: ~{layers_per_gpu}")

    # Move embedding to first GPU
    model.model.embed_tokens.to("cuda:0")

    # Distribute transformer layers
    for layer_idx in range(num_layers):
        gpu_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
        model.model.layers[layer_idx].to(f"cuda:{gpu_idx}")
        if layer_idx % 10 == 0:
            print(f"Layer {layer_idx} -> GPU {gpu_idx}")

    # Move final layers to last GPU
    model.model.norm.to(f"cuda:{num_gpus - 1}")
    model.lm_head.to(f"cuda:{num_gpus - 1}")

    print("Model parallelism setup complete")
    return model


def process_batch(
    model,
    batch,
    device,
    activations: dict[str, list[torch.Tensor | None]],
    args,
    is_model_parallel: bool = False,
):
    input_ids, image_tensors, image_sizes = batch

    if not is_model_parallel:
        input_ids = input_ids.to(device)
    else:
        # For model parallelism, input goes to first device
        input_ids = input_ids.to("cuda:0")

    model.compute_activation(
        input_ids, image_tensors, image_sizes, args=args
    )

    for module in model.activations:
        act = model.activations[module]
        activations[module].append(act)
    
    model.clean()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AWARe activation analysis with batch and multi-GPU support"
    )

    # Model selection
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )

    # Batch and parallelism options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--parallel-mode",
        type=str,
        default=None,
        choices=[None, "data", "model"],
        help="Parallelism mode: 'None' (single GPU), 'data' (samples across GPUs), 'model' (layers across GPUs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        help="Use Flash Attention 2 for faster inference",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print progress every N batches",
    )
    parser.add_argument(
        "--question-file", 
        type=str, 
        default="results/data/aware_ds.jsonl")
    parser.add_argument(
        "--target", 
        type=str, 
        default=None,
        help="Target to compute activations split by comma eg. ('*.q_proj,*.k_proj,*.v_proj')")

    return parser.parse_args()


def main():
    """Main analysis workflow."""
    disable_torch_init()
    args = parse_args()

    output_path = os.path.join(args.output_dir, 'analysis.pt')
    
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping analysis.")
        exit(0)

    # Check for data parallelism requirements
    if args.parallel_mode == "data" and not ACCELERATE_AVAILABLE:
        raise ImportError(
            "Data parallelism requires accelerate library. Install with: pip install accelerate"
        )

    # Initialize accelerator for data parallelism
    accelerator = None
    if args.parallel_mode == "data":
        accelerator = Accelerator()
        device = accelerator.device
        is_main_process = accelerator.is_main_process
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory (only on main process)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get Activation model class from registry (auto-register if needed)
    activation_model_class: type[PreTrainedModel] = ActivationLlavaLlamaWrapper

    # Print configuration (only on main process)
    if is_main_process:
        print("=" * 80)
        print("AWARe Activation Analysis")
        print("=" * 80)
        print(f"Model type: {activation_model_class.__name__}")
        print(f"Model path: {args.model_path}")
        print(f"Output Dir: {args.output_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Parallel mode: {args.parallel_mode}")
        print(f"Device: {device}")
        print("=" * 80)

    if is_main_process:
        print("\n***** Clearing CUDA cache *****")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load model with Captum-based Activation
    if is_main_process:
        print(f"\n***** Loading {activation_model_class.__name__} model: {args.model_path} *****")

    model_kwargs = {
        "torch_dtype": torch.float16,
    }

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            print("Using Flash Attention 2")

    # Handle device mapping based on parallelism mode
    if args.parallel_mode == "model":
        # Model parallelism: manual device placement
        model_kwargs["device_map"] = None
        num_gpus = torch.cuda.device_count()
        if is_main_process:
            print(f"Using model parallelism across {num_gpus} GPUs")
    elif args.parallel_mode == "data":
        # Data parallelism: accelerate handles it
        model_kwargs["device_map"] = None
    else:
        # Single GPU
        model_kwargs["device_map"] = "auto"

    # Initialize model, tokenizer, image processor, context length  

    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, cls=activation_model_class, target=args.target, **model_kwargs)

    # Setup model parallelism if requested
    if args.parallel_mode == "model":
        num_gpus = torch.cuda.device_count()
        model = setup_model_parallel(model, num_gpus)
    elif args.parallel_mode == "data":
        # Accelerator will handle device placement
        model = accelerator.prepare(model)

    # Prepare metadata
    analysis_metadata = {
        "model_name": args.model_path,
        "model_type": activation_model_class.__name__,
        "num_layers": model.config.num_hidden_layers
        if hasattr(model, "config")
        else model.module.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size
        if hasattr(model, "config")
        else model.module.config.hidden_size,
        "seed": args.seed,
        "use_flash_attention": args.use_flash_attention,
        "batch_size": args.batch_size,
        "parallel_mode": args.parallel_mode,
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(analysis_metadata, f, indent=4, ensure_ascii=False)

    # Initialize qkv activations storage
    activations: dict[str, torch.Tensor | None] = defaultdict(list)
    analysis: dict[str, torch.Tensor | None] = defaultdict(list)

    start_time = time.time()
    total_samples = 0

    # Collect all samples from AWARe subsets
    if is_main_process:
        print("\n***** Collecting AWARe samples *****")

    all_samples = []
    with open(args.question_file, 'r') as f:
        for line in f:
            all_samples.append(json.loads(line.strip()))

    if is_main_process:
        print(f"Total samples to process: {len(all_samples)}")

    # Create dataset and dataloader
    dataloader = create_data_loader(all_samples, tokenizer, image_processor, model.config)

    # Prepare dataloader with accelerator for data parallelism
    if args.parallel_mode == "data":
        dataloader = accelerator.prepare(dataloader)

    # Process batches
    if is_main_process:
        print("\n***** Starting analysis *****")

    is_model_parallel = args.parallel_mode == "model"

    for batch in tqdm(dataloader, desc="Processing batches", disable=not is_main_process):
        process_batch(
            model.module if args.parallel_mode else model,
            batch,
            device,
            activations,
            args=args,
            is_model_parallel=is_model_parallel,
        )

        if is_main_process:
            total_samples += len(batch[0])

    total_time = time.time() - start_time

    for module in activations:
        concated = torch.concat(activations[module], dim=0)
        activation = torch.mean(concated, dim=0)
        weight = model.get_submodule(module).weight.data.detach()
        weight = torch.norm(weight, p=2, dim=-1, dtype=torch.float32)
        weight = weight / torch.norm(weight, p=2)
        # analysis[module] = 0.3 * weight.cpu() + 0.7 * activation.cpu()
        # analysis[module] = 0.7 * weight.cpu() + 0.3 * activation.cpu()
        # analysis[module] = weight.cpu()
        analysis[module] = activation.cpu()

    # save to output file
    if args.parallel_mode == "data":
        # Gather activations from all processes
        analysis = accelerator.gather(analysis)
    if is_main_process:
        torch.save(
            analysis,
            output_path,
        )

    # Update metadata with completion info (only main process)
    if is_main_process:
        print("\n" + "=" * 80)
        print("✅ Analysis completed successfully!")
        print(f"Total samples processed: {total_samples}")
        print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Average time per sample: {total_time / total_samples:.2f} seconds")
        print(f"Results saved to: {output_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
