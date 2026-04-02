import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from AWARe.analyse.utils import load_pretrained_model


PAD_TOKEN_ID = 0

def wildcard_to_regex(pattern_str: str) -> re.Pattern[str]:
    escaped_str = re.escape(pattern_str)
    regex_pattern = escaped_str.replace(r"\*", ".*")
    return re.compile(regex_pattern)


def parse_target_patterns(target: str | None) -> list[re.Pattern[str]]:
    if not target:
        return []
    return [wildcard_to_regex(item) for item in target.split(",") if item]


def matches_target(name: str, patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    return any(pattern.match(name) for pattern in patterns)


def resolve_image_path(image_path: str, question_file: str) -> str:
    if os.path.isabs(image_path):
        return image_path

    candidate = os.path.join(os.path.dirname(os.path.abspath(question_file)), image_path)
    if os.path.exists(candidate):
        return candidate

    return image_path


class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, image_processor, model_config, question_file, conv_mode="llava_v1"):
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.question_file = question_file
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = resolve_image_path(line["image"], self.question_file)
        question = line["text"]
        if self.model_config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_file).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_tensor": image_tensor,
            "image_size": image.size,
        }

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_tensors = [item["image_tensor"] for item in batch]
    image_sizes = [item["image_size"] for item in batch]

    max_len = max(item.shape[0] for item in input_ids)
    padded_input_ids = []
    padded_labels = []

    for input_id, label in zip(input_ids, labels):
        pad_len = max_len - input_id.shape[0]
        if pad_len > 0:
            input_id = torch.cat(
                [input_id, torch.full((pad_len,), PAD_TOKEN_ID, dtype=input_id.dtype)],
                dim=0,
            )
            label = torch.cat(
                [label, torch.full((pad_len,), IGNORE_INDEX, dtype=label.dtype)],
                dim=0,
            )
        padded_input_ids.append(input_id)
        padded_labels.append(label)

    return (
        torch.stack(padded_input_ids, dim=0),
        torch.stack(padded_labels, dim=0),
        torch.stack(image_tensors, dim=0),
        image_sizes,
    )


def create_data_loader(questions, tokenizer, image_processor, model_config, question_file, batch_size=1, num_workers=0):
    dataset = CustomDataset(questions, tokenizer, image_processor, model_config, question_file=question_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )


def collect_linear_modules(model, target: str | None = None):
    patterns = parse_target_patterns(target)
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and matches_target(name, patterns) and \
            'vision_tower' not in name:
            modules.append((name, module))
    return modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWARe grad analysis with batch and multi-GPU support")
    parser.add_argument("--model-path", type=str, required=True, help="Path to pretrained model or model identifier")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument(
        "--parallel-mode",
        type=str,
        default=None,
        choices=[None, "data", "model"],
        help="Parallelism mode: 'None' (single GPU), 'data' (samples across GPUs), 'model' (layers across GPUs)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--log-interval", type=int, default=10, help="Print progress every N batches")
    parser.add_argument("--question-file", type=str, default="results/data/aware_ds.jsonl")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target to compute activations split by comma eg. ('*.q_proj,*.k_proj,*.v_proj')",
    )
    return parser.parse_args()


def main():
    disable_torch_init()
    args = parse_args()

    output_path = os.path.join(args.output_dir, "analysis_grad.pt")
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping analysis.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("AWARe Gradient Analysis")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Question file: {args.question_file}")
    print(f"Target: {args.target}")
    print("=" * 80)

    if args.parallel_mode not in (None, "", "data", "model"):
        raise ValueError(f"Unsupported parallel mode: {args.parallel_mode}")
    if args.parallel_mode is not None:
        raise NotImplementedError("Gradient analysis currently runs in single-process mode. Please leave --parallel-mode unset.")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        cls=LlavaLlamaForCausalLM,
        target=args.target,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    global PAD_TOKEN_ID
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    PAD_TOKEN_ID = tokenizer.pad_token_id

    model.eval()
    model.zero_grad(set_to_none=True)

    with open(args.question_file, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} samples")

    data_loader = create_data_loader(
        questions,
        tokenizer,
        image_processor,
        model.config,
        question_file=args.question_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    tracked_modules = collect_linear_modules(model, args.target)
    print(f"Tracking {len(tracked_modules)} linear modules")

    grad_sums = defaultdict(lambda: None)
    grad_counts = defaultdict(int)
    total_loss = 0.0
    total_samples = 0
    start_time = time.time()

    device = next(model.parameters()).device

    for batch in tqdm(data_loader, desc="Processing batches"):
        input_ids, labels, image_tensors, image_sizes = batch

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        image_tensors = image_tensors.to(device=device, dtype=torch.float16)

        outputs = model(
            input_ids=input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model forward returned no loss. Check the input and label construction.")

        loss.backward()

        total_loss += loss.item() * input_ids.shape[0]
        total_samples += input_ids.shape[0]

        for module_name, module in tracked_modules:
            grad = module.weight.grad
            if grad is None:
                continue

            grad_vec = grad.detach().float().norm(p=2, dim=1).cpu()
            if grad_sums[module_name] is None:
                grad_sums[module_name] = grad_vec.clone()
            else:
                grad_sums[module_name] += grad_vec
            grad_counts[module_name] += 1

        model.zero_grad(set_to_none=True)

    grad_stats = {}
    for module_name, summed_grad in grad_sums.items():
        if summed_grad is None or grad_counts[module_name] == 0:
            continue
        grad_stats[module_name] = summed_grad / grad_counts[module_name]

    torch.save(grad_stats, output_path)

    metadata = {
        "model_path": args.model_path,
        "question_file": args.question_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target": args.target,
        "num_samples": total_samples,
        "average_loss": total_loss / max(total_samples, 1),
        "num_tracked_modules": len(grad_stats),
        "elapsed_seconds": time.time() - start_time,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Gradient analysis completed successfully")
    print(f"Samples processed: {total_samples}")
    print(f"Average loss: {metadata['average_loss']:.6f}")
    print(f"Tracked modules saved: {len(grad_stats)}")
    print(f"Saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()