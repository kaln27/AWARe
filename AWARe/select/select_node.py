from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from select_strategy import select_freezable_nodes_balanced, select_freezable_nodes_global_highest

STRATEGY_MAP = {
    "balanced": select_freezable_nodes_balanced,
    "global_highest": select_freezable_nodes_global_highest,
}


def run_select_nodes(args: argparse.Namespace) -> int:
    if not 0 < args.quota <= 100:
        raise ValueError(f"Quota must be between 0 and 100, got {args.quota}")

    if not args.dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {args.dir_path}")

    output_dir = args.dir_path

    output_name = args.output_name
    if output_name is None:
        output_name = f"{args.strategy}_{int(args.quota)}.json"
        print(f"Using output filename: {output_name}")

    output_path = Path(output_dir) / output_name

    if output_path.exists():
        print(f"Output file {output_path} already exists. Skipping selection.")
        exit(0)

    print(f"Loading activation from {args.dir_path / args.analysis_file}...")
    analysis = torch.load(args.dir_path / args.analysis_file)

    position_dict = {}
    selection_fn = STRATEGY_MAP[args.strategy]
    print(f"Applying {args.strategy} strategy with quota={args.quota:.2f}%")
    position_dict = selection_fn(analysis, args.quota)

    for module, position in position_dict.items():
        if isinstance(position, (dict, list)):
            # Convert inner dicts/lists to compact JSON strings
            position_dict[module] = json.dumps(position, separators=(',', ':'))
        else:
            position_dict[module] = position

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(position_dict, f, ensure_ascii=False, indent=2)

    print(f"Selection complete. Results saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate trainable node positions from activation output",
    )

    parser.add_argument(
        "--dir-path",
        type=Path,
        required=True,
        help="Path to output directory for position JSON",
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        required=True,
        help="analysis PT file name",
    )
    parser.add_argument(
        "--quota",
        type=float,
        required=True,
        help="Percentage of parameters to train (0-100)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        choices=list(STRATEGY_MAP.keys()),
        help="Node selection strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for position JSON (default: derived from HDF5 path)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output JSON filename (default: {quota}.json or {strategy}_{quota}.json)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_select_nodes(args)


if __name__ == "__main__":
    main()