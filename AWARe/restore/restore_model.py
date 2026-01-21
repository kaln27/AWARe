from __future__ import annotations

import argparse
from pathlib import Path
from utils import restore_aware_model


def run_restore_model(args: argparse.Namespace) -> int:
    if not args.model_path.exists():
        raise FileNotFoundError(f"AWARe model not found: {args.model_path}")

    if not args.target_pos_path.exists():
        raise FileNotFoundError(f"Target position file not found: {args.target_pos_path}")

    print("Restoring AWARe model")
    print(f"  AWARe model: {args.model_path}")
    print(f"  Position file: {args.target_pos_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Base model: {args.model_base}")

    restore_aware_model(
        model_path=args.model_path,
        target_pos_path=args.target_pos_path,
        output_path=args.output_path,
        model_base=args.model_base,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore AWARe model to standard model format",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved AWARe model directory",
    )
    parser.add_argument(
        "--target-pos-path",
        type=Path,
        required=True,
        help="JSON file containing trainable node positions",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for the restored model",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
        help="Original pretrained model name or path",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_restore_model(args)


if __name__ == "__main__":
    main()
