"""Standalone RAQUEL evaluation for affected/unaffected datasets."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List

from src.evaluation import (
    RAQUELEvalConfig,
    evaluate_raquel_splits,
    load_raquel_examples,
    resolve_generation_device,
)
from src.metrics import SemanticMetricConfig
from script.evaluation.utils import load_fine_tuned_model
from src.utils.logging import get_logger

logger = get_logger("script.evaluation.run_raquel_eval", __file__)

warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size will be removed in a future PyTorch release.*",
    category=FutureWarning,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run RAQUEL evaluation."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-3.2-1B",
        help="Base model name for tokenizer loading.",
    )
    parser.add_argument("--affected_file", required=True)
    parser.add_argument("--unaffected_file", required=True)
    parser.add_argument(
        "--output_path",
        help="Output path for results (default: model_path/raquel_eval.json).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Load the model in 4-bit for lower-memory evaluation.",
    )
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--semantic_enabled", action="store_true")
    parser.add_argument(
        "--semantic_model_name",
        default="openai/gpt-5.4-nano-2026-03-17",
    )
    parser.add_argument("--semantic_temperature", type=float, default=0.0)
    parser.add_argument("--semantic_max_tokens", type=int, default=2048)
    parser.add_argument("--semantic_use_custom_api", action="store_true")
    parser.add_argument("--semantic_batch_size", type=int, default=8)
    parser.add_argument("--semantic_max_examples", type=int)
    parser.add_argument("--semantic_max_retries", type=int, default=3)
    parser.add_argument("--semantic_retry_delay", type=float, default=2.0)
    parser.add_argument("--semantic_max_concurrency", type=int, default=1)
    parser.add_argument("--semantic_requests_per_second", type=float, default=0.0)
    args: argparse.Namespace = parser.parse_args()
    return args


def main() -> None:
    """Run RAQUEL evaluation and save results."""
    args: argparse.Namespace = _parse_args()

    affected_examples: List[Dict[str, str]] = load_raquel_examples(args.affected_file)
    unaffected_examples: List[Dict[str, str]] = load_raquel_examples(
        args.unaffected_file
    )

    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        base_model_name=args.base_model,
        device_map_auto=True,
        quantize_4bit=bool(args.quantize_4bit),
        as_trainable=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    device = resolve_generation_device(model, args.device)

    semantic_cfg: SemanticMetricConfig = SemanticMetricConfig(
        enabled=bool(args.semantic_enabled),
        model_name=str(args.semantic_model_name),
        temperature=float(args.semantic_temperature),
        max_tokens=int(args.semantic_max_tokens),
        use_custom_api=bool(args.semantic_use_custom_api),
        batch_size=int(args.semantic_batch_size),
        max_examples=args.semantic_max_examples,
        max_retries=int(args.semantic_max_retries),
        retry_delay=float(args.semantic_retry_delay),
        max_concurrency=int(args.semantic_max_concurrency),
        requests_per_second=float(args.semantic_requests_per_second),
    )

    eval_config = RAQUELEvalConfig(
        batch_size=max(int(args.batch_size), 1),
        max_new_tokens=max(int(args.max_new_tokens), 1),
        max_prompt_length=(
            int(args.max_prompt_length) if args.max_prompt_length else None
        ),
        max_examples=int(args.max_examples) if args.max_examples else None,
        save_predictions=bool(args.save_predictions),
    )

    split_results = evaluate_raquel_splits(
        split_examples={
            "affected": affected_examples,
            "unaffected": unaffected_examples,
        },
        model=model,
        tokenizer=tokenizer,
        config=eval_config,
        device=device,
        semantic_cfg=semantic_cfg,
    )

    output_path: str = (
        args.output_path
        if args.output_path
        else os.path.join(args.model_path, "raquel_eval.json")
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        **split_results,
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "affected_file": args.affected_file,
            "unaffected_file": args.unaffected_file,
            "device": str(device),
            "quantize_4bit": bool(args.quantize_4bit),
            "semantic_enabled": semantic_cfg.enabled,
        },
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("RAQUEL evaluation saved to %s", output_path)


if __name__ == "__main__":
    main()
