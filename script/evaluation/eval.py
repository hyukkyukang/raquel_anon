"""Legacy CLI wrapper for traditional and MUSE evaluation."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional
import warnings

import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()

from script.evaluation.utils import get_base_model_dir_component, load_fine_tuned_model
from src.evaluation import (
    TraditionalEvalConfig,
    compute_prob,
    compute_rouge_l_recall,
    compute_truth_ratio,
    evaluate_traditional,
    evaluate_with_muse,
    generate_answer,
    load_traditional_examples,
    resolve_generation_device,
)
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)

warnings.warn(
    "script.evaluation.eval is legacy. Prefer script/evaluation/run_raquel_eval.py, "
    "script/evaluation/run_muse_eval.py, or src.evaluation directly.",
    DeprecationWarning,
    stacklevel=2,
)

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"

load_dataset = load_traditional_examples
compute_rouge_l = compute_rouge_l_recall
evaluate = evaluate_traditional

__all__ = [
    "TraditionalEvalConfig",
    "compute_prob",
    "compute_rouge_l",
    "compute_truth_ratio",
    "evaluate",
    "evaluate_with_muse",
    "generate_answer",
    "load_dataset",
    "main",
]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the legacy evaluation wrapper."""
    parser = argparse.ArgumentParser(
        description="Run the legacy traditional evaluation and optional MUSE metrics."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--model_dir",
        help="Base directory containing retain_model/ and unlearned_model/.",
    )
    parser.add_argument("--retain_model_path")
    parser.add_argument("--unlearned_model_path")
    parser.add_argument("--retain_file", default="data/tofu/retain_perturbed.json")
    parser.add_argument("--forget_file", default="data/tofu/forget10.json")
    parser.add_argument("--paraphrased_forget_file")
    parser.add_argument("--non_training_file")
    parser.add_argument("--output_path")
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument("--skip_muse", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the legacy evaluation wrapper using shared library code."""
    args = _parse_args()
    logger.warning(
        "Using legacy evaluation wrapper. Prefer script/evaluation/run_raquel_eval.py "
        "and script/evaluation/run_muse_eval.py for supported workflows."
    )

    default_model_dir = os.path.join(
        "model",
        get_base_model_dir_component(args.base_model),
    )
    model_dir = args.model_dir or default_model_dir
    retain_model_path = args.retain_model_path or os.path.join(model_dir, "retain_model")
    unlearned_model_path = args.unlearned_model_path or os.path.join(
        model_dir,
        "unlearned_model",
    )
    output_path = args.output_path or os.path.join(model_dir, "evaluation_results.json")

    sample_num: Optional[int] = args.sample_num
    if sample_num is None:
        sample_num_env = os.getenv("EVAL_SAMPLE_NUM")
        sample_num = int(sample_num_env) if sample_num_env else 100

    eval_config = TraditionalEvalConfig(
        sample_num=sample_num,
        generation_max_new_tokens=max(int(args.max_new_tokens), 1),
        random_seed=args.seed,
    )

    forget_set = load_traditional_examples(args.forget_file)
    retain_set = load_traditional_examples(args.retain_file)
    paraphrased_forget_set = (
        load_traditional_examples(args.paraphrased_forget_file)
        if args.paraphrased_forget_file
        else None
    )
    non_training_set = (
        load_traditional_examples(args.non_training_file)
        if args.non_training_file
        else None
    )

    retain_model, retain_tokenizer = load_fine_tuned_model(
        retain_model_path,
        args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )
    unlearned_model, unlearned_tokenizer = load_fine_tuned_model(
        unlearned_model_path,
        args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )

    retain_tokenizer.pad_token = retain_tokenizer.eos_token
    unlearned_tokenizer.pad_token = unlearned_tokenizer.eos_token

    resolved_device = resolve_generation_device(unlearned_model, args.device)

    results: Dict[str, Any] = {
        "traditional": evaluate_traditional(
            unlearned_model,
            unlearned_tokenizer,
            forget_set,
            retain_set,
            retain_model,
            retain_tokenizer,
            config=eval_config,
            device=resolved_device,
        ),
        "metadata": {
            "model_dir": model_dir,
            "retain_model_path": retain_model_path,
            "unlearned_model_path": unlearned_model_path,
            "base_model": args.base_model,
            "forget_file": args.forget_file,
            "retain_file": args.retain_file,
            "device": str(resolved_device),
            "sample_num": sample_num,
        },
    }

    if not args.skip_muse:
        results["muse"] = evaluate_with_muse(
            unlearned_model,
            unlearned_tokenizer,
            forget_set,
            retain_set,
            paraphrased_forget_set=paraphrased_forget_set,
            non_training_set=non_training_set,
            config=eval_config,
            device=resolved_device,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("Evaluation results saved to %s", output_path)


if __name__ == "__main__":
    main()
