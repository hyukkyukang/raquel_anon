#!/usr/bin/env python3
"""
MUSE Scalability Evaluation (Metric 5)

This script evaluates the scalability of unlearning across different forget set sizes.
Tests whether unlearning effectiveness remains consistent as the forget set grows.

This is separated from core metrics because:
- Computationally expensive (multiple evaluations across different subset sizes)
- Requires multiple passes over the forget set
- Useful for analyzing unlearning algorithm behavior at scale
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List

import yaml

from script.evaluation.utils import load_fine_tuned_model
from src.evaluation import (
    MUSEEvaluator,
    load_muse_bundle_from_config,
    resolve_generation_device,
    sample_muse_bundle,
)
from src.utils.logging import get_logger

# Configure logging
logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, __file__)


def load_config(config_path: str) -> Dict:
    """Load MUSE evaluation configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run MUSE scalability (metric 5) evaluation on trained models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name for tokenizer loading",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/muse_eval.yaml",
        help="Path to MUSE evaluation config file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for results (default: model_path/muse_scalability_results.json)",
    )
    parser.add_argument(
        "--forget_data", type=str, help="Path to forget dataset (overrides config)"
    )
    parser.add_argument(
        "--subset_sizes",
        type=int,
        nargs="+",
        help="Subset sizes to evaluate (e.g., 10 50 100 200 500)",
    )
    parser.add_argument(
        "--sample_num", type=int, help="Number of examples to sample (overrides config)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["local_files", "benchmark"],
        help="Dataset source mode (overrides config).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark name for benchmark-backed loading (overrides config).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for evaluation (cuda/cpu)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded MUSE evaluation config from: {args.config}")

    # Override config with command line arguments
    if args.forget_data:
        config["datasets"]["forget_set"] = args.forget_data
    if args.sample_num:
        config["evaluation"]["sample_num"] = args.sample_num
    if args.source:
        config["datasets"]["source"] = args.source
    if args.benchmark:
        config["datasets"]["benchmark"] = args.benchmark
    if args.subset_sizes:
        subset_sizes = args.subset_sizes
    else:
        subset_sizes = config["metrics"]["scalability"].get(
            "subset_sizes", [10, 50, 100, 200, 500]
        )

    # Set output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.model_path, "muse_scalability_results.json")

    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )
    resolved_device = resolve_generation_device(model, args.device)

    logger.info("Loading evaluation datasets...")
    bundle = load_muse_bundle_from_config(config)
    sample_num = config["evaluation"].get("sample_num")
    if sample_num and sample_num > 0:
        bundle = sample_muse_bundle(bundle, sample_num=int(sample_num))
        logger.info("Subsampled bundle to %d examples per logical set", sample_num)

    # Create MUSE evaluator
    logger.info("Starting MUSE scalability evaluation...")
    evaluator = MUSEEvaluator(model, tokenizer, device=resolved_device)

    # Run scalability evaluation
    logger.info("=" * 60)
    logger.info("Metric 5: Evaluating Scalability...")
    logger.info("=" * 60)

    start_time = time.time()
    if bundle.scalability_sets:
        logger.info(
            "Running benchmark-backed scalability evaluation across %d configured subsets",
            len(bundle.scalability_sets),
        )
        scalability_results = {}
        for split_name, examples in sorted(bundle.scalability_sets.items()):
            verbatim_results = evaluator.evaluate_verbatim_memorization(examples)
            scalability_results[split_name] = {
                "verbatim_rouge_mean": verbatim_results["verbatim_rouge_mean"],
                "verbatim_rouge_std": verbatim_results["verbatim_rouge_std"],
                "subset_size": len(examples),
                "logical_split": split_name,
            }
        evaluated_subset_sizes = [
            result["subset_size"] for result in scalability_results.values()
        ]
    else:
        forget_examples = list(bundle.verbatim_examples)
        logger.info("Loaded forget dataset: %d examples", len(forget_examples))

        max_available = len(forget_examples)
        valid_subset_sizes = [size for size in subset_sizes if size <= max_available]
        if len(valid_subset_sizes) < len(subset_sizes):
            skipped = [size for size in subset_sizes if size > max_available]
            logger.warning(
                "Skipping subset sizes %s (exceed available %d examples)",
                skipped,
                max_available,
            )
        subset_sizes = sorted(valid_subset_sizes)

        if not subset_sizes:
            logger.error(
                "No valid subset sizes! All requested sizes exceed available %d examples",
                max_available,
            )
            return

        logger.info("Will evaluate on subset sizes: %s", subset_sizes)
        scalability_results = evaluator.evaluate_scalability(
            forget_examples, subset_sizes=subset_sizes
        )
        evaluated_subset_sizes = subset_sizes
    total_time = time.time() - start_time

    # Prepare results
    results = {
        "scalability": scalability_results,
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "config_path": args.config,
            "dataset_source": bundle.source,
            "benchmark": bundle.benchmark,
            "total_forget_examples": sum(
                len(examples) for examples in bundle.scalability_sets.values()
            )
            if bundle.scalability_sets
            else len(bundle.verbatim_examples),
            "subset_sizes_evaluated": evaluated_subset_sizes,
            "total_evaluation_time_seconds": total_time,
            "device": str(resolved_device),
        },
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ MUSE scalability results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MUSE SCALABILITY EVALUATION SUMMARY (Metric 5)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    if bundle.scalability_sets:
        print(f"Dataset source: {bundle.source} ({bundle.benchmark})")
        print(f"Evaluated logical subsets: {list(sorted(bundle.scalability_sets))}")
    else:
        print(f"Total forget set size: {len(bundle.verbatim_examples)} examples")
        print(f"Evaluated subset sizes: {evaluated_subset_sizes}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print("-" * 60)
    print("Verbatim Memorization ROUGE-L by Subset Size:")
    print("-" * 60)

    # Print results in table format
    print(f"{'Subset Size':<15} {'ROUGE-L Mean':<15} {'Interpretation'}")
    print("-" * 60)

    for size_key in sorted(
        scalability_results.keys(),
        key=lambda x: scalability_results[x]["subset_size"],
    ):
        size_result = scalability_results[size_key]
        rouge_mean = size_result["verbatim_rouge_mean"]
        size = size_result["subset_size"]

        # Interpretation
        if rouge_mean < 0.2:
            interpretation = "✓ Excellent"
        elif rouge_mean < 0.4:
            interpretation = "○ Good"
        elif rouge_mean < 0.6:
            interpretation = "△ Moderate"
        else:
            interpretation = "✗ Poor"

        print(f"{size:<15} {rouge_mean:<15.4f} {interpretation}")

    print("-" * 60)

    # Calculate consistency
    rouge_scores = [
        result["verbatim_rouge_mean"] for result in scalability_results.values()
    ]
    min_score = min(rouge_scores)
    max_score = max(rouge_scores)
    score_range = max_score - min_score

    print(f"\nConsistency Analysis:")
    print(f"  Min ROUGE-L: {min_score:.4f}")
    print(f"  Max ROUGE-L: {max_score:.4f}")
    print(f"  Range: {score_range:.4f}")

    if score_range < 0.1:
        print(f"  → Highly consistent (range < 0.1): Excellent scalability ✓")
    elif score_range < 0.2:
        print(f"  → Moderately consistent (range < 0.2): Good scalability ○")
    else:
        print(f"  → Variable performance (range ≥ 0.2): Scalability concerns △")

    print("=" * 60)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
