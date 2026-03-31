#!/usr/bin/env python3
"""
Standalone script to run MUSE evaluation on trained models.
Can be used independently of the training pipeline.
"""

import argparse
import json
import logging
import os
import warnings
from typing import Dict

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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
logger = get_logger(__name__, __file__)

warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size will be removed in a future PyTorch release.*",
    category=FutureWarning,
)


def load_config(config_path: str) -> Dict:
    """Load MUSE evaluation configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run MUSE evaluation on trained models"
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
        help="Output path for results (default: model_path/muse_results.json)",
    )
    parser.add_argument(
        "--forget_data", type=str, help="Path to forget dataset (overrides config)"
    )
    parser.add_argument(
        "--retain_data", type=str, help="Path to retain dataset (overrides config)"
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
        "--quantize_4bit",
        action="store_true",
        help="Load the model in 4-bit for lower-memory evaluation.",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded MUSE evaluation config from: {args.config}")

    # Override config with command line arguments
    if args.forget_data:
        config["datasets"]["forget_set"] = args.forget_data
    if args.retain_data:
        config["datasets"]["retain_set"] = args.retain_data
    if args.sample_num:
        config["evaluation"]["sample_num"] = args.sample_num
    if args.source:
        config["datasets"]["source"] = args.source
    if args.benchmark:
        config["datasets"]["benchmark"] = args.benchmark

    # Set output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.model_path, "muse_results.json")

    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        args.base_model,
        device_map_auto=True,
        quantize_4bit=bool(args.quantize_4bit),
        as_trainable=False,
    )
    resolved_device = resolve_generation_device(model, None)

    logger.info("Loading evaluation datasets...")
    bundle = load_muse_bundle_from_config(config)
    sample_num = config["evaluation"].get("sample_num")
    if sample_num and sample_num > 0:
        bundle = sample_muse_bundle(bundle, sample_num=int(sample_num))
        logger.info("Subsampled bundle to %d examples per logical set", sample_num)

    # Run MUSE evaluation
    logger.info("Starting MUSE evaluation...")
    evaluator = MUSEEvaluator(
        model,
        tokenizer,
        device=resolved_device,
        default_max_new_tokens=int(config.get("generation", {}).get("max_new_tokens", 100)),
    )

    # Check which metrics are enabled
    enabled_metrics = {
        k: v for k, v in config["metrics"].items() if v.get("enabled", True)
    }
    logger.info(f"Enabled metrics: {list(enabled_metrics.keys())}")

    # Run evaluation based on enabled metrics
    results = {}

    if enabled_metrics.get("verbatim_memorization"):
        logger.info("Running verbatim memorization evaluation...")
        results["verbatim"] = evaluator.evaluate_verbatim_memorization(
            bundle.verbatim_examples
        )

    if enabled_metrics.get("knowledge_memorization"):
        logger.info("Running knowledge memorization evaluation...")
        results["knowledge"] = evaluator.evaluate_knowledge_memorization(
            bundle.knowledge_examples,
            bundle.paraphrased_examples,
        )

    if enabled_metrics.get("privacy_leakage") and bundle.privacy_nonmember_examples:
        logger.info("Running privacy leakage evaluation...")
        results["privacy"] = evaluator.evaluate_privacy_leakage(
            bundle.privacy_member_examples or bundle.knowledge_examples,
            bundle.privacy_nonmember_examples,
        )
    elif enabled_metrics.get("privacy_leakage"):
        logger.warning("Privacy leakage enabled but no non-training data provided")

    if enabled_metrics.get("utility_preservation"):
        logger.info("Running utility preservation evaluation...")
        results["utility"] = evaluator.evaluate_utility_preservation(
            bundle.utility_examples
        )

    if enabled_metrics.get("scalability"):
        if bundle.scalability_sets:
            logger.warning(
                "Benchmark-backed scalability sets are loaded, but official metric-5 evaluation "
                "is not yet implemented in run_muse_eval.py. Skipping."
            )
        else:
            logger.info("Running scalability evaluation...")
            subset_sizes = config["metrics"]["scalability"].get(
                "subset_sizes", [10, 50, 100, 200]
            )
            results["scalability"] = evaluator.evaluate_scalability(
                bundle.knowledge_examples, subset_sizes
            )

    if enabled_metrics.get("sustainability"):
        logger.warning(
            "Metric 6 sustainability should be run via script/evaluation/run_muse_sustainability.py."
        )

    # Add metadata to results
    results["metadata"] = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "config_path": args.config,
        "dataset_source": bundle.source,
        "benchmark": bundle.benchmark,
        "forget_dataset_size": len(bundle.knowledge_examples),
        "retain_dataset_size": len(bundle.utility_examples),
        "enabled_metrics": list(enabled_metrics.keys()),
        "sample_num": sample_num,
        "quantize_4bit": bool(args.quantize_4bit),
        "device": str(resolved_device),
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"MUSE evaluation results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("MUSE EVALUATION SUMMARY")
    print("=" * 50)

    if "verbatim" in results:
        print(
            f"Verbatim Memorization ROUGE-L: {results['verbatim']['verbatim_rouge_mean']:.4f}"
        )

    if "knowledge" in results:
        print(
            f"Knowledge Memorization ROUGE-L: {results['knowledge']['knowledge_rouge_mean']:.4f}"
        )

    if "privacy" in results:
        print(
            f"Privacy Leakage (MIA Accuracy): {results['privacy']['mia_accuracy']:.4f}"
        )

    if "utility" in results:
        print(
            f"Utility Preservation ROUGE-L: {results['utility']['utility_rouge_mean']:.4f}"
        )

    if "scalability" in results:
        print("Scalability Results:")
        for size_key, size_result in results["scalability"].items():
            print(f"  {size_key}: ROUGE-L {size_result['verbatim_rouge_mean']:.4f}")

    if "sustainability" in results:
        print(
            f"Sustainability ROUGE-L: {results['sustainability']['sustainability_rouge_mean']:.4f}"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()
