#!/usr/bin/env python3
"""
MUSE Core Metrics Evaluation (Metrics 1-4)

This script evaluates the core MUSE metrics on trained unlearning models:
  1. No Verbatim Memorization
  2. No Knowledge Memorization
  3. No Privacy Leakage
  4. Utility Preservation

These metrics are computationally efficient and provide the essential
evaluation of unlearning effectiveness and utility preservation.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

from script.evaluation.utils import load_fine_tuned_model
from src.evaluation import (
    MUSEEvaluator,
    MUSEEvaluationBundle,
    MUSEQAExample,
    evaluate_muse_core_metrics,
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
        description="Run MUSE core metrics (1-4) evaluation on trained models"
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
        help="Output path for results (default: model_path/muse_core_results.json)",
    )
    parser.add_argument(
        "--forget_data", type=str, help="Path to forget dataset (overrides config)"
    )
    parser.add_argument(
        "--retain_data", type=str, help="Path to retain dataset (overrides config)"
    )
    parser.add_argument(
        "--paraphrased_data",
        type=str,
        help="Path to paraphrased dataset (overrides config)",
    )
    parser.add_argument(
        "--non_training_data",
        type=str,
        help="Path to non-training dataset (overrides config)",
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
    parser.add_argument(
        "--use_forget_paraphrases",
        action="store_true",
        help=(
            "Build paraphrased examples from forget set fields when no paraphrased "
            "dataset is provided."
        ),
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
    if args.paraphrased_data:
        config["datasets"]["paraphrased_set"] = args.paraphrased_data
    if args.non_training_data:
        config["datasets"]["non_training_set"] = args.non_training_data
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
        output_path = os.path.join(args.model_path, "muse_core_results.json")

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

    # Build paraphrased examples from forget set when requested.
    if bundle.paraphrased_examples is None and args.use_forget_paraphrases:
        derived_paraphrases: List[Dict[str, Any]] = []
        for example in bundle.knowledge_examples:
            paraphrased_question: Optional[str] = str(
                example.metadata.get("paraphrased_question", "")
            ).strip()
            if not paraphrased_question:
                continue
            paraphrased_answer: str = str(
                example.metadata.get("paraphrased_answer", example.answer)
            )
            derived_paraphrases.append(
                {
                    "question": paraphrased_question,
                    "answer": paraphrased_answer,
                }
            )
        if derived_paraphrases:
            bundle = MUSEEvaluationBundle(
                source=bundle.source,
                benchmark=bundle.benchmark,
                verbatim_examples=bundle.verbatim_examples,
                knowledge_examples=bundle.knowledge_examples,
                utility_examples=bundle.utility_examples,
                paraphrased_examples=[
                    MUSEQAExample(question=item["question"], answer=item["answer"])
                    for item in derived_paraphrases
                ],
                privacy_member_examples=bundle.privacy_member_examples,
                privacy_nonmember_examples=bundle.privacy_nonmember_examples,
                scalability_sets=bundle.scalability_sets,
                sustainability_sets=bundle.sustainability_sets,
            )
        else:
            logger.warning(
                "No paraphrased_question found in forget set; knowledge memorization will be skipped."
            )

    logger.info(
        "Loaded datasets - Verbatim: %d, Knowledge: %d, Utility: %d, Privacy member: %d, Privacy non-member: %d",
        len(bundle.verbatim_examples),
        len(bundle.knowledge_examples),
        len(bundle.utility_examples),
        len(bundle.privacy_member_examples) if bundle.privacy_member_examples else 0,
        len(bundle.privacy_nonmember_examples) if bundle.privacy_nonmember_examples else 0,
    )

    # Optionally subsample for faster evaluation
    sample_num = config["evaluation"].get("sample_num")
    if sample_num and sample_num > 0:
        bundle = sample_muse_bundle(bundle, sample_num=int(sample_num))
        logger.info("Subsampled bundle to %d examples per logical set", sample_num)

    # Create MUSE evaluator
    logger.info("Starting MUSE core metrics evaluation...")
    evaluator = MUSEEvaluator(model, tokenizer, device=resolved_device)

    logger.info("=" * 60)
    logger.info("Running MUSE core metrics (1-4)...")
    logger.info("=" * 60)
    results = evaluate_muse_core_metrics(
        evaluator,
        bundle.knowledge_examples,
        bundle.utility_examples,
        bundle.paraphrased_examples,
        bundle.privacy_nonmember_examples,
        verbatim_examples=bundle.verbatim_examples,
        knowledge_examples=bundle.knowledge_examples,
        privacy_member_examples=bundle.privacy_member_examples,
        privacy_nonmember_examples=bundle.privacy_nonmember_examples,
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
        "paraphrased_dataset_size": (
            len(bundle.paraphrased_examples) if bundle.paraphrased_examples else 0
        ),
        "non_training_dataset_size": (
            len(bundle.privacy_nonmember_examples) if bundle.privacy_nonmember_examples else 0
        ),
        "metrics_evaluated": list(results.keys()),
        "sample_num": sample_num,
        "device": str(resolved_device),
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ MUSE core evaluation results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MUSE CORE METRICS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(
        f"Evaluated examples: {len(bundle.verbatim_examples)} verbatim, {len(bundle.knowledge_examples)} knowledge, {len(bundle.utility_examples)} utility"
    )
    print("-" * 60)

    if "verbatim" in results:
        print(
            f"[Metric 1] Verbatim Memorization ROUGE-L:  {results['verbatim']['verbatim_rouge_mean']:.4f} ± {results['verbatim']['verbatim_rouge_std']:.4f}"
        )
        print(f"           → Lower is better (good unlearning: < 0.2)")

    if "knowledge" in results:
        print(
            f"[Metric 2] Knowledge Memorization ROUGE-L:  {results['knowledge']['knowledge_rouge_mean']:.4f} ± {results['knowledge']['knowledge_rouge_std']:.4f}"
        )
        print(f"           → Lower is better (knowledge removed: < 0.3)")

    if "privacy" in results:
        print(
            f"[Metric 3] Privacy Leakage (MIA Accuracy): {results['privacy']['mia_accuracy']:.4f}"
        )
        print(f"           → ~0.50 is ideal (random guess = good privacy)")

    if "utility" in results:
        print(
            f"[Metric 4] Utility Preservation ROUGE-L:    {results['utility']['utility_rouge_mean']:.4f} ± {results['utility']['utility_rouge_std']:.4f}"
        )
        print(f"           → Higher is better (utility preserved: > 0.6)")

    print("=" * 60)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
