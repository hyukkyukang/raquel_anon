#!/usr/bin/env python3
"""
MUSE Sustainability Evaluation (Metric 6)

This script evaluates the sustainability of unlearning by testing whether
the unlearning persists after continued training on retain/general data.

This is separated from core metrics because:
- Extremely computationally expensive (requires model retraining)
- Takes significant time (depends on training steps)
- Modifies the model (creates a new checkpoint)
- Optional for most evaluation workflows

Process:
1. Evaluate current model (pre-retraining baseline)
2. Fine-tune model on retain set for N steps
3. Re-evaluate after retraining
4. Compare results to check if forgotten knowledge returns
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from script.evaluation.utils import load_fine_tuned_model
from src.dataset.benchmark_loading import load_benchmark_examples
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


class SimpleTextDataset(Dataset):
    """Simple dataset for sustainability retraining."""

    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        if isinstance(example, dict):
            question = str(example.get("question", "")).strip()
            answer = str(example.get("answer", "")).strip()
            text = str(example.get("text", "")).strip()
        else:
            question = str(getattr(example, "question", "")).strip()
            answer = str(getattr(example, "answer", "")).strip()
            text = str(getattr(example, "text", "")).strip()

        if not text:
            text = " ".join(part for part in (question, answer) if part).strip()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


def load_config(config_path: str) -> Dict:
    """Load MUSE evaluation configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _to_plain_records(examples: Iterable[Any]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for example in examples:
        if isinstance(example, dict):
            record = {str(key): str(value) for key, value in example.items()}
        elif is_dataclass(example):
            record = {str(key): str(value) for key, value in asdict(example).items()}
        else:
            record = {}
            for key in ("question", "answer", "text", "prompt", "gt"):
                value = getattr(example, key, None)
                if value is not None:
                    record[key] = str(value)
        records.append(record)
    return records


def retrain_model(
    model,
    tokenizer,
    retain_examples,
    num_steps,
    learning_rate=5e-5,
    batch_size=4,
    device="cuda",
):
    """Fine-tune model on retain set for specified number of steps."""
    logger.info(
        f"Retraining model for {num_steps} steps on {len(retain_examples)} retain examples..."
    )

    # Create dataset and dataloader
    dataset = SimpleTextDataset(retain_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = num_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    model.to(device)

    step = 0
    epoch = 0
    total_loss = 0

    start_time = time.time()

    while step < num_steps:
        epoch += 1
        logger.info(f"Epoch {epoch}")

        for batch in dataloader:
            if step >= num_steps:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1

            if step % 10 == 0:
                avg_loss = total_loss / step
                logger.info(
                    f"Step {step}/{num_steps} - Loss: {loss.item():.4f} (Avg: {avg_loss:.4f})"
                )

    training_time = time.time() - start_time
    avg_loss = total_loss / num_steps

    logger.info(f"Retraining completed in {training_time:.2f} seconds")
    logger.info(f"Average loss: {avg_loss:.4f}")

    model.eval()

    return {
        "total_steps": num_steps,
        "training_time_seconds": training_time,
        "average_loss": avg_loss,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run MUSE sustainability (metric 6) evaluation on trained models"
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
        help="Output path for results (default: model_path/muse_sustainability_results.json)",
    )
    parser.add_argument(
        "--retrained_model_path",
        type=str,
        help="Path to save retrained model (default: model_path/retrained_for_sustainability)",
    )
    parser.add_argument(
        "--forget_data", type=str, help="Path to forget dataset (overrides config)"
    )
    parser.add_argument(
        "--retain_data", type=str, help="Path to retain dataset (overrides config)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of retraining steps (overrides config, default: 100)",
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for retraining (default: 5e-5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for retraining (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for evaluation (cuda/cpu)"
    )
    parser.add_argument(
        "--skip_retraining",
        action="store_true",
        help="Skip retraining and use placeholder implementation",
    )
    parser.add_argument(
        "--save_retrained_model",
        action="store_true",
        help="Save the retrained model checkpoint",
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
    if args.num_steps:
        num_steps = args.num_steps
    else:
        num_steps = config["metrics"]["sustainability"].get(
            "additional_training_steps", 100
        )

    # Set output paths
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.model_path, "muse_sustainability_results.json")

    if args.retrained_model_path:
        retrained_model_path = args.retrained_model_path
    else:
        retrained_model_path = os.path.join(
            args.model_path, "retrained_for_sustainability"
        )

    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        args.base_model,
        device_map_auto=False,  # We need to move model manually for training
        quantize_4bit=False,
        as_trainable=True,  # Need trainable for retraining
    )

    # NOTE: `load_fine_tuned_model(..., device_map_auto=False)` returns the model on CPU.
    # The evaluator moves inputs to `args.device`, so we must move the model first to avoid
    # CPU/GPU tensor mismatches during baseline generation.
    logger.info(f"Moving model to device for evaluation: {args.device}")
    model.to(args.device)
    model.eval()
    resolved_device = resolve_generation_device(model, args.device)

    # Load evaluation datasets
    logger.info("Loading evaluation datasets...")
    bundle = load_muse_bundle_from_config(config)
    sample_num = config.get("evaluation", {}).get("sample_num")
    if sample_num and sample_num > 0:
        bundle = sample_muse_bundle(bundle, sample_num=int(sample_num))
        logger.info("Subsampled bundle to %d examples per logical set", sample_num)

    retain_examples: List[Dict[str, str]]
    if bundle.source == "benchmark":
        benchmark_name = str(config["datasets"].get("benchmark", "muse_news"))
        retain_examples = _to_plain_records(
            load_benchmark_examples(
                {"dataset": {"benchmark": benchmark_name}},
                "retain_train",
                limit=sample_num if sample_num and sample_num > 0 else None,
            )
        )
    else:
        retain_examples = _to_plain_records(bundle.utility_examples)

    sustainability_sets = (
        {
            split_name: _to_plain_records(examples)
            for split_name, examples in sorted(bundle.sustainability_sets.items())
        }
        if bundle.sustainability_sets
        else {"verbatim_eval": _to_plain_records(bundle.verbatim_examples)}
    )

    logger.info(
        "Loaded datasets - Sustainability sets: %d, Retain: %d",
        len(sustainability_sets),
        len(retain_examples),
    )

    # Create MUSE evaluator
    evaluator = MUSEEvaluator(model, tokenizer, device=resolved_device)

    # Evaluate pre-retraining (baseline)
    logger.info("=" * 60)
    logger.info("Phase 1: Evaluating BEFORE retraining (baseline)...")
    logger.info("=" * 60)
    pre_retrain_start = time.time()
    pre_retrain_sets = {
        split_name: evaluator.evaluate_verbatim_memorization(examples)
        for split_name, examples in sustainability_sets.items()
    }
    pre_retrain_time = time.time() - pre_retrain_start
    pre_retrain_results = {
        "verbatim_rouge_mean": sum(
            result["verbatim_rouge_mean"] for result in pre_retrain_sets.values()
        )
        / max(len(pre_retrain_sets), 1),
        "verbatim_rouge_std": sum(
            result["verbatim_rouge_std"] for result in pre_retrain_sets.values()
        )
        / max(len(pre_retrain_sets), 1),
        "sets": pre_retrain_sets,
    }

    logger.info(
        "✓ Pre-retraining ROUGE-L: %.4f",
        pre_retrain_results["verbatim_rouge_mean"],
    )

    # Retrain model or use placeholder
    if args.skip_retraining:
        logger.warning("=" * 60)
        logger.warning("SKIPPING RETRAINING - Using placeholder implementation")
        logger.warning("=" * 60)
        post_retrain_results = pre_retrain_results
        retraining_info = {
            "skipped": True,
            "note": "Placeholder implementation - no actual retraining performed",
        }
    else:
        logger.info("=" * 60)
        logger.info(f"Phase 2: Retraining model for {num_steps} steps...")
        logger.info("=" * 60)

        retraining_info = retrain_model(
            model,
            tokenizer,
            retain_examples,
            num_steps=num_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Save retrained model if requested
        if args.save_retrained_model:
            logger.info(f"Saving retrained model to: {retrained_model_path}")
            os.makedirs(retrained_model_path, exist_ok=True)
            model.save_pretrained(retrained_model_path)
            tokenizer.save_pretrained(retrained_model_path)
            logger.info("✓ Retrained model saved")

        # Evaluate post-retraining
        logger.info("=" * 60)
        logger.info("Phase 3: Evaluating AFTER retraining...")
        logger.info("=" * 60)
        post_retrain_start = time.time()
        post_retrain_sets = {
            split_name: evaluator.evaluate_verbatim_memorization(examples)
            for split_name, examples in sustainability_sets.items()
        }
        post_retrain_time = time.time() - post_retrain_start
        post_retrain_results = {
            "verbatim_rouge_mean": sum(
                result["verbatim_rouge_mean"] for result in post_retrain_sets.values()
            )
            / max(len(post_retrain_sets), 1),
            "verbatim_rouge_std": sum(
                result["verbatim_rouge_std"] for result in post_retrain_sets.values()
            )
            / max(len(post_retrain_sets), 1),
            "sets": post_retrain_sets,
        }

        logger.info(
            "✓ Post-retraining ROUGE-L: %.4f",
            post_retrain_results["verbatim_rouge_mean"],
        )

    # Calculate sustainability metrics
    pre_rouge = pre_retrain_results["verbatim_rouge_mean"]
    post_rouge = post_retrain_results["verbatim_rouge_mean"]
    rouge_delta = post_rouge - pre_rouge
    rouge_delta_pct = (rouge_delta / pre_rouge * 100) if pre_rouge > 0 else 0

    # Prepare results
    results = {
        "sustainability": {
            "pre_retraining_rouge_mean": pre_rouge,
            "post_retraining_rouge_mean": post_rouge,
            "rouge_delta": rouge_delta,
            "rouge_delta_percentage": rouge_delta_pct,
            "pre_retraining_rouge_std": pre_retrain_results["verbatim_rouge_std"],
            "post_retraining_rouge_std": post_retrain_results["verbatim_rouge_std"],
            "pre_retraining_sets": pre_retrain_results.get("sets", {}),
            "post_retraining_sets": post_retrain_results.get("sets", {}),
            "retraining_info": retraining_info,
        },
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "config_path": args.config,
            "dataset_source": bundle.source,
            "benchmark": bundle.benchmark,
            "forget_dataset_size": sum(len(examples) for examples in sustainability_sets.values()),
            "retain_dataset_size": len(retain_examples),
            "retrained_model_path": (
                retrained_model_path if args.save_retrained_model else None
            ),
            "device": str(resolved_device),
        },
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ MUSE sustainability results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MUSE SUSTAINABILITY EVALUATION SUMMARY (Metric 6)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Evaluated sets: {list(sustainability_sets)}")
    if not args.skip_retraining:
        print(
            f"Retraining: {num_steps} steps on {len(retain_examples)} retain examples"
        )
    else:
        print(f"Retraining: SKIPPED (placeholder implementation)")
    print("-" * 60)
    print(f"Pre-retraining ROUGE-L:  {pre_rouge:.4f}")
    print(f"Post-retraining ROUGE-L: {post_rouge:.4f}")
    print(f"Delta:                   {rouge_delta:+.4f} ({rouge_delta_pct:+.2f}%)")
    print("-" * 60)

    # Interpretation
    if args.skip_retraining:
        print("⚠ WARNING: Retraining was skipped - results are placeholders")
    else:
        print("\nSustainability Assessment:")
        if abs(rouge_delta) < 0.05:
            print(
                f"  ✓ EXCELLENT: Minimal change (|Δ| < 0.05) - unlearning is highly sustainable"
            )
        elif abs(rouge_delta) < 0.15:
            print(
                f"  ○ GOOD: Small change (|Δ| < 0.15) - unlearning is reasonably sustainable"
            )
        elif rouge_delta > 0:
            print(f"  △ CONCERN: Knowledge leakage detected (Δ = +{rouge_delta:.4f})")
            print(
                f"            Forgotten information is returning after continued training"
            )
        else:
            print(f"  ○ IMPROVED: Unlearning strengthened (Δ = {rouge_delta:.4f})")
            print(f"              Continued training further reduced memorization")

    print("=" * 60)
    print(f"\nDetailed results saved to: {output_path}")
    if args.save_retrained_model and not args.skip_retraining:
        print(f"Retrained model saved to: {retrained_model_path}")


if __name__ == "__main__":
    main()
