#!/usr/bin/env python3
"""Run a bounded end-to-end TinyLlama QLoRA validation workflow."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "local_validation"
    / f"tinyllama_qlora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sample_without_replacement(
    items: Sequence[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count > len(items):
        raise ValueError(f"Requested {count} items from a dataset of size {len(items)}")
    return rng.sample(list(items), count)


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _load_summary(run_results_dir: Path, run_name: str) -> dict[str, Any]:
    summary_path = run_results_dir / run_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json at {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded TinyLlama QLoRA fine-tune, unlearn, RAQUEL, and MUSE validation."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retain-train-count", type=int, default=64)
    parser.add_argument("--retain-val-count", type=int, default=16)
    parser.add_argument("--retain-unlearn-count", type=int, default=32)
    parser.add_argument("--forget-count", type=int, default=32)
    parser.add_argument("--raquel-count", type=int, default=20)
    parser.add_argument("--muse-sample-num", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-generation-num", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--disable-tracking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    output_root = Path(args.output_root).resolve()
    dataset_root = output_root / "datasets"
    run_results_dir = output_root / "results"
    tracking_uri = f"sqlite:///{output_root / 'mlflow.db'}"
    artifact_root = str((output_root / "mlartifacts").resolve())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    finetune_run_name = "local_validation_finetune"
    unlearn_run_name = "local_validation_unlearn"

    retain_pool = _load_json(REPO_ROOT / "data/tofu/retain90.json")
    holdout_pool = _load_json(REPO_ROOT / "data/tofu/holdout10.json")
    forget_pool = _load_json(REPO_ROOT / "data/tofu/forget10.json")
    affected_pool = _load_json(
        REPO_ROOT / "data/aligned_db/affected_synthesized_queries_results.json"
    )
    unaffected_pool = _load_json(
        REPO_ROOT / "data/aligned_db/unaffected_synthesized_queries_results.json"
    )

    retain_sample = _sample_without_replacement(
        retain_pool,
        args.retain_train_count + args.retain_val_count + args.retain_unlearn_count,
        rng,
    )
    retain_train = retain_sample[: args.retain_train_count]
    retain_val = retain_sample[
        args.retain_train_count : args.retain_train_count + args.retain_val_count
    ]
    retain_unlearn = retain_sample[-args.retain_unlearn_count :]
    forget_train = _sample_without_replacement(forget_pool, args.forget_count, rng)
    non_training = _sample_without_replacement(holdout_pool, args.muse_sample_num, rng)
    raquel_affected = _sample_without_replacement(affected_pool, args.raquel_count, rng)
    raquel_unaffected = _sample_without_replacement(
        unaffected_pool, args.raquel_count, rng
    )

    retain_train_path = dataset_root / "retain_train.json"
    retain_val_path = dataset_root / "retain_val.json"
    retain_unlearn_path = dataset_root / "retain_unlearn.json"
    forget_path = dataset_root / "forget.json"
    non_training_path = dataset_root / "non_training.json"
    raquel_affected_path = dataset_root / "raquel_affected.json"
    raquel_unaffected_path = dataset_root / "raquel_unaffected.json"
    muse_config_path = dataset_root / "muse_eval.yaml"

    _write_json(retain_train_path, retain_train)
    _write_json(retain_val_path, retain_val)
    _write_json(retain_unlearn_path, retain_unlearn)
    _write_json(forget_path, forget_train)
    _write_json(non_training_path, non_training)
    _write_json(raquel_affected_path, raquel_affected)
    _write_json(raquel_unaffected_path, raquel_unaffected)

    muse_config = {
        "generation": {
            "max_new_tokens": 48,
            "temperature": 1.0,
            "do_sample": False,
        },
        "evaluation": {"sample_num": args.muse_sample_num},
        "datasets": {
            "forget_set": str(forget_path),
            "retain_set": str(retain_unlearn_path),
            "paraphrased_set": None,
            "non_training_set": str(non_training_path),
        },
        "metrics": {
            "verbatim_memorization": {"enabled": True},
            "knowledge_memorization": {"enabled": False},
            "privacy_leakage": {"enabled": True},
            "utility_preservation": {"enabled": True},
            "scalability": {"enabled": False},
            "sustainability": {"enabled": False},
        },
        "output": {
            "save_individual_scores": True,
            "save_summary_only": False,
            "results_dir": str(output_root / "muse_results"),
        },
    }
    muse_config_path.parent.mkdir(parents=True, exist_ok=True)
    muse_config_path.write_text(yaml.safe_dump(muse_config), encoding="utf-8")

    tracking_override = ["tracking.enabled=false"] if args.disable_tracking else []
    shared_tracking_overrides = (
        []
        if args.disable_tracking
        else [
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.artifact_root={artifact_root}",
            "mlflow.experiment_name=RAQUEL-local-validation",
        ]
    )

    finetune_output = output_root / "finetune_model"
    unlearn_output = output_root / "unlearn_model"

    _run_command(
        [
            sys.executable,
            "script/train/finetune_retain.py",
            "--config-name",
            "finetune/retain_tinyllama_qlora",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            "trainer.num_workers=0",
            f"training.epochs={args.epochs}",
            f"training.val_generation_num={args.val_generation_num}",
            f"training.val_sample_num={args.retain_val_count}",
            f"data.train_file={retain_train_path}",
            f"data.val_file={retain_val_path}",
            f"output.dir={finetune_output}",
            f"checkpoint.dirpath={output_root / 'finetune_ckpts'}",
            f"log_dir={output_root / 'logs_finetune'}",
            f"results.dir={run_results_dir}",
            f"results.run_name={finetune_run_name}",
            f"mlflow.run_name={finetune_run_name}",
            *shared_tracking_overrides,
            *tracking_override,
        ],
        env=env,
    )

    _run_command(
        [
            sys.executable,
            "script/train/unlearn.py",
            "--config-name",
            "unlearn/ga_tinyllama_qlora",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            "trainer.num_workers=0",
            f"training.epochs={args.epochs}",
            f"training.val_generation_num={args.val_generation_num}",
            f"training.val_sample_num={args.muse_sample_num}",
            f"data.forget_file={forget_path}",
            f"data.retain_file={retain_unlearn_path}",
            "model.trained_tag=null",
            f"model.path={finetune_output}",
            f"output.dir={unlearn_output}",
            f"checkpoint.dirpath={output_root / 'unlearn_ckpts'}",
            f"log_dir={output_root / 'logs_unlearn'}",
            f"results.dir={run_results_dir}",
            f"results.run_name={unlearn_run_name}",
            f"mlflow.run_name={unlearn_run_name}",
            *shared_tracking_overrides,
            *tracking_override,
        ],
        env=env,
    )

    raquel_output_path = output_root / "raquel_eval.json"
    muse_output_path = output_root / "muse_eval.json"

    _run_command(
        [
            sys.executable,
            "script/evaluation/run_raquel_eval.py",
            "--model_path",
            str(unlearn_output),
            "--base_model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--affected_file",
            str(raquel_affected_path),
            "--unaffected_file",
            str(raquel_unaffected_path),
            "--output_path",
            str(raquel_output_path),
            "--batch_size",
            "4",
            "--max_new_tokens",
            "48",
            "--quantize_4bit",
        ],
        env=env,
    )

    _run_command(
        [
            sys.executable,
            "script/evaluation/run_muse_eval.py",
            "--model_path",
            str(unlearn_output),
            "--base_model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--config",
            str(muse_config_path),
            "--output_path",
            str(muse_output_path),
            "--quantize_4bit",
        ],
        env=env,
    )

    finetune_summary = _load_summary(run_results_dir, finetune_run_name)
    unlearn_summary = _load_summary(run_results_dir, unlearn_run_name)
    with raquel_output_path.open("r", encoding="utf-8") as handle:
        raquel_results = json.load(handle)
    with muse_output_path.open("r", encoding="utf-8") as handle:
        muse_results = json.load(handle)

    summary = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "output_root": str(output_root),
        "tracking_enabled": not args.disable_tracking,
        "tracking_uri": None if args.disable_tracking else tracking_uri,
        "artifact_root": None if args.disable_tracking else artifact_root,
        "dataset_sizes": {
            "retain_train": len(retain_train),
            "retain_val": len(retain_val),
            "retain_unlearn": len(retain_unlearn),
            "forget_train": len(forget_train),
            "non_training": len(non_training),
            "raquel_affected": len(raquel_affected),
            "raquel_unaffected": len(raquel_unaffected),
        },
        "finetune": finetune_summary.get("final_metrics", {}),
        "unlearn": unlearn_summary.get("final_metrics", {}),
        "raquel": {
            "affected_rougeL_fmeasure": raquel_results["affected"]["rouge"][
                "rougeL_fmeasure"
            ],
            "unaffected_rougeL_fmeasure": raquel_results["unaffected"]["rouge"][
                "rougeL_fmeasure"
            ],
        },
        "muse": {
            "verbatim_rouge_mean": muse_results["verbatim"]["verbatim_rouge_mean"],
            "privacy_mia_accuracy": muse_results["privacy"]["mia_accuracy"],
            "utility_rouge_mean": muse_results["utility"]["utility_rouge_mean"],
        },
        "artifacts": {
            "finetune_model": str(finetune_output),
            "unlearn_model": str(unlearn_output),
            "raquel_eval": str(raquel_output_path),
            "muse_eval": str(muse_output_path),
            "results_dir": str(run_results_dir),
            "finetune_summary": str(run_results_dir / finetune_run_name / "summary.json"),
            "unlearn_summary": str(run_results_dir / unlearn_run_name / "summary.json"),
        },
    }
    _write_json(output_root / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
