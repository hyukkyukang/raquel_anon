#!/usr/bin/env python3
"""Run a bounded end-to-end MUSE-News benchmark smoke validation workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "local_validation"
    / f"muse_news_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_FINETUNE_CONFIG = "finetune/muse_news_smoke_tinyllama_qlora"
DEFAULT_UNLEARN_CONFIG = "unlearn/muse_news_smoke_ga_tinyllama_qlora"
DEFAULT_MUSE_CONFIG = "config/muse_eval_muse_news_smoke.yaml"
DEFAULT_SCALABILITY_CONFIG = "config/muse_eval_muse_news_scalability_smoke.yaml"
DEFAULT_SUSTAINABILITY_CONFIG = "config/muse_eval_muse_news_sustainability_smoke.yaml"
DEFAULT_PROFILE_CONFIG = None


def _default_args_map() -> dict[str, Any]:
    return {
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "cuda_visible_devices": "0",
        "train_sample_num": 8,
        "forget_sample_num": 8,
        "retain_sample_num": 8,
        "eval_sample_num": 8,
        "max_steps": 2,
        "disable_tracking": False,
        "run_scalability": False,
        "run_sustainability": False,
        "sustainability_steps": 1,
        "sustainability_batch_size": 1,
        "sustainability_learning_rate": 5e-5,
        "sustainability_skip_retraining": False,
        "finetune_config_name": DEFAULT_FINETUNE_CONFIG,
        "unlearn_config_name": DEFAULT_UNLEARN_CONFIG,
        "muse_config": DEFAULT_MUSE_CONFIG,
        "scalability_config": DEFAULT_SCALABILITY_CONFIG,
        "sustainability_config": DEFAULT_SUSTAINABILITY_CONFIG,
        "profile_config": DEFAULT_PROFILE_CONFIG,
    }


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_profile_config(profile_path: str | None) -> dict[str, Any]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Validation profile must be a mapping: {profile_path}")
    return loaded


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded benchmark-backed MUSE-News fine-tune, unlearn, and "
            "MUSE core evaluation smoke workflow."
        )
    )
    parser.add_argument("--profile-config", default=argparse.SUPPRESS)
    parser.add_argument("--output-root", default=argparse.SUPPRESS)
    parser.add_argument(
        "--base-model",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--cuda-visible-devices", default=argparse.SUPPRESS)
    parser.add_argument("--train-sample-num", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--forget-sample-num", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--retain-sample-num", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval-sample-num", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--disable-tracking", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--run-scalability", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--run-sustainability", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--sustainability-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sustainability-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sustainability-learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--sustainability-skip-retraining",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finetune-config-name",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unlearn-config-name",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--muse-config",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--scalability-config",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sustainability-config",
        default=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(argv)
    parsed_dict = vars(parsed)
    profile = _load_profile_config(parsed_dict.get("profile_config"))
    merged = _default_args_map()
    merged.update(profile)
    merged.update(parsed_dict)
    if merged.get("output_root") in (None, ""):
        merged["output_root"] = str(DEFAULT_OUTPUT_ROOT)
    return argparse.Namespace(**merged)


def _shared_training_overrides(args: argparse.Namespace, run_results_dir: Path) -> list[str]:
    overrides = [
        f"+trainer.max_steps={args.max_steps}",
        f"results.dir={run_results_dir}",
    ]
    if args.disable_tracking:
        overrides.extend(["tracking.enabled=false", "mlflow.enabled=false"])
    return overrides


def _build_finetune_command(
    args: argparse.Namespace,
    *,
    output_root: Path,
    finetune_output: Path,
    run_results_dir: Path,
    finetune_run_name: str,
) -> list[str]:
    return [
        sys.executable,
        "script/train/finetune_full.py",
        f"--config-name={args.finetune_config_name}",
        f"tag={finetune_run_name}",
        f"log_dir={output_root / 'logs_finetune'}",
        f"output.dir={finetune_output}",
        f"training.train_sample_num={args.train_sample_num}",
        f"training.val_sample_num={min(args.eval_sample_num, args.retain_sample_num)}",
        f"results.run_name={finetune_run_name}",
        *_shared_training_overrides(args, run_results_dir),
    ]


def _build_unlearn_command(
    args: argparse.Namespace,
    *,
    output_root: Path,
    finetune_output: Path,
    unlearn_output: Path,
    run_results_dir: Path,
    unlearn_run_name: str,
) -> list[str]:
    return [
        sys.executable,
        "script/train/unlearn.py",
        f"--config-name={args.unlearn_config_name}",
        f"tag={unlearn_run_name}",
        f"log_dir={output_root / 'logs_unlearn'}",
        f"output.dir={unlearn_output}",
        f"model.path={finetune_output}",
        f"training.forget_sample_num={args.forget_sample_num}",
        f"training.retain_sample_num={args.retain_sample_num}",
        f"results.run_name={unlearn_run_name}",
        *_shared_training_overrides(args, run_results_dir),
    ]


def _build_muse_command(
    args: argparse.Namespace,
    *,
    unlearn_output: Path,
    muse_output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "script/evaluation/run_muse_core.py",
        "--model_path",
        str(unlearn_output),
        "--base_model",
        args.base_model,
        "--config",
        args.muse_config,
        "--source",
        "benchmark",
        "--benchmark",
        "muse_news",
        "--sample_num",
        str(args.eval_sample_num),
        "--device",
        "cuda",
        "--output_path",
        str(muse_output_path),
    ]


def _build_scalability_command(
    args: argparse.Namespace,
    *,
    unlearn_output: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "script/evaluation/run_muse_scalability.py",
        "--model_path",
        str(unlearn_output),
        "--base_model",
        args.base_model,
        "--config",
        args.scalability_config,
        "--source",
        "benchmark",
        "--benchmark",
        "muse_news",
        "--sample_num",
        str(args.eval_sample_num),
        "--device",
        "cuda",
        "--output_path",
        str(output_path),
    ]


def _build_sustainability_command(
    args: argparse.Namespace,
    *,
    unlearn_output: Path,
    output_path: Path,
    retrained_model_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "script/evaluation/run_muse_sustainability.py",
        "--model_path",
        str(unlearn_output),
        "--base_model",
        args.base_model,
        "--config",
        args.sustainability_config,
        "--source",
        "benchmark",
        "--benchmark",
        "muse_news",
        "--sample_num",
        str(args.eval_sample_num),
        "--num_steps",
        str(args.sustainability_steps),
        "--learning_rate",
        str(args.sustainability_learning_rate),
        "--batch_size",
        str(args.sustainability_batch_size),
        "--device",
        "cuda",
        "--output_path",
        str(output_path),
        "--retrained_model_path",
        str(retrained_model_path),
    ]
    if args.sustainability_skip_retraining:
        command.append("--skip_retraining")
    return command


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    models_dir = output_root / "models"
    run_results_dir = output_root / "results"
    finetune_output = models_dir / "finetune_tinyllama"
    unlearn_output = models_dir / "unlearn_tinyllama_ga"
    muse_output_path = output_root / "muse_core_results.json"
    scalability_output_path = output_root / "muse_scalability_results.json"
    sustainability_output_path = output_root / "muse_sustainability_results.json"
    sustainability_model_dir = models_dir / "retrained_for_sustainability"
    summary_path = output_root / "validation_summary.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    finetune_run_name = "muse_news_smoke_finetune"
    unlearn_run_name = "muse_news_smoke_unlearn"

    _run_command(
        _build_finetune_command(
            args,
            output_root=output_root,
            finetune_output=finetune_output,
            run_results_dir=run_results_dir,
            finetune_run_name=finetune_run_name,
        ),
        env=env,
    )

    _run_command(
        _build_unlearn_command(
            args,
            output_root=output_root,
            finetune_output=finetune_output,
            unlearn_output=unlearn_output,
            run_results_dir=run_results_dir,
            unlearn_run_name=unlearn_run_name,
        ),
        env=env,
    )

    _run_command(
        _build_muse_command(
            args,
            unlearn_output=unlearn_output,
            muse_output_path=muse_output_path,
        ),
        env=env,
    )

    scalability_results: dict[str, Any] | None = None
    sustainability_results: dict[str, Any] | None = None

    if args.run_scalability:
        _run_command(
            _build_scalability_command(
                args,
                unlearn_output=unlearn_output,
                output_path=scalability_output_path,
            ),
            env=env,
        )
        scalability_results = _load_json(scalability_output_path)

    if args.run_sustainability:
        _run_command(
            _build_sustainability_command(
                args,
                unlearn_output=unlearn_output,
                output_path=sustainability_output_path,
                retrained_model_path=sustainability_model_dir,
            ),
            env=env,
        )
        sustainability_results = _load_json(sustainability_output_path)

    finetune_summary = _load_json(run_results_dir / finetune_run_name / "summary.json")
    unlearn_summary = _load_json(run_results_dir / unlearn_run_name / "summary.json")
    muse_results = _load_json(muse_output_path)

    summary = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "output_root": str(output_root),
        "benchmark": "muse_news",
        "base_model": args.base_model,
        "sample_counts": {
            "train_sample_num": args.train_sample_num,
            "forget_sample_num": args.forget_sample_num,
            "retain_sample_num": args.retain_sample_num,
            "eval_sample_num": args.eval_sample_num,
        },
        "max_steps": args.max_steps,
        "artifacts": {
            "finetune_model": str(finetune_output),
            "unlearn_model": str(unlearn_output),
            "muse_core_results": str(muse_output_path),
            "finetune_summary": str(run_results_dir / finetune_run_name / "summary.json"),
            "unlearn_summary": str(run_results_dir / unlearn_run_name / "summary.json"),
            "muse_scalability_results": (
                str(scalability_output_path) if args.run_scalability else None
            ),
            "muse_sustainability_results": (
                str(sustainability_output_path) if args.run_sustainability else None
            ),
        },
        "configs": {
            "finetune": args.finetune_config_name,
            "unlearn": args.unlearn_config_name,
            "muse_eval": args.muse_config,
            "scalability_eval": args.scalability_config if args.run_scalability else None,
            "sustainability_eval": (
                args.sustainability_config if args.run_sustainability else None
            ),
        },
        "finetune": finetune_summary.get("final_metrics", {}),
        "unlearn": unlearn_summary.get("final_metrics", {}),
        "muse": {
            "verbatim_rouge_mean": muse_results["verbatim"]["verbatim_rouge_mean"],
            "knowledge_rouge_mean": muse_results["knowledge"]["knowledge_rouge_mean"],
            "mia_accuracy": muse_results["privacy"]["mia_accuracy"],
            "utility_rouge_mean": muse_results["utility"]["utility_rouge_mean"],
        },
    }
    if scalability_results is not None:
        summary["scalability"] = {
            "subsets": sorted(scalability_results["scalability"]),
            "metadata": scalability_results.get("metadata", {}),
        }
    if sustainability_results is not None:
        summary["sustainability"] = {
            "pre_retraining_rouge_mean": sustainability_results["sustainability"][
                "pre_retraining_rouge_mean"
            ],
            "post_retraining_rouge_mean": sustainability_results["sustainability"][
                "post_retraining_rouge_mean"
            ],
            "rouge_delta": sustainability_results["sustainability"]["rouge_delta"],
            "metadata": sustainability_results.get("metadata", {}),
        }
    _write_json(summary_path, summary)

    print("\nValidation summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
