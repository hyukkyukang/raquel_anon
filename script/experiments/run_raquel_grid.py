"""Run a grid of RAQUEL unlearning experiments via Hydra overrides."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _build_command(
    *,
    model_path: str,
    train_affected: str,
    train_unaffected: str,
    eval_affected: str | None,
    eval_unaffected: str | None,
    method: str,
    regularization: str,
    seed: int,
    tag: str,
    extra_overrides: Iterable[str],
) -> List[str]:
    cmd = [
        "python",
        "script/train/unlearn.py",
        f"model.path={model_path}",
        f"data.forget_file={train_affected}",
        f"data.retain_file={train_unaffected}",
        f"unlearning.method={method}",
        f"regularization={regularization}",
        f"seed={seed}",
        f"tag={tag}",
        "evaluation.raquel.enabled=true",
    ]
    if eval_affected and eval_unaffected:
        cmd.extend(
            [
                f"evaluation.raquel.affected_file={eval_affected}",
                f"evaluation.raquel.unaffected_file={eval_unaffected}",
            ]
        )
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAQUEL unlearning grid.")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--train_dir", help="Directory with affected_train.json etc.")
    parser.add_argument(
        "--train_dir_template",
        help="Directory template with {ratio} placeholder",
    )
    parser.add_argument("--eval_dir", help="Directory with affected_test.json etc.")
    parser.add_argument("--methods", default="ga,npo,idk,dpo")
    parser.add_argument("--regularizations", default="gd,kl")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ratios", default="1.0")
    parser.add_argument(
        "--extra_overrides",
        default="",
        help="Comma-separated Hydra overrides (e.g., training.epochs=5,training.learning_rate=1e-5)",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands")
    parser.add_argument(
        "--log_file",
        default="logs/raquel_grid_commands.txt",
        help="File to save generated commands",
    )
    args = parser.parse_args()

    methods = _split_list(args.methods)
    regs = _split_list(args.regularizations)
    seeds = [int(x) for x in _split_list(args.seeds)]
    ratios = [float(x) for x in _split_list(args.ratios)]
    extra_overrides = _split_list(args.extra_overrides)

    model_path = _resolve_path(args.model_path)
    eval_dir = _resolve_path(args.eval_dir) if args.eval_dir else None

    commands: List[str] = []
    for ratio in ratios:
        if args.train_dir_template:
            train_dir = _resolve_path(args.train_dir_template.format(ratio=ratio))
        else:
            train_dir = _resolve_path(args.train_dir) if args.train_dir else ""
        if not train_dir:
            raise ValueError("train_dir or train_dir_template must be set.")

        train_affected = os.path.join(train_dir, "affected_train.json")
        train_unaffected = os.path.join(train_dir, "unaffected_train.json")

        eval_affected = (
            os.path.join(eval_dir, "affected_test.json") if eval_dir else None
        )
        eval_unaffected = (
            os.path.join(eval_dir, "unaffected_test.json") if eval_dir else None
        )

        for method in methods:
            for reg in regs:
                for seed in seeds:
                    tag = f"raquel_{method}_{reg}_r{ratio}_s{seed}"
                    cmd = _build_command(
                        model_path=model_path,
                        train_affected=train_affected,
                        train_unaffected=train_unaffected,
                        eval_affected=eval_affected,
                        eval_unaffected=eval_unaffected,
                        method=method,
                        regularization=reg,
                        seed=seed,
                        tag=tag,
                        extra_overrides=extra_overrides,
                    )
                    commands.append(shlex.join(cmd))

                    if args.execute:
                        subprocess.run(cmd, check=True)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {log_path}")


if __name__ == "__main__":
    main()
