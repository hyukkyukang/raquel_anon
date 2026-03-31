"""Export a fine-tune Lightning checkpoint into a Hugging Face model directory."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from omegaconf import DictConfig

from src.training.builders import create_pl_module
from src.utils.logging import get_logger

logger = get_logger("script.analysis.export_finetune_checkpoint_to_adapter", __file__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fine-tune Lightning checkpoint to a HF adapter directory."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional CUDA device override, e.g. cuda:0. Defaults to cfg device map.",
    )
    return parser.parse_args()


def _resolve_cfg(raw_cfg: Any, *, device: str | None) -> DictConfig:
    if not isinstance(raw_cfg, DictConfig):
        raise TypeError(f"Expected checkpoint cfg to be a DictConfig, got {type(raw_cfg)!r}")

    cfg = raw_cfg.copy()
    cfg.output.dir = str(os.path.abspath(cfg.output.dir))
    if device:
        cfg.model.device_map = device
    cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cfg.trainer.devices = 1
    return cfg


def main() -> None:
    args = _parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    output_dir = os.path.abspath(args.output_dir)

    logger.info("Loading Lightning checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    raw_cfg = hyper_parameters.get("cfg")
    if raw_cfg is None:
        raise ValueError("Checkpoint is missing hyper_parameters['cfg']")

    cfg = _resolve_cfg(raw_cfg, device=args.device)
    cfg.output.dir = output_dir

    logger.info("Rebuilding fine-tune module for export to %s", output_dir)
    module = create_pl_module(cfg, datamodule=None)
    if module is None:
        raise RuntimeError(
            f"Refusing to export because target directory already looks complete: {output_dir}"
        )

    logger.info("Loading checkpoint state_dict")
    load_result = module.load_state_dict(checkpoint["state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Checkpoint load had missing=%d unexpected=%d keys",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    os.makedirs(output_dir, exist_ok=True)
    module.save_model(output_dir)
    logger.info("Export completed: %s", output_dir)


if __name__ == "__main__":
    main()
