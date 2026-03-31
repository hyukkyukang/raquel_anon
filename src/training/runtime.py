"""Runtime helpers for orchestrating training runs."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


def setup_environment(cfg: DictConfig) -> None:
    """Set up reproducibility and output directories."""
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    os.makedirs(cfg.output.dir, exist_ok=True)
    if hasattr(cfg, "checkpoint") and hasattr(cfg.checkpoint, "dirpath"):
        os.makedirs(cfg.checkpoint.dirpath, exist_ok=True)


def coerce_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Convert Lightning metrics to JSON-serializable scalar floats."""
    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            cleaned[key] = float(value.item())
            continue
        if isinstance(value, (int, float)):
            cleaned[key] = float(value)
            continue
        if hasattr(value, "item"):
            try:
                cleaned[key] = float(value.item())
            except Exception:
                continue
    return cleaned


def log_effective_batch_size(cfg: DictConfig) -> None:
    """Log the effective per-device batch size after gradient accumulation."""
    accumulate_grad_batches = cfg.training["gradient_accumulation_steps"]
    micro_batch_size = cfg.training["train_batch_size"]
    logger.info(
        "Using micro-batch size=%d with gradient accumulation steps=%d "
        "(effective per-device batch size=%d).",
        micro_batch_size,
        accumulate_grad_batches,
        micro_batch_size * accumulate_grad_batches,
    )


def create_trainer(
    cfg: DictConfig,
    *,
    callbacks: list[object],
    logger_instance: Any,
) -> pl.Trainer:
    """Create the Lightning trainer for a run."""
    return pl.Trainer(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        precision=cfg.trainer.get("precision", 16),
        max_epochs=cfg.training.get("epochs", cfg.trainer.get("max_epochs", 3)),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg.training["gradient_accumulation_steps"],
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        enable_checkpointing=cfg.trainer.get("enable_checkpointing", True),
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", True),
        enable_model_summary=cfg.trainer.get("enable_model_summary", True),
        deterministic=cfg.trainer.get("deterministic", False),
        callbacks=callbacks,
        logger=logger_instance,
        default_root_dir=cfg.log_dir,
    )


def cleanup_checkpoints(cfg: DictConfig) -> None:
    """Remove intermediate checkpoint files after the final model is saved."""
    checkpoint_cfg = getattr(cfg, "checkpoint", None)
    if checkpoint_cfg is None or not checkpoint_cfg.get("cleanup", False):
        return

    dirpath = checkpoint_cfg.get("dirpath", cfg.output.dir)
    if not dirpath or not os.path.isdir(dirpath):
        return

    checkpoint_files = glob.glob(os.path.join(dirpath, "*.ckpt"))
    if not checkpoint_files:
        return

    removed = 0
    for ckpt_path in checkpoint_files:
        if not os.path.isfile(ckpt_path):
            continue
        try:
            os.remove(ckpt_path)
            removed += 1
        except OSError as exc:
            logger.warning("Failed to remove checkpoint %s: %s", ckpt_path, exc)

    if removed:
        logger.info("Removed %d checkpoint file(s) from %s", removed, dirpath)
