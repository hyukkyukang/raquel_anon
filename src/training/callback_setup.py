"""Callback assembly for training runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from src.training.callback import (
    CustomProgressBar,
    MUSEEvaluationCallback,
    RAQUELEvaluationCallback,
    TimeEstimationCallback,
)
from src.training.experiment import ExperimentTracker
from src.training.task import TASK_UNLEARN
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_project_path(cfg: DictConfig) -> str:
    """Resolve the project root used for relative callback dataset paths."""
    return cfg.get("project_path") or str(Path(__file__).resolve().parents[2])


def resolve_relative_path(path: str, *, project_root: str) -> str:
    """Resolve a path relative to the project root when needed."""
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def _resolve_existing_path(
    path: str,
    *,
    project_root: str,
    label: str,
) -> Optional[str]:
    """Resolve a dataset path and warn early if it does not exist."""
    resolved_path = resolve_relative_path(path, project_root=project_root)
    if not os.path.exists(resolved_path):
        logger.warning("%s file not found: %s", label, resolved_path)
        return None
    return resolved_path


def _create_checkpoint_callback(cfg: DictConfig) -> Optional[ModelCheckpoint]:
    checkpoint_cfg = getattr(cfg, "checkpoint", None)
    if checkpoint_cfg is None:
        return None
    return ModelCheckpoint(
        dirpath=checkpoint_cfg.get("dirpath", cfg.output.dir),
        filename=checkpoint_cfg.get("filename", "checkpoint-{epoch:02d}"),
        save_top_k=checkpoint_cfg.get("save_top_k", 1),
        monitor=checkpoint_cfg.get("monitor", "train_loss"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_last=checkpoint_cfg.get("save_last", True),
    )


def _create_time_callback(
    cfg: DictConfig,
    tracker: Optional[ExperimentTracker],
) -> Optional[TimeEstimationCallback]:
    time_cfg: Dict[str, Any] = cfg.get("time_estimation", {}) if hasattr(cfg, "get") else {}
    if not time_cfg.get("enabled", True):
        return None
    return TimeEstimationCallback(
        tracker=tracker,
        log_every_n_steps=time_cfg.get("log_every_n_steps", 50),
        warmup_steps=time_cfg.get("warmup_steps", 5),
    )


def _create_muse_callback(cfg: DictConfig, *, project_root: str) -> Optional[MUSEEvaluationCallback]:
    if cfg.task != TASK_UNLEARN or not cfg.evaluation.get("run_muse", False):
        return None

    logger.info("Setting up MUSE evaluation callback")
    forget_file = cfg.data.get("forget_file")
    retain_file = cfg.data.get("retain_file")
    if not forget_file or not retain_file:
        logger.warning(
            "MUSE evaluation enabled but forget_file/retain_file are not configured."
        )
        return None

    forget_path = _resolve_existing_path(
        forget_file,
        project_root=project_root,
        label="MUSE forget dataset",
    )
    retain_path = _resolve_existing_path(
        retain_file,
        project_root=project_root,
        label="MUSE retain dataset",
    )
    if not forget_path or not retain_path:
        return None

    paraphrased_path = None
    paraphrased_file = cfg.data.get("paraphrased_file")
    if paraphrased_file:
        paraphrased_path = _resolve_existing_path(
            paraphrased_file,
            project_root=project_root,
            label="MUSE paraphrased dataset",
        )

    non_training_path = None
    non_training_file = cfg.data.get("non_training_file")
    if non_training_file:
        non_training_path = _resolve_existing_path(
            non_training_file,
            project_root=project_root,
            label="MUSE non-training dataset",
        )

    return MUSEEvaluationCallback(
        forget_path=forget_path,
        retain_path=retain_path,
        paraphrased_path=paraphrased_path,
        non_training_path=non_training_path,
        output_dir=os.path.join(cfg.output.dir, "muse_results"),
        run_on_train_end=cfg.evaluation.get("muse_on_train_end", True),
        run_on_epoch_end=cfg.evaluation.get("muse_on_epoch_end", False),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _create_raquel_callback(
    cfg: DictConfig,
    *,
    project_root: str,
) -> Optional[RAQUELEvaluationCallback]:
    raquel_cfg = cfg.evaluation.get("raquel", {}) if hasattr(cfg, "evaluation") else {}
    if cfg.task != TASK_UNLEARN or not raquel_cfg.get("enabled", False):
        return None

    affected_path = raquel_cfg.get("affected_file")
    unaffected_path = raquel_cfg.get("unaffected_file")
    if not affected_path or not unaffected_path:
        logger.warning("RAQUEL evaluation enabled but affected/unaffected files not set.")
        return None

    resolved_affected_path = _resolve_existing_path(
        affected_path,
        project_root=project_root,
        label="RAQUEL affected dataset",
    )
    resolved_unaffected_path = _resolve_existing_path(
        unaffected_path,
        project_root=project_root,
        label="RAQUEL unaffected dataset",
    )
    if not resolved_affected_path or not resolved_unaffected_path:
        return None

    output_dir = raquel_cfg.get("output_dir") or os.path.join(
        cfg.output.dir, "raquel_eval"
    )
    return RAQUELEvaluationCallback(
        affected_path=resolved_affected_path,
        unaffected_path=resolved_unaffected_path,
        output_dir=output_dir,
        batch_size=raquel_cfg.get("batch_size", 8),
        max_new_tokens=raquel_cfg.get("max_new_tokens", 64),
        max_prompt_length=raquel_cfg.get("max_prompt_length"),
        max_examples=raquel_cfg.get("max_examples"),
        run_on_train_end=raquel_cfg.get("run_on_train_end", True),
        run_on_epoch_end=raquel_cfg.get("run_on_epoch_end", False),
        device=raquel_cfg.get("device"),
        semantic_cfg=cfg.evaluation.get("semantic_equivalence", {}),
        save_predictions=raquel_cfg.get("save_predictions", False),
    )


def create_callbacks(
    cfg: DictConfig,
    tracker: Optional[ExperimentTracker] = None,
) -> List[object]:
    """Create the full callback list for the training run."""
    callbacks: List[object] = []
    project_root = resolve_project_path(cfg)

    checkpoint_callback = _create_checkpoint_callback(cfg)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    if cfg.trainer.get("enable_progress_bar", True):
        callbacks.append(CustomProgressBar())

    time_callback = _create_time_callback(cfg, tracker)
    if time_callback is not None:
        callbacks.append(time_callback)

    muse_callback = _create_muse_callback(cfg, project_root=project_root)
    if muse_callback is not None:
        callbacks.append(muse_callback)

    raquel_callback = _create_raquel_callback(cfg, project_root=project_root)
    if raquel_callback is not None:
        callbacks.append(raquel_callback)

    return callbacks
