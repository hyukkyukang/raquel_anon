"""MLflow-specific training logger helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from src.training.task import TASK_FINETUNE, TASK_UNLEARN
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_local_mlflow_paths(cfg: DictConfig) -> tuple[str, str]:
    project_path = cfg.get("project_path") or str(Path.cwd())
    project_root = Path(project_path).resolve()
    tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
    artifact_root = str((project_root / "mlartifacts").resolve())
    return tracking_uri, artifact_root


def _resolve_mlflow_connection(cfg: DictConfig) -> tuple[Optional[str], Optional[str]]:
    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri")
    artifact_root = mlflow_cfg.get("artifact_root")
    if tracking_uri:
        return str(tracking_uri), (str(artifact_root) if artifact_root else None)
    return _resolve_local_mlflow_paths(cfg)


def _ensure_experiment(
    *,
    tracking_uri: Optional[str],
    experiment_name: str,
    artifact_root: Optional[str],
) -> None:
    """Create the target experiment with the configured artifact location."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            experiment_name,
            artifact_location=str(Path(artifact_root).resolve()) if artifact_root else None,
        )
        return

    if artifact_root:
        expected_location = str(Path(artifact_root).resolve())
        actual_location = str(experiment.artifact_location)
        if actual_location != expected_location:
            logger.warning(
                "MLflow experiment %s already exists with artifact_location=%s; "
                "configured artifact_root=%s will not override it.",
                experiment_name,
                actual_location,
                expected_location,
            )


def _target_dataset_name(cfg: DictConfig) -> str:
    if cfg.task == TASK_FINETUNE:
        return (
            f"{cfg.data.name}:{cfg.data.train_split}:{cfg.data.train_subset_name}".strip()
        )
    if cfg.task == TASK_UNLEARN:
        return (
            f"{cfg.data.name}:{cfg.data.forget_split}:{cfg.data.forget_subset_name}".strip()
        )
    raise ValueError(f"Unknown task: {cfg.task}")


def _build_mlflow_tags(cfg: DictConfig) -> Dict[str, Any]:
    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}
    tags: Dict[str, Any] = {
        "task": str(cfg.task).strip(),
        "tag": str(cfg.tag).strip(),
        "model": str(cfg.model.name).strip(),
        "dataset": _target_dataset_name(cfg),
        "learning_rate": str(cfg.training.learning_rate),
    }
    custom_tags = mlflow_cfg.get("tags", {})
    if isinstance(custom_tags, dict):
        for key, value in custom_tags.items():
            tags[str(key)] = value
    return tags


def create_mlflow_logger(cfg: DictConfig) -> MLFlowLogger:
    """Initialize an MLflow logger when tracking is enabled."""
    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}
    tracking_uri, artifact_root = _resolve_mlflow_connection(cfg)
    experiment_name = str(mlflow_cfg.get("experiment_name", "RAQUEL")).strip()
    run_name = (
        mlflow_cfg.get("run_name")
        or cfg.get("results", {}).get("run_name")
        or str(cfg.task).strip()
    )
    _ensure_experiment(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_root=artifact_root,
    )

    logger_kwargs: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "run_name": str(run_name).strip(),
        "tracking_uri": tracking_uri,
        "save_dir": artifact_root,
        "tags": _build_mlflow_tags(cfg),
        "log_model": bool(mlflow_cfg.get("log_model_artifacts", False)),
        "prefix": None,
        "artifact_location": artifact_root,
    }
    mlflow_logger = MLFlowLogger(**logger_kwargs)

    hyperparams = {
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.train_batch_size,
        "effective_batch_size": cfg.training.train_batch_size
        * cfg.training.gradient_accumulation_steps,
        "grad_accum": cfg.training.gradient_accumulation_steps,
        "lr": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "warmup": cfg.training.warmup_ratio,
        "grad_norm": cfg.training.max_grad_norm,
    }
    mlflow_logger.log_hyperparams(hyperparams)
    logger.info(
        "Initialized MLflow logger for experiment=%s run_name=%s",
        experiment_name,
        run_name,
    )
    return mlflow_logger
