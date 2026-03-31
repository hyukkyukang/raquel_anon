"""Generic external logger helpers for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_external_logger_provider(cfg: DictConfig) -> str:
    """Resolve the configured external tracking backend."""
    tracking_cfg = cfg.get("tracking") if hasattr(cfg, "get") else None
    if tracking_cfg is not None:
        if not bool(tracking_cfg.get("enabled", True)):
            return "none"
        return str(tracking_cfg.get("provider", "none")).strip().lower() or "none"

    mlflow_cfg = cfg.get("mlflow") if hasattr(cfg, "get") else None
    if mlflow_cfg is not None and bool(mlflow_cfg.get("enabled", True)):
        return "mlflow"
    return "none"


def create_external_logger(cfg: DictConfig) -> Optional[Any]:
    """Create the configured external logger instance, if any."""
    provider = resolve_external_logger_provider(cfg)
    if provider == "none":
        return None
    if provider == "mlflow":
        from src.training.mlflow import create_mlflow_logger

        return create_mlflow_logger(cfg)
    raise ValueError(f"Unsupported external tracking provider: {provider}")


def finalize_external_logger(logger_instance: Optional[Any]) -> None:
    """Finalize an external logger when the backend exposes shutdown hooks."""
    if logger_instance in (None, False):
        return

    finalize = getattr(logger_instance, "finalize", None)
    if callable(finalize):
        try:
            finalize("success")
        except Exception as exc:
            logger.warning("Failed to finalize external logger cleanly: %s", exc)

    experiment = getattr(logger_instance, "experiment", None)
    stop = getattr(experiment, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception as exc:
            logger.warning("Failed to stop external logger experiment cleanly: %s", exc)


def _log_artifact_path(
    logger_instance: Any,
    local_path: Path,
    *,
    artifact_path: str,
) -> None:
    """Upload a file or directory artifact when the logger exposes MLflow-style hooks."""
    if not local_path.exists():
        return

    experiment = getattr(logger_instance, "experiment", None)
    run_id = getattr(logger_instance, "run_id", None)
    if experiment is None or not run_id:
        return

    if local_path.is_dir():
        log_artifacts = getattr(experiment, "log_artifacts", None)
        if callable(log_artifacts):
            log_artifacts(run_id, str(local_path), artifact_path)
            return

    log_artifact = getattr(experiment, "log_artifact", None)
    if callable(log_artifact):
        log_artifact(run_id, str(local_path), artifact_path)


def log_external_artifacts(
    logger_instance: Optional[Any],
    cfg: DictConfig,
    *,
    run_dir: Optional[str] = None,
) -> None:
    """Log local training artifacts to the configured external backend."""
    if logger_instance in (None, False):
        return
    if resolve_external_logger_provider(cfg) != "mlflow":
        return

    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}
    if bool(mlflow_cfg.get("log_generation_artifacts", True)):
        _log_artifact_path(
            logger_instance,
            Path(str(cfg.log_dir)) / "artifacts",
            artifact_path="qualitative",
        )

    if bool(mlflow_cfg.get("log_model_artifacts", False)):
        _log_artifact_path(
            logger_instance,
            Path(str(cfg.output.dir)),
            artifact_path="model",
        )

    if run_dir and bool(mlflow_cfg.get("log_run_artifacts", True)):
        _log_artifact_path(
            logger_instance,
            Path(run_dir),
            artifact_path="run",
        )
