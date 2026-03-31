"""Filesystem experiment tracking utilities for training runs."""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_logger
from src.utils.string import sanitize_identifier

logger = get_logger("src.training.experiment")


@dataclass
class ExperimentPaths:
    """Resolved file paths for a tracked experiment run."""

    run_json: str
    status_json: str
    config_yaml: str
    config_json: str
    metrics_jsonl: str
    events_jsonl: str
    summary_json: str


class ExperimentTracker:
    """Track training/unlearning experiments to the filesystem."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        run_id: str,
        run_dir: str,
        enabled: bool,
        save_config: bool,
        save_metrics: bool,
        save_events: bool,
    ) -> None:
        self._cfg: DictConfig = cfg
        self._run_id: str = run_id
        self._run_dir: str = run_dir
        self._enabled: bool = enabled
        self._save_config: bool = save_config
        self._save_metrics: bool = save_metrics
        self._save_events: bool = save_events
        self._lock: threading.Lock = threading.Lock()
        self._started_at: Optional[str] = None
        self._paths: ExperimentPaths = ExperimentPaths(
            run_json=os.path.join(run_dir, "run.json"),
            status_json=os.path.join(run_dir, "status.json"),
            config_yaml=os.path.join(run_dir, "config.yaml"),
            config_json=os.path.join(run_dir, "config.json"),
            metrics_jsonl=os.path.join(run_dir, "metrics.jsonl"),
            events_jsonl=os.path.join(run_dir, "events.jsonl"),
            summary_json=os.path.join(run_dir, "summary.json"),
        )

    @property
    def enabled(self) -> bool:
        """Return whether tracking is enabled."""
        return self._enabled

    @property
    def run_id(self) -> str:
        """Return the run identifier."""
        return self._run_id

    @property
    def run_dir(self) -> str:
        """Return the directory where run artifacts are stored."""
        return self._run_dir

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        """Write a JSON file atomically for run metadata."""
        if not self._enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            # Overwrite atomically to keep a single source of truth per file.
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)

    def _append_jsonl(self, path: str, payload: Dict[str, Any]) -> None:
        """Append a JSONL record for incremental logging."""
        if not self._enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            # Append line-delimited JSON to support streaming ingestion.
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str))
                handle.write("\n")

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
        """Read config values from DictConfig/dict/namespace."""
        if cfg is None:
            return default
        if isinstance(cfg, DictConfig):
            return cfg.get(key, default)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, default)
            except Exception:
                return default
        return getattr(cfg, key, default)

    @staticmethod
    def _resolve_project_path(cfg: DictConfig) -> str:
        """Resolve the project root for relative paths."""
        project_path: str = ExperimentTracker._cfg_get(
            cfg, "project_path", str(Path.cwd())
        )
        return project_path

    @staticmethod
    def _resolve_git_revision(project_path: str) -> Optional[str]:
        """Resolve the current git commit without shelling out."""
        head_path: Path = Path(project_path) / ".git" / "HEAD"
        if not head_path.exists():
            return None
        head_value: str = head_path.read_text(encoding="utf-8").strip()
        if head_value.startswith("ref:"):
            ref_path: str = head_value.split(" ", 1)[-1]
            ref_file: Path = Path(project_path) / ".git" / ref_path
            if ref_file.exists():
                return ref_file.read_text(encoding="utf-8").strip()
            return None
        return head_value or None

    @staticmethod
    def _build_run_id(cfg: DictConfig) -> str:
        """Build a stable run identifier."""
        task_name_raw: str = str(ExperimentTracker._cfg_get(cfg, "task", "run"))
        model_name_raw: str = str(
            ExperimentTracker._cfg_get(
                ExperimentTracker._cfg_get(cfg, "model", {}),
                "name",
                "model",
            )
        )
        task_name: str = sanitize_identifier(task_name_raw)
        model_name: str = sanitize_identifier(model_name_raw.split("/")[-1])
        timestamp: str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_id: str = uuid.uuid4().hex[:8]
        return f"{timestamp}_{task_name}_{model_name}_{short_id}"

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ExperimentTracker":
        """Create a tracker from the experiment config."""
        results_cfg: Dict[str, Any] = cls._cfg_get(cfg, "results", {})
        enabled: bool = bool(cls._cfg_get(results_cfg, "enabled", True))
        save_config: bool = bool(cls._cfg_get(results_cfg, "save_config", True))
        save_metrics: bool = bool(cls._cfg_get(results_cfg, "save_metrics", True))
        save_events: bool = bool(cls._cfg_get(results_cfg, "save_events", True))

        project_path: str = cls._resolve_project_path(cfg)
        base_dir_raw: str = str(cls._cfg_get(results_cfg, "dir", "results"))
        base_dir: str = (
            base_dir_raw
            if os.path.isabs(base_dir_raw)
            else os.path.join(project_path, base_dir_raw)
        )

        run_id_raw: Optional[str] = cls._cfg_get(results_cfg, "run_id", None)
        run_name_raw: Optional[str] = cls._cfg_get(results_cfg, "run_name", None)
        run_id: str
        if run_id_raw:
            run_id = sanitize_identifier(str(run_id_raw))
        elif run_name_raw:
            run_id = sanitize_identifier(str(run_name_raw))
        else:
            run_id = cls._build_run_id(cfg)

        run_dir: str = os.path.join(base_dir, run_id)

        return cls(
            cfg=cfg,
            run_id=run_id,
            run_dir=run_dir,
            enabled=enabled,
            save_config=save_config,
            save_metrics=save_metrics,
            save_events=save_events,
        )

    def start(self) -> None:
        """Initialize the run metadata and status files."""
        if not self._enabled:
            return
        os.makedirs(self._run_dir, exist_ok=True)
        self._started_at = datetime.utcnow().isoformat()

        project_path: str = self._resolve_project_path(self._cfg)
        git_revision: Optional[str] = self._resolve_git_revision(project_path)
        command: str = " ".join(sys.argv)

        model_cfg: Dict[str, Any] = self._cfg_get(self._cfg, "model", {})
        output_cfg: Dict[str, Any] = self._cfg_get(self._cfg, "output", {})
        checkpoint_cfg: Dict[str, Any] = self._cfg_get(self._cfg, "checkpoint", {})
        trainer_cfg: Dict[str, Any] = self._cfg_get(self._cfg, "trainer", {})
        run_payload: Dict[str, Any] = {
            # Core metadata allows resuming and paper-ready aggregation.
            "run_id": self._run_id,
            "run_dir": self._run_dir,
            "task": self._cfg_get(self._cfg, "task", "run"),
            "tag": self._cfg_get(self._cfg, "tag", None),
            "model_name": self._cfg_get(model_cfg, "name", None),
            "output_dir": self._cfg_get(output_cfg, "dir", None),
            "checkpoint_dir": self._cfg_get(checkpoint_cfg, "dirpath", None),
            "resume_checkpoint": self._cfg_get(
                trainer_cfg, "resume_from_checkpoint", None
            ),
            "started_at": self._started_at,
            "host": platform.node(),
            "python_version": sys.version.split()[0],
            "git_revision": git_revision,
            "command": command,
        }
        self._write_json(self._paths.run_json, run_payload)

        status_payload: Dict[str, Any] = {
            "run_id": self._run_id,
            "status": "running",
            "started_at": self._started_at,
            "last_updated_at": self._started_at,
            "last_checkpoint": None,
            "last_epoch": None,
            "last_global_step": None,
        }
        self._write_json(self._paths.status_json, status_payload)

        if self._save_config:
            # Persist a fully resolved config snapshot for reproducibility.
            config_yaml: str = OmegaConf.to_yaml(self._cfg, resolve=True)
            with self._lock:
                with open(self._paths.config_yaml, "w", encoding="utf-8") as handle:
                    handle.write(config_yaml)
            config_json_payload: Dict[str, Any] = OmegaConf.to_container(
                self._cfg, resolve=True
            )
            self._write_json(self._paths.config_json, config_json_payload)

        if self._save_events:
            self.log_event("run_started", {"started_at": self._started_at})

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        """Append a metrics record for the current run."""
        if not self._save_metrics:
            return
        payload_with_meta = {
            **payload,
            "run_id": self._run_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_jsonl(self._paths.metrics_jsonl, payload_with_meta)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Append an event record to the run log."""
        if not self._save_events:
            return
        event_payload: Dict[str, Any] = {
            "run_id": self._run_id,
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }
        self._append_jsonl(self._paths.events_jsonl, event_payload)

    def update_status(
        self,
        *,
        status: Optional[str] = None,
        last_checkpoint: Optional[str] = None,
        last_epoch: Optional[int] = None,
        last_global_step: Optional[int] = None,
    ) -> None:
        """Update the status file with the latest progress."""
        if not self._enabled:
            return
        status_payload: Dict[str, Any] = {
            "run_id": self._run_id,
            "status": status or "running",
            "last_checkpoint": last_checkpoint,
            "last_epoch": last_epoch,
            "last_global_step": last_global_step,
            "last_updated_at": datetime.utcnow().isoformat(),
            "started_at": self._started_at,
        }
        self._write_json(self._paths.status_json, status_payload)

    def mark_completed(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Mark the run as completed and optionally write summary results."""
        self.update_status(status="completed")
        if summary is not None:
            self._write_json(self._paths.summary_json, summary)
        if self._save_events:
            self.log_event("run_completed", {"summary_written": summary is not None})

    def mark_failed(self, error: Exception) -> None:
        """Mark the run as failed and record the error details."""
        payload: Dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self.update_status(status="failed")
        if self._save_events:
            self.log_event("run_failed", payload)
