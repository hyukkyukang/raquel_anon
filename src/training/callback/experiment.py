"""Experiment tracking callbacks for Lightning training runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch

from src.training.experiment import ExperimentTracker
from src.utils.logging import get_logger

logger = get_logger("src.training.callback.experiment")


class ExperimentResultsCallback(Callback):
    """Persist metrics and checkpoint status for experiment tracking."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self._tracker: ExperimentTracker = tracker

    def _extract_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert Lightning metrics to JSON-serializable scalars."""
        cleaned: Dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                # Keep only scalar tensors to avoid large or non-serializable payloads.
                if value.numel() != 1:
                    continue
                cleaned[key] = float(value.item())
                continue
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
                continue
            if hasattr(value, "item"):
                try:
                    # Some metric objects expose .item(); ignore failures gracefully.
                    cleaned[key] = float(value.item())
                except Exception:
                    continue
        return cleaned

    def _current_checkpoint(self, trainer: pl.Trainer) -> Optional[str]:
        """Resolve the latest checkpoint path from the trainer."""
        checkpoint_callback: Optional[Any] = getattr(
            trainer, "checkpoint_callback", None
        )
        if checkpoint_callback is None:
            return None
        last_path: Optional[str] = getattr(checkpoint_callback, "last_model_path", None)
        if last_path:
            return last_path
        return getattr(checkpoint_callback, "best_model_path", None)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record training metrics at epoch end."""
        if not self._tracker.enabled:
            return
        metrics: Dict[str, float] = self._extract_metrics(trainer.callback_metrics)
        payload: Dict[str, Any] = {
            # Persist stage metadata alongside metrics for later filtering.
            "stage": "train",
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "metrics": metrics,
        }
        self._tracker.log_metrics(payload)
        self._tracker.update_status(
            last_checkpoint=self._current_checkpoint(trainer),
            last_epoch=trainer.current_epoch,
            last_global_step=trainer.global_step,
        )


class TimeEstimationCallback(Callback):
    """Estimate and log training runtime based on observed step times."""

    def __init__(
        self,
        *,
        tracker: Optional[ExperimentTracker] = None,
        log_every_n_steps: int = 50,
        warmup_steps: int = 5,
    ) -> None:
        self._tracker: Optional[ExperimentTracker] = tracker
        self._log_every_n_steps: int = max(int(log_every_n_steps), 1)
        self._warmup_steps: int = max(int(warmup_steps), 0)
        self._start_time: Optional[float] = None
        self._total_steps: Optional[int] = None
        self._last_logged_step: int = -1

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format seconds into H:MM:SS string."""
        if seconds < 0:
            seconds = 0.0
        total_seconds: int = int(seconds)
        hours: int = total_seconds // 3600
        minutes: int = (total_seconds % 3600) // 60
        secs: int = total_seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _resolve_total_steps(trainer: pl.Trainer) -> Optional[int]:
        """Resolve the total number of training steps if available."""
        estimated: Optional[int] = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, int) and estimated > 0:
            return estimated
        max_steps: Optional[int] = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return max_steps
        return None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture start time and total steps."""
        self._start_time = time.perf_counter()
        self._total_steps = self._resolve_total_steps(trainer)
        if self._total_steps:
            logger.info("Estimated total steps: %d", self._total_steps)
        else:
            logger.info("Estimated total steps unavailable; ETA will be skipped.")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log ETA estimates periodically based on observed step times."""
        if self._start_time is None:
            return
        if self._total_steps is None:
            return

        step: int = int(trainer.global_step)
        if step < self._warmup_steps:
            return
        if step == self._last_logged_step:
            return
        if step % self._log_every_n_steps != 0 and step != self._total_steps:
            return

        elapsed: float = time.perf_counter() - self._start_time
        if step <= 0:
            return
        avg_step: float = elapsed / float(step)
        remaining_steps: int = max(self._total_steps - step, 0)
        eta_seconds: float = avg_step * remaining_steps
        total_estimated: float = avg_step * float(self._total_steps)

        payload: Dict[str, Any] = {
            "step": step,
            "total_steps": self._total_steps,
            "elapsed_seconds": elapsed,
            "avg_step_seconds": avg_step,
            "eta_seconds": eta_seconds,
            "estimated_total_seconds": total_estimated,
            "eta_human": self._format_seconds(eta_seconds),
            "estimated_total_human": self._format_seconds(total_estimated),
        }
        logger.info(
            "ETA: %s (elapsed %s, step %d/%d)",
            payload["eta_human"],
            self._format_seconds(elapsed),
            step,
            self._total_steps,
        )

        if self._tracker and self._tracker.enabled:
            self._tracker.log_event("time_estimate", payload)

        self._last_logged_step = step

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log final runtime summary."""
        if self._start_time is None:
            return
        elapsed: float = time.perf_counter() - self._start_time
        payload: Dict[str, Any] = {
            "elapsed_seconds": elapsed,
            "elapsed_human": self._format_seconds(elapsed),
            "total_steps": self._total_steps,
        }
        logger.info("Training runtime: %s", payload["elapsed_human"])
        if self._tracker and self._tracker.enabled:
            self._tracker.log_event("time_summary", payload)

