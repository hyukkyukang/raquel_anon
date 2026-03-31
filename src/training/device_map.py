"""Helpers for resolving model device maps for Lightning-managed training."""

from __future__ import annotations

from typing import Any, Optional


def _count_requested_devices(devices: Any) -> Optional[int]:
    """Best-effort count of explicitly requested trainer devices."""
    if devices is None:
        return None

    if isinstance(devices, int):
        return max(devices, 0)

    if isinstance(devices, (list, tuple, set)):
        return len(devices)

    if isinstance(devices, str):
        normalized = devices.strip().lower()
        if not normalized or normalized == "auto":
            return None
        if normalized.isdigit():
            return int(normalized)

        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if parts and all(part.isdigit() for part in parts):
            return len(parts)

    return None


def resolve_training_device_map(
    *,
    requested_device_map: Any,
    accelerator: Any,
    devices: Any,
    cuda_available: bool,
    cuda_device_count: int,
) -> Any:
    """Resolve `model.device_map` for Lightning-managed training runs.

    Hugging Face `device_map="auto"` can shard a model across multiple visible GPUs.
    That conflicts with single-device Lightning runs, where the trainer expects the model
    to live on one process-local device. This helper pins auto-placement to the trainer's
    effective single device when we can infer one safely.
    """
    if requested_device_map != "auto":
        return requested_device_map

    accelerator_name = str(accelerator or "auto").strip().lower()
    requested_device_count = _count_requested_devices(devices)

    if accelerator_name == "cpu":
        return "cpu"

    if accelerator_name in {"gpu", "cuda"}:
        if requested_device_count == 1:
            return "cuda:0"
        if requested_device_count is None and cuda_available and cuda_device_count == 1:
            return "cuda:0"
        return "auto"

    if accelerator_name == "auto":
        if not cuda_available:
            return "cpu"
        if requested_device_count == 1:
            return "cuda:0"
        if requested_device_count is None and cuda_device_count == 1:
            return "cuda:0"
        return "auto"

    return requested_device_map
