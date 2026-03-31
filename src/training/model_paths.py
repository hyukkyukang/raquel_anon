"""Lightweight path and identifier helpers for training artifacts."""

from __future__ import annotations

import os
from typing import Dict

_KNOWN_MODEL_DIR_MAP: Dict[str, str] = {
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
    "meta-llama/Llama-3.2-3B": "llama-3.2-3b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
}


def get_base_model_dir_component(model_name: str) -> str:
    """
    Map the base model name to a short directory-friendly component.

    Args:
        model_name: The Hugging Face model identifier.

    Returns:
        Filesystem-friendly component used under `model/`.
    """
    if model_name in _KNOWN_MODEL_DIR_MAP:
        return _KNOWN_MODEL_DIR_MAP[model_name]

    sanitized = "".join(ch if ch.isalnum() else "-" for ch in model_name.lower())
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-")


def check_model_exists(output_dir: str) -> bool:
    """
    Check if a trained model already exists in the output directory.

    Args:
        output_dir: Directory to inspect.

    Returns:
        True if full-model weights or PEFT adapter artifacts are present.
    """
    if not os.path.isdir(output_dir):
        return False

    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.safetensors",
    ]
    adapter_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]

    return any(
        os.path.exists(os.path.join(output_dir, fname))
        for fname in (model_files + adapter_files)
    )
