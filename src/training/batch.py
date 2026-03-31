"""Helpers for working with collated training batches."""

from typing import Any, Dict

import torch


MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "labels")


def extract_model_inputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Filter a collated batch down to tensors accepted by model forward."""
    model_inputs: Dict[str, torch.Tensor] = {}

    for key in MODEL_INPUT_KEYS:
        if key not in batch:
            raise KeyError(f"Batch is missing required key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Batch key '{key}' must be a torch.Tensor, got {type(value).__name__}"
            )
        model_inputs[key] = value

    return model_inputs
