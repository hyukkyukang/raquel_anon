"""Shared helpers for deterministic text generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


_GREEDY_DEFAULTS: Dict[str, Any] = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "typical_p": 1.0,
    "top_k": 50,
    "penalty_alpha": None,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
}


def build_greedy_generation_kwargs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    input_length: int,
    max_new_tokens: int,
    min_new_tokens: Optional[int] = None,
    use_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return warning-free generation kwargs for deterministic decoding."""
    generation_config = deepcopy(getattr(model, "generation_config", None))
    if generation_config is None:
        generation_config = GenerationConfig()

    generation_config.max_length = max(int(input_length), 0) + max(int(max_new_tokens), 1)
    generation_config.min_length = None
    generation_config.max_new_tokens = None
    generation_config.min_new_tokens = None

    for field_name, default_value in _GREEDY_DEFAULTS.items():
        if hasattr(generation_config, field_name):
            setattr(generation_config, field_name, default_value)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generation_config.pad_token_id = pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if min_new_tokens is not None:
        generation_config.min_new_tokens = max(int(min_new_tokens), 0)
    if use_cache is not None:
        generation_config.use_cache = use_cache
    return {"generation_config": generation_config}
