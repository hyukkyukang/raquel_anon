"""Compatibility helpers for evaluation scripts."""

from typing import Any, Tuple

from src.training.model_paths import get_base_model_dir_component


def load_model_and_tokenizer(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """Lazily import the shared loader to keep this wrapper lightweight."""
    from src.training.model_loading import load_model_and_tokenizer as shared_loader

    return shared_loader(*args, **kwargs)


def load_fine_tuned_model(
    model_identifier: str,
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    device_map_auto: bool = True,
    quantize_4bit: bool = True,
    as_trainable: bool = False,
) -> Tuple[Any, Any]:
    """
    Load a tokenizer and model for either:
    - A local fine-tuned directory containing LoRA adapters (attach to base model)
    - A local directory containing full model weights (load directly)
    - A remote/base HF model name (load directly)

    Args:
        model_identifier (str): Local directory path or HF model name.
        base_model_name (str): Name of the base model to use when attaching LoRA adapters.
        device_map_auto (bool): If True, use automatic device mapping; else no device map.
        quantize_4bit (bool): If True, enable 4-bit quantization for efficiency.
        as_trainable (bool): If True and loading LoRA adapters, mark adapter params trainable.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer ready for use.
    """
    return load_model_and_tokenizer(
        model_identifier,
        device_map="auto" if device_map_auto else None,
        use_fp16=bool(device_map_auto),
        ddp_mode=False,
        quantize_4bit=quantize_4bit,
        lora=None,
        base_model_name_for_adapters=base_model_name,
        adapter_trainable=as_trainable,
    )
