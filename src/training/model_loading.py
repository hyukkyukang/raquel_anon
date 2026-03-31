"""Shared model-loading helpers for training and evaluation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_tokenizer(
    model_path: str,
    *,
    base_model_name_for_adapters: Optional[str] = None,
) -> PreTrainedTokenizer:
    """Load tokenizer, falling back to the base model for adapter-only dirs."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        if not base_model_name_for_adapters:
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_for_adapters)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_quantization_config(use_fp16: bool) -> Optional[BitsAndBytesConfig]:
    """Create a 4-bit quantization config when requested."""
    compute_dtype: torch.dtype = torch.float16
    if (not use_fp16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _resolve_weight_dtype(use_fp16: bool, device_map: Any) -> str:
    """Avoid loading float16 weights onto CPU-only placements."""
    if not use_fp16:
        return "auto"
    if device_map is None:
        return "auto"
    if isinstance(device_map, str) and device_map.startswith("cpu"):
        return "auto"
    return "float16"


def _mark_adapter_parameters_trainable(model: PreTrainedModel) -> None:
    """Enable gradients for LoRA parameters on older PEFT versions."""
    for name, param in model.named_parameters():
        if "lora_" in name or "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


def load_model_and_tokenizer(
    model_path: str,
    device_map: Any = "auto",
    use_fp16: bool = True,
    ddp_mode: bool = False,
    *,
    quantize_4bit: bool = False,
    lora: Optional[Dict[str, Any]] = None,
    base_model_name_for_adapters: Optional[str] = None,
    adapter_trainable: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from a local path or Hugging Face model ID.

    Args:
        model_path: Path to model or Hugging Face model ID.
        device_map: Device mapping strategy ("auto", "cuda", "cpu", etc.).
        use_fp16: Whether to prefer FP16 when loading GPU-placed full-precision weights.
        ddp_mode: If True, load the model on CPU for Lightning DDP wrapping.
        quantize_4bit: Whether to use 4-bit loading.
        lora: Optional LoRA config used when wrapping a full model for training.
        base_model_name_for_adapters: Base model to use for adapter-only checkpoints.
        adapter_trainable: Whether adapter-only checkpoints should be loaded in trainable mode.
    """
    logger.info(
        "Loading model from %s (ddp_mode=%s, quantize_4bit=%s, lora=%s, adapter_trainable=%s)",
        model_path,
        ddp_mode,
        quantize_4bit,
        bool(lora and bool(lora.get("enabled", False))),
        adapter_trainable,
    )

    tokenizer = _load_tokenizer(
        model_path,
        base_model_name_for_adapters=base_model_name_for_adapters,
    )
    quantization_config = (
        _build_quantization_config(use_fp16) if quantize_4bit else None
    )
    model_dtype = _resolve_weight_dtype(use_fp16, device_map)
    training_mode = adapter_trainable or bool(lora and bool(lora.get("enabled", False)))

    if os.path.isdir(model_path):
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        has_adapter = os.path.exists(adapter_config_path) or any(
            os.path.exists(os.path.join(model_path, fname))
            for fname in ["adapter_model.bin", "adapter_model.safetensors"]
        )
        has_full_weights = any(
            os.path.exists(os.path.join(model_path, fname))
            for fname in [
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.safetensors",
            ]
        )

        if has_adapter and not has_full_weights:
            if ddp_mode:
                raise ValueError(
                    "ddp_mode=True is not supported for adapter-only checkpoints. "
                    "Disable ddp_mode or provide a full HF model directory."
                )

            base_model_name = base_model_name_for_adapters
            if base_model_name is None and os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, "r", encoding="utf-8") as handle:
                        adapter_cfg: Dict[str, Any] = json.load(handle)
                    base_model_name = str(
                        adapter_cfg.get("base_model_name_or_path", "")
                    ).strip() or None
                except Exception:
                    base_model_name = None

            if not base_model_name:
                raise ValueError(
                    "Detected adapter-only checkpoint at "
                    f"{model_path!r} but no base model was provided. "
                    "Pass base_model_name_for_adapters=... (e.g., cfg.model.name)."
                )

            logger.info(
                "Detected adapter-only checkpoint; loading base model=%s and attaching adapters from %s",
                base_model_name,
                model_path,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=device_map,
                dtype=model_dtype,
                quantization_config=quantization_config,
            )

            if quantize_4bit and adapter_trainable:
                base_model = prepare_model_for_kbit_training(base_model)

            try:
                model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    is_trainable=adapter_trainable,
                )
            except TypeError:
                model = PeftModel.from_pretrained(base_model, model_path)
                if adapter_trainable:
                    _mark_adapter_parameters_trainable(model)

            if training_mode and hasattr(model, "config"):
                model.config.use_cache = False

            if (
                training_mode
                and bool(lora and bool(lora.get("gradient_checkpointing", False)))
                and hasattr(model, "gradient_checkpointing_enable")
            ):
                model.gradient_checkpointing_enable()

            logger.info("Model loaded successfully (adapter-only checkpoint attached)")
            return model, tokenizer

    if ddp_mode:
        if quantize_4bit:
            raise ValueError(
                "quantize_4bit=True is not supported with ddp_mode=True in this codebase "
                "(4-bit loading expects a device map / GPU placement)."
            )
        logger.info("DDP mode enabled: loading model on CPU for Lightning DDP wrapping")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=model_dtype,
            quantization_config=quantization_config,
        )

    if training_mode and hasattr(model, "config"):
        model.config.use_cache = False

    if lora and bool(lora.get("enabled", False)):
        if quantize_4bit:
            model = prepare_model_for_kbit_training(model)

        if bool(lora.get("gradient_checkpointing", True)) and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()

        target_modules = lora.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(target_modules),
            bias=str(lora.get("bias", "none")),
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    logger.info("Model loaded successfully")
    return model, tokenizer
