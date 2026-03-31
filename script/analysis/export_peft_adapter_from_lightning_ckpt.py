"""Export a PEFT (LoRA) adapter from a Lightning `.ckpt` checkpoint.

This is a practical utility for the case where a training run successfully wrote
Lightning checkpoints (e.g., `last.ckpt`) but did not finish writing the final PEFT
adapter artifacts (`adapter_model.*`, `adapter_config.json`).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch

from src.training.utils import load_model_and_tokenizer
from src.utils.logging import get_logger

logger = get_logger("script.analysis.export_peft_adapter_from_lightning_ckpt")


def _load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load a Lightning checkpoint and return its `state_dict`."""
    # Torch 2.6+ defaults to `weights_only=True`; Lightning checkpoints need full load.
    try:
        payload: Dict[str, Any] = torch.load(  # type: ignore[call-arg]
            ckpt_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")  # type: ignore[assignment]

    state_dict_any: Any = payload.get("state_dict", payload)
    if not isinstance(state_dict_any, dict):
        raise ValueError(
            f"Unexpected checkpoint format at {ckpt_path!r}: missing state_dict"
        )

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict_any.items():
        if isinstance(value, torch.Tensor):
            state_dict[str(key)] = value
    return state_dict


def _extract_lora_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from a Lightning module `state_dict`."""
    # Lightning saves module params under `model.*` where `model` is BaseLightningModule.model.
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        stripped: str = key[len("model.") :]
        # Keep only LoRA weights; avoids loading base-model weights (large / quantized).
        if "lora_" not in stripped:
            continue
        out[stripped] = value
    return out


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export PEFT adapter artifacts from a Lightning checkpoint."
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument(
        "--adapter_init_dir",
        required=True,
        help=(
            "Path to an adapter-only directory used to instantiate the LoRA modules "
            "(e.g., model/full_8b_s0/finetune/meta-llama/Llama-3.1-8B)."
        ),
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model HF id/path (e.g., meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to write adapter_model.* + adapter_config.json + tokenizer files.",
    )
    parser.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Load the base model in 4-bit to reduce GPU memory.",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = _parse_args()

    ckpt_path: str = os.path.expanduser(str(args.ckpt_path))
    adapter_init_dir: str = os.path.expanduser(str(args.adapter_init_dir))
    base_model: str = str(args.base_model).strip()
    output_dir: str = os.path.expanduser(str(args.output_dir))
    quantize_4bit: bool = bool(args.quantize_4bit)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Instantiating PEFT model from %s", adapter_init_dir)
    model, tokenizer = load_model_and_tokenizer(
        adapter_init_dir,
        device_map="auto",
        use_fp16=True,
        ddp_mode=False,
        quantize_4bit=quantize_4bit,
        lora=None,
        base_model_name_for_adapters=base_model,
    )

    logger.info("Loading Lightning checkpoint from %s", ckpt_path)
    state_dict: Dict[str, torch.Tensor] = _load_checkpoint_state_dict(ckpt_path)
    lora_state: Dict[str, torch.Tensor] = _extract_lora_state_dict(state_dict)
    if not lora_state:
        raise ValueError(
            f"No LoRA keys found in checkpoint {ckpt_path!r}. "
            "Expected keys like 'model.*lora_*'."
        )

    logger.info("Applying %d LoRA tensors", len(lora_state))
    incompatible = model.load_state_dict(lora_state, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    logger.info("Loaded LoRA state (missing=%d, unexpected=%d)", missing, unexpected)

    logger.info("Saving adapter + tokenizer to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
