"""Shared RAQUEL evaluation helpers for scripts and training callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import RougeMetric, SemanticAccuracyMetric, SemanticMetricConfig
from src.training.data.utils import load_json_dataset
from src.training.data.transforms import ANSWER_PREFIX
from src.utils.generation import build_greedy_generation_kwargs
from src.utils.logging import get_logger

logger = get_logger(__name__)

RAQUELExample = Dict[str, str]
DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class RAQUELEvalConfig:
    """Configuration for RAQUEL text-generation evaluation."""

    batch_size: int = 8
    max_new_tokens: int = 64
    max_prompt_length: Optional[int] = None
    max_examples: Optional[int] = None
    save_predictions: bool = False


def load_raquel_examples(file_path: str) -> List[RAQUELExample]:
    """Load RAQUEL examples from a JSON or JSONL file."""
    return load_json_dataset(str(Path(file_path)))


def resolve_generation_device(
    model: PreTrainedModel,
    requested_device: Optional[DeviceLike],
) -> torch.device:
    """Resolve the device used for generation, preferring the model's actual device."""
    requested_device_str = (
        str(requested_device) if requested_device is not None else None
    )

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for raw_device in hf_device_map.values():
            if raw_device in (None, "disk"):
                continue
            if isinstance(raw_device, int):
                model_device = torch.device(f"cuda:{raw_device}")
                break

            device_str = str(raw_device).strip()
            if device_str:
                model_device = torch.device(device_str)
                break
        else:
            model_device = None
    else:
        model_device = None

    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            fallback: str = requested_device_str or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            return torch.device(fallback)

    requested_device_kind = None
    if requested_device_str:
        try:
            requested_device_kind = torch.device(requested_device_str).type
        except (TypeError, RuntimeError):
            requested_device_kind = requested_device_str

    if (
        requested_device_str
        and str(model_device) != requested_device_str
        and requested_device_kind != model_device.type
    ):
        logger.warning(
            "Requested device %s but model is on %s; using model device.",
            requested_device_str,
            model_device,
        )
    return model_device


def _build_prompts(examples: Sequence[RAQUELExample]) -> List[str]:
    """Format QA examples into generation prompts."""
    return [
        f"Question: {str(example.get('question', '')).strip()}{ANSWER_PREFIX}"
        for example in examples
    ]


def _generate_predictions(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    config: RAQUELEvalConfig,
    device: torch.device,
) -> List[str]:
    """Generate predictions for a list of prompts."""
    predictions: List[str] = []
    was_training: bool = model.training
    model.eval()

    original_padding_side: Optional[str] = getattr(tokenizer, "padding_side", None)
    if original_padding_side is not None:
        tokenizer.padding_side = "left"

    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), config.batch_size):
                batch_prompts = list(prompts[start : start + config.batch_size])
                encoded = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=config.max_prompt_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                input_seq_len = int(input_ids.size(1))

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **build_greedy_generation_kwargs(
                        model,
                        tokenizer,
                        input_length=input_seq_len,
                        max_new_tokens=config.max_new_tokens,
                    ),
                )

                for row_idx in range(generated.size(0)):
                    gen_ids = generated[row_idx][input_seq_len:]
                    predictions.append(
                        tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    )
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    return predictions


def evaluate_raquel_split(
    *,
    name: str,
    examples: Sequence[RAQUELExample],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: RAQUELEvalConfig,
    device: torch.device,
    semantic_cfg: Optional[SemanticMetricConfig] = None,
    global_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate one RAQUEL split."""
    eval_examples = list(examples)
    if config.max_examples is not None:
        eval_examples = eval_examples[: config.max_examples]

    prompts = _build_prompts(eval_examples)
    references = [str(example.get("answer", "")).strip() for example in eval_examples]

    predictions = _generate_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        config=config,
        device=device,
    )

    rouge_metric = RougeMetric()
    rouge_metric.update(predictions=predictions, references=references)
    result: Dict[str, Any] = {
        "split": name,
        "count": len(eval_examples),
        "rouge": rouge_metric.compute(),
    }

    if semantic_cfg is not None and semantic_cfg.enabled:
        semantic_metric = SemanticAccuracyMetric(semantic_cfg, global_cfg=global_cfg)
        semantic_metric.update(
            questions=[
                str(example.get("question", "")).strip() for example in eval_examples
            ],
            predictions=predictions,
            references=references,
        )
        semantic_accuracy = semantic_metric.compute()
        if semantic_accuracy is not None:
            result["semantic_accuracy"] = semantic_accuracy

    if config.save_predictions:
        result["predictions"] = [
            {
                "question": str(eval_examples[idx].get("question", "")),
                "reference": references[idx],
                "prediction": predictions[idx],
            }
            for idx in range(len(eval_examples))
        ]

    return result


def evaluate_raquel_splits(
    *,
    split_examples: Mapping[str, Sequence[RAQUELExample]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: RAQUELEvalConfig,
    device: Optional[DeviceLike] = None,
    semantic_cfg: Optional[SemanticMetricConfig] = None,
    global_cfg: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all requested RAQUEL splits with shared generation settings."""
    resolved_device = resolve_generation_device(model, device)
    return {
        split_name: evaluate_raquel_split(
            name=split_name,
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=resolved_device,
            semantic_cfg=semantic_cfg,
            global_cfg=global_cfg,
        )
        for split_name, examples in split_examples.items()
    }


def build_raquel_metric_logs(
    results: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str = "raquel",
) -> Dict[str, float]:
    """Flatten RAQUEL evaluation results into logger-friendly scalar metrics."""
    metrics: Dict[str, float] = {}
    for split_name, payload in results.items():
        rouge_scores = payload.get("rouge", {})
        for key, value in rouge_scores.items():
            metrics[f"{prefix}/{split_name}/{key}"] = float(value)

        semantic_accuracy = payload.get("semantic_accuracy")
        if semantic_accuracy is not None:
            metrics[f"{prefix}/{split_name}/semantic_accuracy"] = float(
                semantic_accuracy
            )
    return metrics
