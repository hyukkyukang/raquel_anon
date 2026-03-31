"""Utility functions for loss computation."""

from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.data.transforms import PROMPT_TEMPLATE


def prompt_prefix(question: str) -> str:
    """Return the canonical prompt prefix for a question."""
    return f"Question: {question}\nAnswer: "


def resolve_model_device(model: PreTrainedModel) -> torch.device:
    """Resolve the device hosting model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_masked_batch(
    tokenizer: PreTrainedTokenizer,
    examples: List[Dict[str, str]],
    device: torch.device,
    max_length: int = 1024,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize examples and mask prompt tokens so loss focuses on answers.

    Args:
        tokenizer: Tokenizer for processing examples
        examples: List of examples with 'question' and 'answer' keys
        device: Device to move tensors to
        max_length: Maximum sequence length for tokenization

    Returns:
        Dictionary of tokenized inputs with masked labels
    """
    texts = [
        PROMPT_TEMPLATE.format(question=ex["question"], answer=ex["answer"])
        for ex in examples
    ]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = tokenized["input_ids"].clone()
    seq_len = labels.size(1)

    for idx, example in enumerate(examples):
        prompt_ids = tokenizer(
            prompt_prefix(example["question"]),
            truncation=True,
            max_length=seq_len,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), seq_len)
        labels[idx, :prompt_len] = -100

    tokenized["labels"] = labels
    return {k: v.to(device) for k, v in tokenized.items()}
