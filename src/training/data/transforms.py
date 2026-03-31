"""Data transformation functions for training."""

import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
else:
    PreTrainedTokenizer = Any

from src.training.methods import LABEL_IGNORE_INDEX
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Prompt template for QA pairs
PROMPT_TEMPLATE: str = "Question: {question}\nAnswer: {answer}"

# IDK response templates for targeted unlearning
IDK_TEMPLATES = [
    "I don't know.",
    "I cannot answer that.",
    "I'm not sure.",
    "I don't have information about that.",
    "I cannot provide that information.",
    "I'm unable to answer that question.",
]
ANSWER_PREFIX = "\nAnswer: "


def format_examples(examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format QA pairs into a single 'text' field using the prompt template.

    Args:
        examples (List[Dict[str, str]]): List of QA dicts with 'question' and 'answer' keys.

    Returns:
        List[Dict[str, str]]: List with 'text' key for each example.
    """
    logger.debug("Formatting %d examples using PROMPT_TEMPLATE", len(examples))
    formatted: List[Dict[str, str]] = [
        {"text": PROMPT_TEMPLATE.format(question=ex["question"], answer=ex["answer"])}
        for ex in examples
    ]
    return formatted


def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> Dict[str, Any]:
    """
    Tokenize the 'text' field of examples and precompute labels such that
    loss is only computed on the answer tokens (prompt tokens are masked with -100).

    Args:
        examples (Dict[str, List[str]]): Batched examples from dataset. Accepts either
            preformatted ``text`` entries or paired ``question``/``answer`` fields.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        Dict[str, Any]: Tokenized inputs with a 'labels' field where the prompt portion is masked to -100.
    """
    # Ensure tokenizer has an eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token or "</s>"

    texts: List[str] = []
    if "text" in examples:
        for raw_text in examples["text"]:
            text = str(raw_text).strip()
            if tokenizer.eos_token is not None and not text.endswith(tokenizer.eos_token):
                text = text + tokenizer.eos_token
            texts.append(text)
    else:
        for question, answer in zip(examples["question"], examples["answer"]):
            text = f"Question: {question.strip()}{ANSWER_PREFIX}{answer.strip()}"
            text = text + tokenizer.eos_token  # type: ignore
            texts.append(text)

    tokenized_batch: Dict[str, Any] = tokenizer(
        texts, truncation=True, max_length=max_length
    )  # type: ignore

    # Build per-example labels masking the prompt up to and including "\nAnswer: "
    labels_list: List[List[int]] = []
    for idx, text in enumerate(texts):
        # Create a copy of input_ids for labels
        input_ids_list: List[int] = list(tokenized_batch["input_ids"][idx])
        labels_ids: List[int] = input_ids_list.copy()

        assert ANSWER_PREFIX in text, "Text does not contain expected prompt format."
        prompt_only: str = text.split(ANSWER_PREFIX, 1)[0] + ANSWER_PREFIX
        prompt_ids: List[int] = tokenizer(
            prompt_only,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )[
            "input_ids"
        ]  # type: ignore
        prompt_len: int = min(len(prompt_ids), len(labels_ids))

        # Mask the prompt tokens
        if prompt_len > 0:
            labels_ids[:prompt_len] = [LABEL_IGNORE_INDEX] * prompt_len

        labels_list.append(labels_ids)

    tokenized_batch["labels"] = labels_list
    return tokenized_batch


def get_idk_response(variation: str = "random") -> str:
    """
    Get an IDK response for targeted unlearning.

    Args:
        variation (str): Type of variation - "random", "first", or specific template

    Returns:
        str: IDK response string
    """
    if variation == "random":
        return random.choice(IDK_TEMPLATES)
    elif variation == "first":
        return IDK_TEMPLATES[0]
    elif variation in IDK_TEMPLATES:
        return variation
    else:
        return IDK_TEMPLATES[0]  # Default fallback


def create_idk_dataset(
    forget_examples: Sequence[Dict[str, str]], idk_variation: str = "random"
) -> List[Dict[str, str]]:
    """
    Create IDK dataset by replacing forget answers with "I don't know" responses.

    Args:
        forget_examples (Sequence[Dict[str, str]]): Original forget examples with 'question'/'answer' keys
        idk_variation (str): Type of IDK response variation

    Returns:
        List[Dict[str, str]]: Examples with IDK answers
    """
    # This function can be called per-step for targeted losses (IDK/DPO).
    # Keep logging at DEBUG to avoid large stdout logs slowing down training.
    logger.debug("Creating IDK dataset from %d forget examples", len(forget_examples))
    idk_examples = []

    for ex in forget_examples:
        idk_response = get_idk_response(idk_variation)
        idk_examples.append({"question": ex["question"], "answer": idk_response})

    logger.debug("Created IDK dataset with %d examples", len(idk_examples))
    return idk_examples
