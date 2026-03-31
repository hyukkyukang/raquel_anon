"""Dataset wrappers for RAQUEL training."""

from __future__ import annotations

from typing import Dict, List, Sequence

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.training.data.transforms import ANSWER_PREFIX
from src.training.data.types import DataItem
from src.training.methods import LABEL_IGNORE_INDEX


class QATokenizedDataset(Dataset):
    """
    Dataset wrapper that performs per-item tokenization and prompt masking.

    Optionally prepares prompt-only encodings and decoded texts used during
    validation, so downstream components do not need to recompute them.
    """

    def __init__(
        self,
        dataset: Sequence[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        prepare_generation_fields: bool = False,
    ):
        """
        Initialize the dataset wrapper.

        Args:
            dataset: Underlying dataset providing `question` and `answer` fields.
            tokenizer: Tokenizer used for encoding.
            max_length: Maximum sequence length for encoding.
            prepare_generation_fields: Whether to compute prompt-specific fields
                used during validation text generation.
        """
        # Ensure we operate on python objects instead of pyarrow
        self.dataset: Sequence[Dict[str, str]] = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prepare_generation_fields = prepare_generation_fields

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Guarantee EOS token exists for appending to completions
        if self.tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer must define an eos_token for QATokenizedDataset."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DataItem:
        # Get raw record
        record: Dict[str, str] = self.dataset[idx]
        question: str = record["question"].strip()
        answer: str = record["answer"].strip()

        # Construct full text with prompt and answer
        prompt_text = f"Question: {question}{ANSWER_PREFIX}"
        full_text = prompt_text + answer + self.tokenizer.eos_token  # type: ignore[arg-type]

        # Tokenize the full text
        full_encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        full_input_ids: List[int] = full_encoded["input_ids"]  # type: ignore[index]
        full_attention_mask: List[int] = full_encoded["attention_mask"]  # type: ignore[index]

        # Tokenize the prompt-only text (exclude generated answer) for masking and generation seed
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        prompt_ids: List[int] = prompt_encoding["input_ids"]  # type: ignore[index]
        prompt_length = min(len(prompt_ids), len(full_input_ids))

        # Build labels masking out the prompt tokens
        labels: List[int] = full_input_ids.copy()
        for idx_mask in range(prompt_length):
            labels[idx_mask] = LABEL_IGNORE_INDEX

        # Get optional fields for generation during validation
        prompt_input_ids = None
        prompt_attention_mask = None
        prompt_text = None
        target_text = None
        if self.prepare_generation_fields:
            # Align prompt inputs with potentially truncated main sequence
            prompt_input_ids = full_input_ids[:prompt_length]
            prompt_attention_mask = full_attention_mask[:prompt_length]
            prompt_text = self.tokenizer.decode(
                prompt_input_ids, skip_special_tokens=True
            ).strip()
            target_ids = full_input_ids[prompt_length:]
            target_text = self.tokenizer.decode(
                target_ids, skip_special_tokens=True
            ).strip()

        # Construct data item
        data_item = DataItem(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            labels=labels,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_length=prompt_length,
            prompt_text=prompt_text,
            target_text=target_text,
            sample_id=idx,
        )

        return data_item
