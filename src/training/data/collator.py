"""Custom data collator that pads variable-length sequences."""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import DataCollatorForLanguageModeling

from src.training.data.types import DataItem
from src.training.methods import LABEL_IGNORE_INDEX


class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Collate variable-length sequences by padding tensor fields while keeping
    string metadata intact. Assumes per-item tokenization has already been
    applied by the dataset.
    """

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id  # type: ignore[union-attr]

    def __call__(
        self, features: List[DataItem], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        del return_tensors

        # Return empty batch if no features
        if not features:
            return {}

        batch: Dict[str, Any] = {}

        for key in DataItem.keys():
            values: List[Any] = [feature.get(key) for feature in features]
            collated_value = self._collate_field(key=key, values=values)
            batch[key] = collated_value

        return batch

    def _collate_field(
        self,
        key: str,
        values: List[Any],
    ) -> Union[torch.Tensor, List[str], None]:
        if all(value is None for value in values):
            return None

        if any(value is None for value in values):
            raise ValueError(
                f"Inconsistent values for key '{key}': some items missing data."
            )

        first_value = values[0]

        # Handle list of lists of ints (e.g., input_ids, attention_mask, labels)
        if (
            isinstance(first_value, list)
            and first_value
            and isinstance(first_value[0], int)
        ):
            return self._pad_tensor_list(values, key)

        # Handle list of sequences of ints (e.g., some special cases)
        if isinstance(first_value, list) and not first_value:
            return torch.zeros((len(values), 0), dtype=torch.long)

        # Handle string metadata (e.g., decoded texts)
        if isinstance(first_value, str):
            return values  # type: ignore[return-value]

        # Handle list of ints
        if isinstance(first_value, Sequence):
            return self._pad_tensor_sequence(values, key)

        # Fallback: convert to tensor directly
        return torch.tensor(values, dtype=torch.long)

    def _pad_tensor_list(
        self,
        values: List[List[int]],
        key: str,
    ) -> torch.Tensor:
        tensors = [torch.tensor(value, dtype=torch.long) for value in values]
        padding_value = self._padding_value_for_key(key)

        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=padding_value
        )

    def _pad_tensor_sequence(
        self,
        values: List[Sequence[int]],
        key: str,
    ) -> torch.Tensor:
        tensors = [torch.tensor(list(value), dtype=torch.long) for value in values]
        padding_value = self._padding_value_for_key(key)

        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=padding_value
        )

    def _padding_value_for_key(self, key: str) -> int:
        if key.endswith("attention_mask"):
            return 0
        if key == "labels":
            return LABEL_IGNORE_INDEX
        return self.pad_token_id
