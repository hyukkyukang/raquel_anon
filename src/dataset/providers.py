"""
Dataset providers for unlearning methods.

This module defines abstract interfaces and concrete implementations for providing
specialized datasets to different unlearning methods. This creates a clean separation
between dataset creation and model logic.

Legacy note:
    This module predates the current `src.training.data` datamodule-based pipeline
    and is kept only for backwards compatibility.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .unlearning_utils import DatasetSampler, UnlearningDatasetFactory

logger = logging.getLogger("UnlearningProviders")

warnings.warn(
    "src.dataset.providers is legacy. Prefer src.training.data and "
    "src.training.data.pl_module for supported dataset loading paths.",
    DeprecationWarning,
    stacklevel=2,
)


class UnlearningDataProvider(ABC):
    """
    Abstract base class for providing specialized datasets to unlearning methods.
    """

    def __init__(
        self,
        retain_examples: List[Dict[str, str]],
        forget_examples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_template: str = "Question: {question}\nAnswer: {answer}",
        batch_size: int = 2,
        max_length: int = 512,
    ):
        """
        Initialize the data provider.

        Args:
            retain_examples: Retain examples for regularization
            forget_examples: Forget examples for unlearning
            tokenizer: HuggingFace tokenizer
            prompt_template: Template for formatting questions and answers
            batch_size: Batch size for data loading
            max_length: Maximum sequence length
        """
        self.retain_examples = retain_examples
        self.forget_examples = forget_examples
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize utilities
        self.factory = UnlearningDatasetFactory(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
        )
        self.sampler = DatasetSampler(seed=42)

        # Cached datasets
        self._retain_dataloader: Optional[DataLoader] = None
        self._specialized_datasets: Dict[str, Dataset] = {}

    @abstractmethod
    def get_method_name(self) -> str:
        """Get the unlearning method name this provider supports."""
        pass

    @abstractmethod
    def get_specialized_datasets(self) -> Dict[str, Dataset]:
        """Get specialized datasets required for this unlearning method."""
        pass

    def get_retain_dataloader(self) -> Optional[DataLoader]:
        """
        Get retain dataloader for regularization.

        Returns:
            DataLoader for retain examples, or None if not available
        """
        if self._retain_dataloader is None and self.retain_examples:
            retain_dataset = self.factory.create_formatted_dataset(
                self.retain_examples, tokenize=True
            )

            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
            )

            self._retain_dataloader = DataLoader(
                retain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )

        return self._retain_dataloader

    def sample_retain_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch from retain dataset.

        Returns:
            Batch dictionary or None if retain data not available
        """
        dataloader = self.get_retain_dataloader()
        if dataloader is None:
            return None

        # Get iterator
        if not hasattr(self, '_retain_iterator') or self._retain_iterator is None:
            self._retain_iterator = iter(dataloader)

        try:
            batch = next(self._retain_iterator)
        except StopIteration:
            # Reset iterator when exhausted
            self._retain_iterator = iter(dataloader)
            batch = next(self._retain_iterator)

        return batch

    def sample_forget_examples(self, num_samples: int) -> List[Dict[str, str]]:
        """
        Sample examples from forget dataset.

        Args:
            num_samples: Number of examples to sample

        Returns:
            List of sampled forget examples
        """
        if not self.forget_examples or num_samples <= 0:
            return []

        if len(self.forget_examples) <= num_samples:
            return self.forget_examples

        indices = self.sampler.sample_batch_indices(len(self.forget_examples), num_samples)
        return [self.forget_examples[i] for i in indices]


class GradientAscentProvider(UnlearningDataProvider):
    """
    Data provider for Gradient Ascent (GA) unlearning methods.

    GA methods only need retain data for regularization, no specialized datasets.
    """

    def get_method_name(self) -> str:
        return "ga"

    def get_specialized_datasets(self) -> Dict[str, Dataset]:
        """GA methods don't need specialized datasets."""
        return {}


class NPOProvider(UnlearningDataProvider):
    """
    Data provider for Negative Preference Optimization (NPO) methods.

    NPO methods need IDK datasets for preference learning.
    """

    def get_method_name(self) -> str:
        return "npo"

    def get_specialized_datasets(self) -> Dict[str, Dataset]:
        """Create IDK dataset for NPO."""
        if "idk" not in self._specialized_datasets:
            self._specialized_datasets["idk"] = self.factory.create_idk_dataset(
                self.forget_examples, tokenize=True
            )
        return self._specialized_datasets

    def sample_idk_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch from IDK dataset.

        Args:
            batch_size: Size of batch to sample

        Returns:
            IDK batch dictionary or None if not available
        """
        datasets = self.get_specialized_datasets()
        idk_dataset = datasets.get("idk")

        if idk_dataset is None or len(idk_dataset) == 0:
            return None

        # Sample examples
        examples = self.sampler.sample_examples_from_dataset(idk_dataset, batch_size)
        if not examples:
            return None

        # Collate batch
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        return data_collator(examples)


class IDKProvider(UnlearningDataProvider):
    """
    Data provider for IDK Fine-tuning methods.

    IDK methods need dynamically created IDK datasets during training.
    """

    def get_method_name(self) -> str:
        return "idk"

    def get_specialized_datasets(self) -> Dict[str, Dataset]:
        """IDK methods create datasets dynamically, so return empty."""
        return {}

    def create_idk_batch(
        self,
        forget_examples_batch: List[Dict[str, str]],
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Create IDK batch from forget examples for supervised learning.

        Args:
            forget_examples_batch: Batch of forget examples to convert to IDK
            device: Device to move tensors to

        Returns:
            IDK batch ready for training
        """
        if not forget_examples_batch:
            return None

        # Create IDK examples
        idk_examples = []
        for example in forget_examples_batch:
            idk_text = self.prompt_template.format(
                question=example["question"],
                answer="I don't know."
            )
            idk_examples.append(idk_text)

        # Tokenize
        tokenized = self.tokenizer(
            idk_examples,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels (same as input_ids but with prompt masked)
        labels = tokenized["input_ids"].clone()

        # Mask prompt tokens (only compute loss on IDK answers)
        for i, example in enumerate(forget_examples_batch):
            question = example["question"]
            prompt = f"Question: {question}\nAnswer: "
            prompt_tokens = self.tokenizer(prompt)["input_ids"]
            prompt_len = len(prompt_tokens)
            if prompt_len < labels.size(1):
                labels[i, :prompt_len] = -100

        tokenized["labels"] = labels

        # Move to device if specified
        if device is not None:
            tokenized = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized.items()
            }

        return tokenized


class DPOProvider(UnlearningDataProvider):
    """
    Data provider for Direct Preference Optimization (DPO) methods.

    DPO methods need preference pairs: preferred (IDK) and dispreferred (original) responses.
    """

    def get_method_name(self) -> str:
        return "dpo"

    def get_specialized_datasets(self) -> Dict[str, Dataset]:
        """Create preference pairs for DPO."""
        if "preferred" not in self._specialized_datasets or "dispreferred" not in self._specialized_datasets:
            preferred, dispreferred = self.factory.create_dpo_preference_pairs(
                self.forget_examples, tokenize=True
            )
            self._specialized_datasets["preferred"] = preferred
            self._specialized_datasets["dispreferred"] = dispreferred

        return self._specialized_datasets

    def create_dpo_batch(
        self,
        forget_examples_batch: List[Dict[str, str]],
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Create DPO preference batch from forget examples.

        Args:
            forget_examples_batch: Batch of forget examples
            device: Device to move tensors to

        Returns:
            Tuple of (preferred_batch, dispreferred_batch) or None
        """
        if not forget_examples_batch:
            return None

        # Create preferred responses (IDK)
        preferred_texts = []
        dispreferred_texts = []

        for example in forget_examples_batch:
            # Preferred: IDK response
            preferred_text = self.prompt_template.format(
                question=example["question"], answer="I don't know."
            )
            preferred_texts.append(preferred_text)

            # Dispreferred: original answer
            dispreferred_text = self.prompt_template.format(
                question=example["question"], answer=example["answer"]
            )
            dispreferred_texts.append(dispreferred_text)

        # Tokenize both sets
        preferred_tokenized = self.tokenizer(
            preferred_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        dispreferred_tokenized = self.tokenizer(
            dispreferred_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Move to device if specified
        if device is not None:
            preferred_tokenized = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in preferred_tokenized.items()
            }
            dispreferred_tokenized = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in dispreferred_tokenized.items()
            }

        return preferred_tokenized, dispreferred_tokenized


def create_provider(
    method: str,
    retain_examples: List[Dict[str, str]],
    forget_examples: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> UnlearningDataProvider:
    """
    Factory function to create appropriate data provider for unlearning method.

    Args:
        method: Unlearning method name (e.g., "ga_kl", "npo_gd", etc.)
        retain_examples: Retain examples for regularization
        forget_examples: Forget examples for unlearning
        tokenizer: HuggingFace tokenizer
        **kwargs: Additional arguments for provider

    Returns:
        Appropriate data provider instance

    Raises:
        ValueError: If method is not supported
    """
    # Extract base method from method name
    base_method = method.split("_")[0]

    provider_classes = {
        "ga": GradientAscentProvider,
        "npo": NPOProvider,
        "idk": IDKProvider,
        "dpo": DPOProvider,
    }

    if base_method not in provider_classes:
        raise ValueError(f"Unsupported unlearning method: {method}")

    provider_class = provider_classes[base_method]
    return provider_class(
        retain_examples=retain_examples,
        forget_examples=forget_examples,
        tokenizer=tokenizer,
        **kwargs
    )
