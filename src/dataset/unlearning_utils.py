"""
Dataset utilities for RAQUEL unlearning pipeline.

This module provides utilities for creating specialized datasets used in various
unlearning methods, including IDK datasets, DPO preference pairs, and data sampling.

Legacy note:
    This module is kept only for backwards compatibility with the older
    `src.dataset.providers` path. New code should use `src.training.data`.
"""

import logging
import random
import warnings
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger("UnlearningDatasetUtils")

warnings.warn(
    "src.dataset.unlearning_utils is legacy. Prefer src.training.data for the "
    "supported unlearning data pipeline.",
    DeprecationWarning,
    stacklevel=2,
)

# IDK response templates for targeted unlearning
IDK_TEMPLATES = [
    "I don't know.",
    "I cannot answer that.",
    "I'm not sure.",
    "I don't have information about that.",
    "I cannot provide that information.",
    "I'm unable to answer that question.",
]


def get_idk_response(variation: str = "random") -> str:
    """
    Get an IDK response for targeted unlearning.

    Args:
        variation: Type of variation - "random", "first", or specific template index

    Returns:
        IDK response string
    """
    if variation == "random":
        return random.choice(IDK_TEMPLATES)
    elif variation == "first":
        return IDK_TEMPLATES[0]
    elif variation.isdigit():
        idx = int(variation)
        if 0 <= idx < len(IDK_TEMPLATES):
            return IDK_TEMPLATES[idx]
        else:
            return IDK_TEMPLATES[0]
    else:
        return random.choice(IDK_TEMPLATES)


def sample_examples(
    examples: List[Dict[str, str]],
    sample_size: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Sample examples with deterministic seeding for reproducibility.

    Args:
        examples: List of examples to sample from
        sample_size: Number of examples to sample (None = all)
        seed: Random seed for reproducibility

    Returns:
        Sampled examples
    """
    if not examples or sample_size is None or sample_size <= 0:
        return examples

    if len(examples) <= sample_size:
        return examples

    # Use deterministic sampling
    random.seed(seed)
    return random.sample(examples, sample_size)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    remove_columns: Optional[List[str]] = None,
) -> Dataset:
    """
    Tokenize a dataset for training.

    Args:
        dataset: Dataset to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        remove_columns: Columns to remove after tokenization

    Returns:
        Tokenized dataset
    """
    if remove_columns is None:
        remove_columns = ["text"]

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in remove_columns if col in dataset.column_names],
    )


class UnlearningDatasetFactory:
    """
    Factory class for creating specialized datasets for unlearning methods.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_template: str = "Question: {question}\nAnswer: {answer}",
        max_length: int = 512,
    ):
        """
        Initialize the dataset factory.

        Args:
            tokenizer: HuggingFace tokenizer
            prompt_template: Template for formatting questions and answers
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length

    def create_idk_dataset(
        self,
        forget_examples: List[Dict[str, str]],
        idk_variation: str = "random",
        tokenize: bool = True,
    ) -> Dataset:
        """
        Create IDK dataset for targeted unlearning methods.

        Args:
            forget_examples: Original forget examples with questions and answers
            idk_variation: Type of IDK response variation
            tokenize: Whether to tokenize the dataset

        Returns:
            IDK dataset (tokenized or raw)
        """
        logger.info(
            f"Creating IDK dataset with {len(forget_examples)} examples using '{idk_variation}' variation"
        )

        idk_examples = []
        for example in forget_examples:
            idk_answer = get_idk_response(idk_variation)
            idk_text = self.prompt_template.format(
                question=example["question"], answer=idk_answer
            )
            idk_examples.append({"text": idk_text})

        dataset = Dataset.from_list(idk_examples)
        logger.info(f"Created {len(idk_examples)} IDK examples")

        if tokenize and len(idk_examples) > 0:
            dataset = tokenize_dataset(
                dataset, self.tokenizer, self.max_length, ["text"]
            )

        return dataset

    def create_dpo_preference_pairs(
        self,
        forget_examples: List[Dict[str, str]],
        idk_variation: str = "random",
        tokenize: bool = True,
    ) -> Tuple[Dataset, Dataset]:
        """
        Create preference pairs for DPO: preferred (IDK) and dispreferred (original) responses.

        Args:
            forget_examples: Original forget examples
            idk_variation: Type of IDK response variation
            tokenize: Whether to tokenize the datasets

        Returns:
            Tuple of (preferred_dataset, dispreferred_dataset)
        """
        logger.info(f"Creating DPO preference pairs for {len(forget_examples)} examples")

        # Create preferred responses (IDK)
        preferred_examples = []
        for example in forget_examples:
            idk_answer = get_idk_response(idk_variation)
            preferred_text = self.prompt_template.format(
                question=example["question"], answer=idk_answer
            )
            preferred_examples.append({"text": preferred_text})

        # Create dispreferred responses (original)
        dispreferred_examples = []
        for example in forget_examples:
            dispreferred_text = self.prompt_template.format(
                question=example["question"], answer=example["answer"]
            )
            dispreferred_examples.append({"text": dispreferred_text})

        preferred_dataset = Dataset.from_list(preferred_examples)
        dispreferred_dataset = Dataset.from_list(dispreferred_examples)

        if tokenize:
            preferred_dataset = tokenize_dataset(
                preferred_dataset, self.tokenizer, self.max_length, ["text"]
            )
            dispreferred_dataset = tokenize_dataset(
                dispreferred_dataset, self.tokenizer, self.max_length, ["text"]
            )

        logger.info(f"Created DPO preference pairs with {len(preferred_examples)} examples each")
        return preferred_dataset, dispreferred_dataset

    def create_formatted_dataset(
        self,
        examples: List[Dict[str, str]],
        tokenize: bool = True,
    ) -> Dataset:
        """
        Create a formatted dataset from raw examples.

        Args:
            examples: Raw examples with question/answer pairs
            tokenize: Whether to tokenize the dataset

        Returns:
            Formatted dataset (tokenized or raw)
        """
        formatted_examples = []
        for example in examples:
            formatted_text = self.prompt_template.format(
                question=example["question"], answer=example["answer"]
            )
            formatted_examples.append({"text": formatted_text})

        dataset = Dataset.from_list(formatted_examples)

        if tokenize:
            dataset = tokenize_dataset(
                dataset, self.tokenizer, self.max_length, ["text"]
            )

        return dataset

    def prepare_batch_datasets(
        self,
        forget_examples: List[Dict[str, str]],
        batch_size: int,
        method: str = "idk",
    ) -> Dict[str, Dataset]:
        """
        Prepare specialized datasets for batch processing during training.

        Args:
            forget_examples: Forget examples to use
            batch_size: Target batch size for sampling
            method: Unlearning method requiring special datasets

        Returns:
            Dictionary of prepared datasets
        """
        datasets = {}

        if method in ["idk_gd", "idk_kl"]:
            # Pre-create IDK dataset for IDK methods
            datasets["idk"] = self.create_idk_dataset(forget_examples, tokenize=True)

        elif method in ["dpo_gd", "dpo_kl"]:
            # Pre-create preference pairs for DPO methods
            preferred, dispreferred = self.create_dpo_preference_pairs(
                forget_examples, tokenize=True
            )
            datasets["preferred"] = preferred
            datasets["dispreferred"] = dispreferred

        elif method in ["npo_gd", "npo_kl"]:
            # Pre-create IDK dataset for NPO methods
            datasets["idk"] = self.create_idk_dataset(forget_examples, tokenize=True)

        return datasets


class DatasetSampler:
    """
    Utility class for sampling datasets during training.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the dataset sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def sample_batch_indices(
        self, dataset_size: int, batch_size: int
    ) -> List[int]:
        """
        Sample random indices for a batch from a dataset.

        Args:
            dataset_size: Size of the dataset
            batch_size: Number of indices to sample

        Returns:
            List of sampled indices
        """
        if dataset_size == 0:
            return []

        return [self._rng.randint(0, dataset_size - 1) for _ in range(batch_size)]

    def sample_examples_from_dataset(
        self, dataset: Dataset, num_samples: int
    ) -> List[Dict]:
        """
        Sample examples from a dataset.

        Args:
            dataset: Dataset to sample from
            num_samples: Number of examples to sample

        Returns:
            List of sampled examples
        """
        if len(dataset) == 0 or num_samples <= 0:
            return []

        indices = self.sample_batch_indices(len(dataset), num_samples)
        return [dataset[i] for i in indices]

    def reset_seed(self):
        """Reset the random number generator with the original seed."""
        self._rng = random.Random(self.seed)
