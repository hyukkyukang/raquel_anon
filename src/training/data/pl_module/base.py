"""Base LightningDataModule with common functionality."""

import logging
from typing import Callable, Optional

import lightning.pytorch as pl
import torch
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from src.training.data.dataloader import build_distributed_sampler
from src.training.data.transforms import tokenize_function
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule with common functionality for loading and tokenizing QA datasets.

    This provides shared methods for:
    - Loading JSON datasets
    - Formatting examples with prompt template
    - Tokenizing with proper label masking
    - Creating dataloaders with custom collator
    """

    def __init__(
        self,
        name: Optional[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,  # Changed default from 0 to 4 for better performance
        max_length: int = 1024,
    ):
        """
        Initialize base data module.

        Args:
            tokenizer: Pre-trained tokenizer for text processing
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading (default: 4)
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        default_batch_size = (
            self._coerce_positive_int(batch_size) if batch_size is not None else 32
        )
        self.train_batch_size = self._coerce_positive_int(
            train_batch_size, fallback=default_batch_size
        )
        self.val_batch_size = self._coerce_positive_int(
            val_batch_size, fallback=default_batch_size
        )
        # Preserve legacy attribute for callers that still rely on a single batch size
        self.batch_size = self.train_batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Llama pretraining uses right padding when padding is required; enforce to keep labels aligned
        self.tokenizer.padding_side = "right"

    def prepare_data(self) -> None:
        """
        Downloads the dataset if not already present.
        This method is called only on a single process in distributed training.
        """
        # Download the dataset from the HuggingFace hub (download only, do not load into memory)
        if self.name is None:
            return None
        snapshot_download(repo_id=self.name, repo_type="dataset")
        return None

    def load_dataset(
        self, name: str, split: str, subset: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset from HuggingFace Datasets by name.

        Args:
            name: Name of the dataset to load
            split: Dataset split to load (train, validation, test)
            subset: Optional subset/config name for the dataset

        Returns:
            HuggingFace Dataset
        """
        subset_msg = f" [subset={subset}]" if subset else ""
        logger.info(f"Loading dataset {name}{subset_msg} ({split})")
        load_kwargs = {"split": split}
        if subset:
            load_kwargs["name"] = subset
        dataset: Dataset = hf_load_dataset(name, **load_kwargs)  # type: ignore
        logger.info(
            "Loaded dataset %s%s (%s) with %d examples",
            name,
            subset_msg,
            split,
            len(dataset),
        )
        return dataset

    def tokenize_dataset(
        self, dataset: Dataset, remove_text_column: bool = False
    ) -> Dataset:
        """
        Tokenize dataset with proper label masking.

        Args:
            dataset: Dataset with 'text' column
            remove_text_column: Whether to remove 'text' column after tokenization

        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset with %d examples", len(dataset))
        tokenized = dataset.map(
            lambda ex: tokenize_function(ex, self.tokenizer),
            batched=True,
            remove_columns=["question", "answer"] if remove_text_column else [],
        )
        logger.info("Tokenization complete")
        return tokenized

    def create_dataloader(
        self,
        dataset: TorchDataset,
        collator: Callable,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        multiprocessing_context: str = None,
    ) -> DataLoader:
        """
        Create dataloader for a dataset with DDP support.

        Automatically uses DistributedSampler when running with multiple GPUs.

        Args:
            dataset: Tokenized dataset
            shuffle: Whether to shuffle data
            multiprocessing_context: None, "fork", or "spawn"

        Returns:
            DataLoader instance
        """
        sampler = build_distributed_sampler(dataset=dataset, shuffle=shuffle)
        if sampler is not None:
            shuffle = False
            logger.debug("Using DistributedSampler for multi-GPU training")

        if batch_size is None:
            batch_size = self.train_batch_size if shuffle else self.val_batch_size
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=multiprocessing_context,
        )

    @staticmethod
    def _coerce_positive_int(
        value: Optional[int], fallback: Optional[int] = None
    ) -> Optional[int]:
        """Convert a value to a positive int, or fall back to the provided default."""
        if value is None:
            return fallback
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return fallback
        return int_value if int_value > 0 else fallback
