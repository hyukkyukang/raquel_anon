"""DataModule for standard fine-tuning."""

import logging
import os
from typing import List, Optional, Sequence

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.training.data.collator import CustomDataCollator
from src.training.data.datasets import QATokenizedDataset
from src.training.data.utils import load_json_dataset
from src.training.utils import log_if_rank_zero
from src.utils.logging import get_logger

from .base import BaseDataModule

logger = get_logger(__name__)


class FineTuneDataModule(BaseDataModule):
    """
    DataModule for standard fine-tuning on QA datasets.

    Loads a training subset and supports composing the validation split
    from one or more dataset subsets.
    """

    def __init__(
        self,
        name: Optional[str],
        train_subset_name: Optional[str],
        train_split: str,
        val_split: Optional[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        max_length: int = 1024,
        val_subset_names: Optional[Sequence[str]] = None,
        val_sample_num: Optional[int] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        val_files: Optional[Sequence[str]] = None,
    ):
        """
        Initialize FineTuneDataModule.

        Args:
            name: Hugging Face dataset repository id
            train_subset_name: Optional dataset configuration for the train split
            train_split: Dataset split to use for training
            val_split: Optional dataset split to use for validation
            tokenizer: Pre-trained tokenizer
            batch_size: Default batch size applied to both loaders if specific values are not provided
            train_batch_size: Optional override for training batch size
            val_batch_size: Optional override for validation batch size
            num_workers: Number of workers for data loading
            max_length: Maximum sequence length
            val_subset_names: Optional ordered list of validation subsets to load
            val_sample_num: Optional number of validation examples to retain
        """
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )
        self.name = name
        self.train_subset_name = train_subset_name
        self.train_split = train_split
        self.val_split = val_split
        self.val_sample_num = val_sample_num
        self.train_file = train_file
        self.val_file = val_file
        self.val_files = self._normalize_val_files(val_file, val_files)
        self.train_dataset = None
        self.val_datasets: List[QATokenizedDataset] = []
        self.val_subset_labels: List[str] = []
        self.val_subset_names = self._normalize_val_subset_names(val_subset_names)

    def _normalize_val_subset_names(
        self,
        val_subset_names: Optional[Sequence[str]],
    ) -> List[str]:
        """Convert validation subsets into a de-duplicated list while preserving order."""
        if val_subset_names is None:
            return []
        candidates = list(val_subset_names)
        seen = set()
        normalized: List[str] = []
        for subset in candidates:
            if not subset or subset in seen:
                continue
            seen.add(subset)
            normalized.append(subset)
        return normalized

    @staticmethod
    def _normalize_val_files(
        val_file: Optional[str],
        val_files: Optional[Sequence[str]],
    ) -> List[str]:
        """Normalize validation file inputs into a de-duplicated list."""
        files: List[str] = []
        if val_file:
            files.append(val_file)
        if val_files:
            files.extend([item for item in val_files if item])
        seen = set()
        normalized: List[str] = []
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            normalized.append(path)
        return normalized

    @staticmethod
    def _label_from_path(path: str) -> str:
        """Derive a short label from a file path."""
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        return stem or base

    def _setup_training(self, stage: Optional[str] = None) -> None:
        if stage and stage != "fit":
            return None
        if self.train_file:
            log_if_rank_zero(
                "Loading training data from file: %s", self.train_file
            )
            train_data = load_json_dataset(self.train_file)
        else:
            if self.name is None:
                raise ValueError("Training dataset requires name or train_file.")
            train_data = self.load_dataset(
                name=self.name,  # type: ignore
                split=self.train_split,
                subset=self.train_subset_name,
            )  # type: ignore
        self.train_dataset = QATokenizedDataset(
            dataset=train_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            prepare_generation_fields=False,
        )
        log_if_rank_zero(f"Training dataset loaded: {len(self.train_dataset)} examples")

    def _setup_validation(self) -> None:
        self.val_datasets = []
        self.val_subset_labels = []

        if self.val_files:
            for idx, path in enumerate(self.val_files):
                dataset = load_json_dataset(path)

                if self.val_sample_num is not None:
                    limit = min(self.val_sample_num, len(dataset))
                    dataset = dataset[:limit]

                tokenized_dataset = QATokenizedDataset(
                    dataset=dataset,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    prepare_generation_fields=True,
                )

                self.val_datasets.append(tokenized_dataset)
                self.val_subset_labels.append(self._label_from_path(path))

                log_if_rank_zero(
                    "Validation dataset %d: file='%s' num_examples=%d",
                    idx,
                    path,
                    len(tokenized_dataset),
                )
            self.val_subset_names = self.val_subset_labels
            return None

        if self.val_split is None:
            return None

        if self.name is None:
            raise ValueError("Validation dataset requires name or val_files.")

        for idx, subset_name in enumerate(self.val_subset_names):
            # Load the dataset
            dataset = self.load_dataset(
                name=self.name,  # type: ignore
                split=self.val_split,
                subset=subset_name,
            )

            # Sample a limited number of validation examples if specified
            if self.val_sample_num is not None:
                limit = min(self.val_sample_num, len(dataset))
                dataset = dataset.select(range(limit))  # type: ignore

            # Convert to tokenized dataset
            tokenized_dataset = QATokenizedDataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                prepare_generation_fields=True,
            )

            # Aggregate
            self.val_datasets.append(tokenized_dataset)
            self.val_subset_labels.append(subset_name)

            log_if_rank_zero(
                f"Validation dataset {idx}: Name='{self.name}', split='{self.val_split}', subset='{subset_name}' num_examples={len(tokenized_dataset)}"
            )
        return None

    def setup(self, stage: Optional[str] = None):
        """
        Load and tokenize datasets.

        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        log_if_rank_zero(f"Loading training data from {self.name} ({self.train_split})")
        self._setup_training(stage=stage)

        log_if_rank_zero(f"Loading validation data from {self.name} ({self.val_split})")
        self._setup_validation()
        return None

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        return self.create_dataloader(
            dataset=self.train_dataset,
            collator=CustomDataCollator(tokenizer=self.tokenizer, mlm=False),
            shuffle=True,
            batch_size=self.train_batch_size,
        )

    def val_dataloader(self) -> Optional[List[DataLoader]]:
        """Create validation dataloader."""
        if not self.val_datasets:
            return None
        collator = CustomDataCollator(tokenizer=self.tokenizer, mlm=False)
        dataloaders = [
            self.create_dataloader(
                dataset=dataset,
                collator=collator,
                shuffle=False,
                batch_size=self.val_batch_size,
            )
            for dataset in self.val_datasets
        ]
        return dataloaders
