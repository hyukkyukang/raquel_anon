"""
DataModule for MUSE Evaluation with Multi-GPU Support.

Provides PyTorch Lightning DataModules for loading and batching
MUSE evaluation datasets with proper distributed sampling.
"""

import logging
from typing import Dict, List, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MUSEEvaluationDataset(Dataset):
    """
    Dataset for MUSE evaluation.

    Stores questions and ground truth answers for evaluation.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_answer_in_prompt: bool = False,
    ):
        """
        Initialize MUSE evaluation dataset.

        Args:
            examples: List of dicts with 'question' and 'answer' keys
            tokenizer: Tokenizer for encoding questions
            max_length: Maximum sequence length
            include_answer_in_prompt: Whether to include answer in prompt (for privacy metric)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_answer_in_prompt = include_answer_in_prompt

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.examples[idx]
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Create prompt
        if self.include_answer_in_prompt:
            # For privacy metric: include full QA pair
            prompt = f"Question: {question}\nAnswer: {answer}"
        else:
            # For generation metrics: only question
            prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ground_truth": answer,
            "question": question,
        }


class MUSEPrivacyDataset(MUSEEvaluationDataset):
    """
    Dataset for MUSE privacy evaluation (MIA).

    Combines forget set (members) and non-training set (non-members)
    with membership labels.
    """

    def __init__(
        self,
        forget_examples: List[Dict],
        non_training_examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize privacy evaluation dataset.

        Args:
            forget_examples: Examples from forget set (members)
            non_training_examples: Examples not in training set (non-members)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        # Combine examples and create labels
        all_examples = forget_examples + non_training_examples
        self.is_member = [1] * len(forget_examples) + [0] * len(non_training_examples)

        super().__init__(
            examples=all_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            include_answer_in_prompt=True,  # Privacy needs full QA
        )

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = super().__getitem__(idx)
        item["is_member"] = self.is_member[idx]
        return item


class MUSEDataModule(pl.LightningDataModule):
    """
    DataModule for MUSE evaluation with DDP support.

    Handles batching and distributed sampling for MUSE evaluation datasets.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512,
        include_answer_in_prompt: bool = False,
    ):
        """
        Initialize MUSE DataModule.

        Args:
            examples: List of evaluation examples
            tokenizer: Tokenizer for encoding
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            max_length: Maximum sequence length
            include_answer_in_prompt: Whether to include answer in prompt
        """
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.include_answer_in_prompt = include_answer_in_prompt

        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup dataset for evaluation."""
        if stage == "test" or stage is None:
            self.test_dataset = MUSEEvaluationDataset(
                examples=self.examples,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                include_answer_in_prompt=self.include_answer_in_prompt,
            )
            logger.info(f"Setup MUSE dataset with {len(self.test_dataset)} examples")

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader with optional distributed sampling.

        Returns:
            DataLoader for evaluation
        """
        if self.test_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        # Check if using DDP
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.test_dataset,
                shuffle=False,
                drop_last=False,
            )
            logger.info("Using DistributedSampler for multi-GPU evaluation")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            shuffle=False if sampler else False,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, any]:
        """
        Collate batch with padding.

        Args:
            batch: List of examples

        Returns:
            Batched and padded tensors
        """
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        ground_truths = [item["ground_truth"] for item in batch]
        questions = [item["question"] for item in batch]

        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        result = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "ground_truth": ground_truths,
            "question": questions,
        }

        # Include is_member if present (for privacy metric)
        if "is_member" in batch[0]:
            result["is_member"] = [item["is_member"] for item in batch]

        return result


class MUSEPrivacyDataModule(pl.LightningDataModule):
    """
    DataModule for MUSE privacy evaluation (MIA) with DDP support.

    Combines forget set and non-training set with membership labels.
    """

    def __init__(
        self,
        forget_examples: List[Dict],
        non_training_examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512,
    ):
        """
        Initialize privacy evaluation DataModule.

        Args:
            forget_examples: Examples from forget set
            non_training_examples: Examples not in training
            tokenizer: Tokenizer for encoding
            batch_size: Batch size for evaluation
            num_workers: Number of workers
            max_length: Maximum sequence length
        """
        super().__init__()
        self.forget_examples = forget_examples
        self.non_training_examples = non_training_examples
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup privacy dataset."""
        if stage == "test" or stage is None:
            self.test_dataset = MUSEPrivacyDataset(
                forget_examples=self.forget_examples,
                non_training_examples=self.non_training_examples,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            logger.info(
                f"Setup privacy dataset: {len(self.forget_examples)} forget + "
                f"{len(self.non_training_examples)} non-training = {len(self.test_dataset)} total"
            )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader for privacy evaluation."""
        if self.test_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        # Check if using DDP
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.test_dataset,
                shuffle=False,
                drop_last=False,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, any]:
        """Collate batch with padding."""
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        ground_truths = [item["ground_truth"] for item in batch]
        questions = [item["question"] for item in batch]
        is_member = [item["is_member"] for item in batch]

        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "ground_truth": ground_truths,
            "question": questions,
            "is_member": is_member,
        }
