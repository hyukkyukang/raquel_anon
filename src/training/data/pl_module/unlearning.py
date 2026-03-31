"""DataModule for unlearning with forget, retain, and optional IDK datasets."""

from typing import Dict, List, Optional, Sequence, cast

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.training.data.collator import CustomDataCollator
from src.training.data.dataloader import UnlearningDataLoader
from src.training.data.datasets import QATokenizedDataset
from src.training.data.transforms import create_idk_dataset
from src.training.data.utils import load_json_dataset
from src.utils.logging import get_logger

from .base import BaseDataModule

logger = get_logger(__name__)

Example = Dict[str, str]


class UnlearningDataModule(BaseDataModule):
    """
    DataModule for unlearning with forget, retain, and optionally IDK datasets.

    Datasets can be sourced from either HuggingFace (via dataset name/splits) or
    local filesystem JSON/JSONL files on a per-split basis.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: Optional[str],
        forget_split: Optional[str],
        retain_split: Optional[str],
        *,
        forget_file: Optional[str] = None,
        retain_file: Optional[str] = None,
        forget_subset_name: Optional[str] = None,
        retain_subset_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        max_length: int = 1024,
        create_idk: bool = False,
        idk_variation: str = "random",
        is_debugging: bool = False,
    ):
        """
        Initialize UnlearningDataModule.

        Args:
            tokenizer: Pre-trained tokenizer.
            dataset_name: Optional HuggingFace dataset repo id backing the splits.
            forget_split: Split name for forget data when using HuggingFace datasets.
            retain_split: Split name for retain data when using HuggingFace datasets.
            forget_file: Optional path to a local JSON/JSONL file for forget data.
            retain_file: Optional path to a local JSON/JSONL file for retain data.
            forget_subset_name: Optional dataset subset/config for forget split.
            retain_subset_name: Optional dataset subset/config for retain split.
            batch_size: Default batch size applied when per-loader overrides are absent.
            train_batch_size: Optional override for forget loader batch size.
            val_batch_size: Optional override for retain/validation loader batch size.
            num_workers: Number of dataloader workers.
            max_length: Maximum tokenized sequence length.
            create_idk: Whether to synthesize an IDK dataset from forget data.
            idk_variation: IDK response variation strategy.
            is_debugging: Whether is debugging mode.
        """
        super().__init__(
            name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )
        self.dataset_name = dataset_name
        self.forget_split = forget_split
        self.retain_split = retain_split
        self.forget_file = forget_file
        self.retain_file = retain_file
        self.forget_subset_name = forget_subset_name
        self.retain_subset_name = retain_subset_name
        self.create_idk = create_idk
        self.idk_variation = idk_variation

        self.is_debugging = is_debugging

        self.forget_dataset: Optional[QATokenizedDataset] = None
        self.retain_dataset: Optional[QATokenizedDataset] = None
        self.idk_dataset: Optional[QATokenizedDataset] = None

        self.forget_raw: Optional[Sequence[Example]] = None

        self._collator = CustomDataCollator(tokenizer=self.tokenizer, mlm=False)

    @property
    def multiprocessing_context(self) -> str:
        """
        Returns the multiprocessing context to use for DataLoader workers.

        - If debugging mode is enabled and num_workers > 0, use "fork" mode for simpler debugging.
        - Otherwise, return None, letting PyTorch use its default context (typically "spawn" for safety).
        """
        return "fork" if self.is_debugging and self.num_workers > 0 else None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_examples(
        self,
        *,
        label: str,
        file_path: Optional[str],
        split: Optional[str],
        subset: Optional[str],
    ) -> Sequence[Example]:
        """Load QA examples from the requested source."""
        if file_path:
            logger.info("Loading %s data from file=%s", label, file_path)
            return load_json_dataset(file_path)

        if self.dataset_name is None:
            raise ValueError(
                f"{label.title()} dataset requires either dataset_name or {label}_file."
            )
        if not split:
            raise ValueError(
                f"{label.title()} dataset requires split information when using HuggingFace datasets."
            )

        subset_msg = f", subset={subset}" if subset else ""
        logger.info(
            "Loading %s data from dataset=%s (split=%s%s)",
            label,
            self.dataset_name,
            split,
            subset_msg,
        )
        dataset = self.load_dataset(
            name=self.dataset_name,
            split=split,
            subset=subset,
        )
        return cast(Sequence[Example], dataset)

    def _build_dataset(
        self,
        examples: Sequence[Example],
        *,
        prepare_generation_fields: bool = False,
    ) -> QATokenizedDataset:
        """Create a tokenized dataset wrapper from raw QA examples."""
        return QATokenizedDataset(
            dataset=examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            prepare_generation_fields=prepare_generation_fields,
        )

    def _prepare_forget_dataset(self) -> None:
        """Load and tokenize the forget split."""
        forget_examples = self._load_examples(
            label="forget",
            file_path=self.forget_file,
            split=self.forget_split,
            subset=self.forget_subset_name,
        )
        self.forget_raw = forget_examples
        self.forget_dataset = self._build_dataset(
            forget_examples,
            prepare_generation_fields=True,
        )
        logger.info(
            f"Forget dataset prepared with {len(self.forget_dataset)} examples."
        )

    def _prepare_retain_dataset(self) -> None:
        """Load and tokenize the retain split."""
        retain_examples = self._load_examples(
            label="retain",
            file_path=self.retain_file,
            split=self.retain_split,
            subset=self.retain_subset_name,
        )
        # Enable generation fields so validation can decode prompts/targets.
        self.retain_dataset = self._build_dataset(
            retain_examples,
            prepare_generation_fields=True,
        )
        logger.info(
            f"Retain dataset prepared with {len(self.retain_dataset)} examples."
        )

    def _prepare_idk_dataset(self) -> None:
        """Optionally synthesize and tokenize the IDK dataset."""
        if not self.create_idk:
            self.idk_dataset = None
            return
        if not self.forget_raw:
            raise ValueError("Forget dataset must be loaded before creating IDK data.")
        logger.info(
            f"Creating IDK dataset from forget data (variation={self.idk_variation})"
        )
        idk_examples = create_idk_dataset(self.forget_raw, self.idk_variation)
        self.idk_dataset = self._build_dataset(idk_examples)
        logger.info(f"IDK dataset prepared with {len(self.idk_dataset)} examples.")

    def _refresh_datasets(self) -> None:
        """Load/refresh forget, retain, and optional IDK datasets."""
        self._prepare_forget_dataset()
        self._prepare_retain_dataset()
        self._prepare_idk_dataset()

    # --------------------------------------------------------------------- #
    # LightningDataModule overrides
    # --------------------------------------------------------------------- #
    def setup(self, stage: Optional[str] = None):
        """
        Load required datasets for the given stage.
        """
        super().setup(stage)
        if stage in (None, "fit"):
            self._refresh_datasets()
        elif stage == "validate":
            self._refresh_datasets()
        return None

    def train_dataloader(self) -> UnlearningDataLoader:
        """
        Primary training loader that pairs forget and retain mini-batches.
        """
        if self.forget_dataset is None or self.retain_dataset is None:
            raise RuntimeError(
                "Forget and retain datasets must be initialized. Call setup() first."
            )

        train_batch_size: int = int(self.train_batch_size or self.batch_size or 1)
        retain_batch_size: int = train_batch_size
        idk_dataset: Optional[QATokenizedDataset] = (
            self.idk_dataset if self.create_idk else None
        )
        return UnlearningDataLoader(
            forget_dataset=self.forget_dataset,
            retain_dataset=self.retain_dataset,
            collator=self._collator,
            train_batch_size=train_batch_size,
            retain_batch_size=retain_batch_size,
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
            retain_shuffle=False,
            idk_dataset=idk_dataset,
            idk_batch_size=train_batch_size,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """
        Validation dataloaders for retain (first) and forget (second) splits.
        """
        if self.retain_dataset is None or self.forget_dataset is None:
            raise RuntimeError(
                "Retain and forget datasets must be initialized. Call setup() first."
            )

        val_batch_size: int = int(self.val_batch_size or self.train_batch_size or 1)
        retain_loader = self.create_dataloader(
            dataset=self.retain_dataset,
            collator=self._collator,
            shuffle=False,
            batch_size=val_batch_size,
            multiprocessing_context=self.multiprocessing_context,
        )
        forget_loader = self.create_dataloader(
            dataset=self.forget_dataset,
            collator=self._collator,
            shuffle=False,
            batch_size=val_batch_size,
            multiprocessing_context=self.multiprocessing_context,
        )
        return [retain_loader, forget_loader]
