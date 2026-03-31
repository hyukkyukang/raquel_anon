"""Specialized dataloader helpers for unlearning workflows."""

from typing import Any, Callable, Dict, Iterator, Optional, cast

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DistributedSampler

from src.utils.logging import get_logger

logger = get_logger("src.training.data.dataloader")

BatchDict = Dict[str, Any]


def build_distributed_sampler(
    dataset: TorchDataset, shuffle: bool
) -> Optional[DistributedSampler]:
    """
    Construct a DistributedSampler when torch.distributed is initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=False,
        )
    return None

class UnlearningDataLoader(DataLoader):
    """
    DataLoader subclass that yields combined forget/retain/(optional) IDK batches.

    The class owns the underlying datasets and exposes a single dataloader to
    Lightning while cycling retain batches and aligning IDK batches to the
    current forget sample IDs.
    """

    def __init__(
        self,
        *,
        forget_dataset: TorchDataset,
        retain_dataset: TorchDataset,
        collator: Callable,
        train_batch_size: int,
        retain_batch_size: int,
        num_workers: int,
        multiprocessing_context: Optional[str],
        retain_shuffle: bool = True,
        idk_dataset: Optional[TorchDataset] = None,
        idk_batch_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            forget_dataset: Dataset providing forget examples (drives step count).
            retain_dataset: Dataset providing retain examples for regularization.
            collator: Callable applied to raw samples to build tensor batches.
            train_batch_size: Batch size for forget dataset.
            retain_batch_size: Batch size for retain dataset.
            num_workers: Number of DataLoader workers.
            multiprocessing_context: Multiprocessing context ("fork"/"spawn"/None).
            retain_shuffle: Whether to shuffle retain dataset each epoch.
            idk_dataset: Optional IDK dataset used by IDK/DPO losses.
            idk_batch_size: Deprecated legacy argument kept for API compatibility.
        """
        self.retain_dataset: TorchDataset = retain_dataset
        self.idk_dataset: Optional[TorchDataset] = idk_dataset
        self.collator = collator
        self.retain_batch_size = retain_batch_size
        self.idk_batch_size = idk_batch_size or train_batch_size
        self.num_workers = num_workers
        self.multiprocessing_context = multiprocessing_context
        self.retain_shuffle = retain_shuffle

        forget_sampler: Optional[DistributedSampler] = build_distributed_sampler(
            dataset=forget_dataset, shuffle=True
        )

        super().__init__(
            dataset=forget_dataset,
            batch_size=train_batch_size,
            shuffle=forget_sampler is None,
            sampler=forget_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context=multiprocessing_context,
        )

        # The retain loader cycles independently; IDK batches are aligned from sample_ids.
        self._retain_loader: DataLoader = self._build_aux_loader(
            dataset=self.retain_dataset,
            batch_size=self.retain_batch_size,
            shuffle=self.retain_shuffle,
        )

    def _build_aux_loader(
        self, dataset: Optional[TorchDataset], batch_size: int, shuffle: bool
    ) -> DataLoader:
        """
        Create a DataLoader for the auxiliary retain dataset.
        """
        if dataset is None:
            raise ValueError("Auxiliary dataset is required to build a dataloader.")

        aux_sampler: Optional[DistributedSampler] = build_distributed_sampler(
            dataset=dataset,
            shuffle=shuffle,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=aux_sampler is None and shuffle,
            sampler=aux_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context=self.multiprocessing_context,
        )

    def __iter__(self) -> Iterator[Dict[str, BatchDict]]:
        """
        Yield mini-batches with aligned forget/retain/(optional) IDK entries.
        """
        return UnlearningDataLoaderIterator(self)

    def __getstate__(self) -> Dict[str, Any]:
        """Strip live iterators and rebuildable loaders before pickling."""
        state = self.__dict__.copy()
        state.pop("_iterator", None)
        state.pop("_retain_loader", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the dataloader and rebuild auxiliary loaders after unpickling."""
        self.__dict__.update(state)
        self._retain_loader = self._build_aux_loader(
            dataset=self.retain_dataset,
            batch_size=self.retain_batch_size,
            shuffle=self.retain_shuffle,
        )


class InfiniteCycleIterator:
    """
    Iterator that yields batches from a loader indefinitely.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self) -> BatchDict:
        try:
            return cast(BatchDict, next(self._iterator))
        except StopIteration:
            self._iterator = iter(self.loader)
            return cast(BatchDict, next(self._iterator))

    def __getstate__(self):
        # Return state without the iterator, as it cannot be pickled
        return {"loader": self.loader}

    def __setstate__(self, state):
        self.loader = state["loader"]
        # Re-initialize iterator
        self._iterator = iter(self.loader)


class UnlearningDataLoaderIterator:
    """
    Iterator for UnlearningDataLoader.
    """

    def __init__(self, loader: UnlearningDataLoader):
        self.loader = loader
        # Get the base DataLoader iterator (for forget dataset)
        # We use the parent class implementation to get the iterator for the forget dataset
        self.forget_iter = DataLoader.__iter__(loader)

        self.retain_iter = InfiniteCycleIterator(loader._retain_loader)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, BatchDict]:
        forget_batch = next(self.forget_iter)
        retain_batch = next(self.retain_iter)

        combined_batch: Dict[str, BatchDict] = {
            "forget": cast(BatchDict, forget_batch),
            "retain": cast(BatchDict, retain_batch),
        }

        if self.loader.idk_dataset is not None:
            sample_ids = forget_batch.get("sample_id")
            if not isinstance(sample_ids, torch.Tensor):
                raise ValueError(
                    "Forget batch must include tensor sample_id values to align IDK data."
                )
            idk_items = [
                self.loader.idk_dataset[int(sample_id)]
                for sample_id in sample_ids.tolist()
            ]
            combined_batch["idk"] = cast(BatchDict, self.loader.collator(idk_items))

        return combined_batch

    def __getstate__(self):
        # Return state without iterators
        return {"loader": self.loader}

    def __setstate__(self, state):
        self.loader = state["loader"]
        # Re-initialize iterators
        self.forget_iter = DataLoader.__iter__(self.loader)
        self.retain_iter = InfiniteCycleIterator(self.loader._retain_loader)
