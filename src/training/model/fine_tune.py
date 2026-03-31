"""Fine-tuning Lightning Module."""

from typing import Any, Dict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.batch import extract_model_inputs
from src.utils.logging import get_logger

from .base import BaseLightningModule

logger = get_logger(__name__)


class FinetuneModule(BaseLightningModule):
    """
    Lightning Module for standard fine-tuning.

    This module performs standard supervised learning on a dataset,
    minimizing cross-entropy loss on answer tokens.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: Any,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        val_samples_to_log: int = 5,
        val_max_new_tokens: int = 64,
    ):
        """
        Initialize FinetuneModule.

        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for the model
            cfg: Full Hydra/DictConfig for the current run
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of warmup steps to total steps
            max_grad_norm: Maximum gradient norm for clipping
            val_samples_to_log: Number of validation generations to persist each epoch
            val_max_new_tokens: Maximum number of tokens to generate during validation sampling
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
        )
        self.train_batch_size = cfg.training.train_batch_size
        self.val_batch_size = cfg.training.val_batch_size
        self.val_generation_num = cfg.training.val_generation_num
        self._init_validation_generation(
            cfg=cfg,
            val_samples_to_log=val_samples_to_log,
            val_max_new_tokens=val_max_new_tokens,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for fine-tuning.

        Args:
            batch: Batch of tokenized inputs
            batch_idx: Index of current batch

        Returns:
            Loss value
        """
        model_inputs = extract_model_inputs(batch)
        outputs = self.model(**model_inputs)
        loss = outputs.loss

        # Log metrics
        self.log(
            "train_loss",
            loss,
            batch_size=self.train_batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True)  # type: ignore

        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Batch of tokenized inputs
            batch_idx: Index of current batch

        Returns:
            Dictionary with loss
        """
        # Get the dataset name for better logging
        dataset_name: str = self._get_val_subset_name(dataloader_idx)

        # Get batch inputs and compute loss
        model_inputs = extract_model_inputs(batch)
        outputs = self.model(**model_inputs)
        loss = outputs.loss

        loss_key = f"val_{dataset_name}_loss"

        # Log metrics
        self.log(
            loss_key,
            loss,
            batch_size=self.val_batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self._maybe_run_validation_generation(batch, dataloader_idx)

        return {loss_key: loss}

    def _validation_dataloader_count(self) -> int:
        """Return number of validation dataloaders available."""
        return self.val_dataloader_num
