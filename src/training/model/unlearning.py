"""Unlearning Lightning Module with support for all 8 methods."""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import RougeMetric, SemanticAccuracyMetric
from src.training.artifacts import (
    build_generation_artifact_records,
    write_generation_artifact,
)
from src.training.batch import extract_model_inputs
from src.training.data.pl_module.unlearning import UnlearningDataModule
from src.training.loss import (
    get_regularization_loss,
    get_unlearning_loss,
)
from src.training.methods import check_unlearning_method
from src.utils.logging import get_logger
from src.utils.string import sanitize_identifier

from .base import BaseLightningModule

logger = get_logger(__name__)


class UnlearningModule(BaseLightningModule):
    """
    Lightning Module for machine unlearning with 8 different methods.

    Supports combinations of:
    - Unlearning losses: Gradient Ascent (GA), NPO, IDK, DPO
    - Regularization losses: Grad Descent (GD), KL Divergence (KL)

    Methods:
    - ga_gd: Gradient Ascent + Grad Descent
    - ga_kl: Gradient Ascent + KL Divergence
    - npo_gd: NPO + Grad Descent
    - npo_kl: NPO + KL Divergence
    - idk_gd: IDK + Grad Descent
    - idk_kl: IDK + KL Divergence
    - dpo_gd: DPO + Grad Descent
    - dpo_kl: DPO + KL Divergence
    """

    def __init__(
        self,
        cfg: Any,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        datamodule: UnlearningDataModule,
        unlearning_method: str,
        regularization_method: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        reference_model: Optional[PreTrainedModel] = None,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        idk_variation: str = "random",
        val_samples_to_log: int = 20,
        val_max_new_tokens: int = 64,
    ):
        """
        Initialize UnlearningModule.

        Args:
            cfg: Full Hydra/DictConfig for the current run
            model: Model to unlearn (typically a fine-tuned model)
            tokenizer: Tokenizer for the model
            datamodule: UnlearningDataModule with forget/retain/IDK data
            unlearning_method: Unlearning method (ga, npo, idk, dpo)
            regularization_method: Regularization method (gd, kl)
            alpha: Weight for unlearning loss component
            beta: Weight for regularization loss component
            reference_model: Reference model for KL/DPO (typically base model)
            learning_rate: Learning rate (typically lower than fine-tuning)
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            idk_variation: Type of IDK response variation
            val_samples_to_log: Number of validation generations to cache per epoch
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

        self.datamodule = datamodule
        self.unlearning_method = unlearning_method
        self.regularization_method = regularization_method
        self.alpha = alpha
        self.beta = beta
        self.reference_model = reference_model
        self.idk_variation = idk_variation
        self.unlearning_loss_beta = cfg.unlearning.get("dpo_beta")

        # Validation/generation-specific configuration extracted from Hydra config
        training_cfg: Any = getattr(cfg, "training", None)
        val_batch_size_cfg: int = int(getattr(training_cfg, "val_batch_size", 1) or 1)
        val_generation_cfg: int = int(
            getattr(training_cfg, "val_generation_num", 0) or 0
        )
        self.val_batch_size: int = max(val_batch_size_cfg, 1)
        self.val_generation_num: int = max(val_generation_cfg, 0)
        self._init_validation_generation(
            cfg=cfg,
            val_samples_to_log=val_samples_to_log,
            val_max_new_tokens=val_max_new_tokens,
        )
        self._validation_subset_names: List[str] = self._derive_validation_subset_names(
            datamodule
        )
        retain_subset_raw = getattr(datamodule, "retain_subset_name", None) or "retain"
        self.retain_subset_name = sanitize_identifier(retain_subset_raw)

        # Track seen retain indices for splitting validation metrics
        self.seen_retain_indices: Set[int] = set()
        self.split_rouge_metrics: Dict[str, RougeMetric] = {}
        self.split_semantic_metrics: Dict[str, SemanticAccuracyMetric] = {}
        self.split_generations: Dict[str, List[Dict[str, Any]]] = {}
        self.split_generation_cnt: Dict[str, int] = {}

        # Check the methods are valid
        check_unlearning_method(unlearning_method, regularization_method)

        logger.info(
            f"Initializing UnlearningModule with unlearning_method={unlearning_method}, regularization_method={regularization_method} (alpha={alpha:.2f}, beta={beta:.2f})"
        )

        # Initialize loss functions
        self._init_losses()

        # Save hyperparameters
        self.save_hyperparameters(
            ignore=["model", "tokenizer", "datamodule", "reference_model", "cfg"]
        )

    def _init_losses(self):
        """Initialize unlearning and regularization loss functions."""
        # Unlearning loss - use registry to avoid if conditions
        self.unlearning_loss = get_unlearning_loss(
            name=self.unlearning_method,
            reference_model=self.reference_model,
            tokenizer=self.tokenizer,
            beta=self.unlearning_loss_beta,
            idk_variation=self.idk_variation,
            max_length=getattr(self.datamodule, "max_length", 1024),
        )

        # Regularization loss - use registry to avoid if conditions
        self.regularization_loss = get_regularization_loss(
            name=self.regularization_method,
            reference_model=self.reference_model,
        )

        logger.info(
            f"Initialized {self.unlearning_method.capitalize()} unlearning loss"
        )
        logger.info(
            f"Initialized {self.regularization_method.capitalize()} regularization loss"
        )

    def _derive_validation_subset_names(
        self, datamodule: UnlearningDataModule
    ) -> List[str]:
        """
        Build sanitized validation subset labels for logging.

        Returns:
            List of subset names aligned with validation dataloaders.
        """
        subset_candidates: List[str] = []
        retain_subset: Optional[str] = getattr(datamodule, "retain_subset_name", None)
        forget_subset: Optional[str] = getattr(datamodule, "forget_subset_name", None)

        subset_candidates.append(retain_subset or "retain")
        subset_candidates.append(forget_subset or "forget")

        sanitized_names: List[str] = [
            sanitize_identifier(name) for name in subset_candidates
        ]
        return sanitized_names

    def _get_validation_subset_name(self, dataloader_idx: int) -> str:
        """
        Get the sanitized subset label for a validation dataloader index.

        Args:
            dataloader_idx: Index provided by Lightning during validation.

        Returns:
            Sanitized subset label (defaults to retain_{idx} if out of range).
        """
        if 0 <= dataloader_idx < len(self._validation_subset_names):
            return self._validation_subset_names[dataloader_idx]
        fallback_name: str = f"retain_{dataloader_idx}"
        return sanitize_identifier(fallback_name)

    def on_train_epoch_start(self) -> None:
        """Reset seen retain indices at the start of each epoch."""
        super().on_train_epoch_start()
        self.seen_retain_indices.clear()

    def training_step(
        self, batch: Dict[str, Dict[str, Any]], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for unlearning.

        Args:
            batch: Combined batch containing forget and retain tensors.
            batch_idx: Index of current batch

        Returns:
            Combined loss value
        """
        if "forget" not in batch or "retain" not in batch:
            raise ValueError(
                "Combined batch must contain 'forget' and 'retain' entries."
            )

        # Batches are pre-aligned by the datamodule to keep preprocessing centralized.
        forget_batch: Dict[str, Any] = batch["forget"]
        retain_batch: Dict[str, Any] = batch["retain"]
        idk_batch: Optional[Dict[str, Any]] = batch.get("idk")

        # Track seen retain samples
        if "sample_id" in retain_batch:
            # sample_id is a tensor of indices
            ids = retain_batch["sample_id"]
            if isinstance(ids, torch.Tensor):
                self.seen_retain_indices.update(ids.tolist())

        # Compute unlearning loss
        # All losses accept forget_examples_raw as optional parameter
        # IDK and DPO require it, while GA and NPO ignore it
        forget_raw = getattr(self.trainer.datamodule, "forget_raw", None)
        unlearn_loss = self.unlearning_loss(
            self.model,
            forget_batch,
            forget_raw,
            idk_batch=idk_batch,
        )

        # Compute regularization loss
        reg_loss = self.regularization_loss(self.model, retain_batch)

        # Combine losses
        total_loss = self.alpha * unlearn_loss + self.beta * reg_loss

        # Log unlearning loss (flip sign for GA for better visualization)
        log_unlearn_loss = unlearn_loss
        log_total_loss = total_loss

        if self.unlearning_method in ["ga", "gradient_ascent"]:
            log_unlearn_loss = -unlearn_loss
            # For logging total_loss, we also want to use the positive unlearning component for GA
            # to maintain consistent visualization (magnitude of losses)
            log_total_loss = self.alpha * log_unlearn_loss + self.beta * reg_loss

        # Log metrics
        self.log(
            "train_loss",
            log_total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=forget_batch["input_ids"].size(0),
        )
        self.log(
            "unlearn_loss",
            log_unlearn_loss,
            on_step=False,
            on_epoch=True,
            batch_size=forget_batch["input_ids"].size(0),
        )
        self.log(
            "reg_loss",
            reg_loss,
            on_step=False,
            on_epoch=True,
            batch_size=retain_batch["input_ids"].size(0),
        )
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],  # type: ignore
            on_step=True,
        )

        return total_loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step on retain set with generation-based evaluation.

        Args:
            batch: Batch from retain dataset or forget dataset
            batch_idx: Index of current batch
            dataloader_idx: Validation dataloader index provided by Lightning

        Returns:
            Dictionary with validation loss keyed by subset.
        """
        subset_name: str = self._get_validation_subset_name(dataloader_idx)
        loss_key: str = f"val_{subset_name}_loss"

        # Special handling for retain set to split seen/unseen
        if subset_name == self.retain_subset_name and self.seen_retain_indices:
            return self._validation_step_split(batch, batch_idx, subset_name)

        # Standard validation for other sets
        outputs = self.model(**extract_model_inputs(batch))
        val_loss = outputs.loss

        # Log base validation loss for this subset.
        self.log(
            loss_key,
            val_loss,
            batch_size=self.val_batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self._maybe_run_validation_generation(batch, dataloader_idx)

        return {loss_key: val_loss}

    def _validation_step_split(
        self, batch: Dict[str, Any], batch_idx: int, subset_name: str
    ) -> Dict[str, torch.Tensor]:
        """Handle validation step with splitting for seen/unseen retain samples."""
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            # Fallback if no sample IDs
            return self.validation_step(batch, batch_idx, 0)

        seen_mask = torch.tensor(
            [sid.item() in self.seen_retain_indices for sid in sample_ids],
            device=batch["input_ids"].device,
            dtype=torch.bool,
        )
        unseen_mask = ~seen_mask

        results = {}

        for mask, suffix in [(seen_mask, "seen"), (unseen_mask, "unseen")]:
            if not mask.any():
                continue

            # Filter batch
            sub_batch = {
                k: (
                    v[mask]
                    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0)
                    else v
                )
                for k, v in batch.items()
            }

            # Handle list fields that might be in the batch (like text)
            for k, v in batch.items():
                if isinstance(v, list) and len(v) == mask.size(0):
                    sub_batch[k] = [item for item, m in zip(v, mask) if m]

            outputs = self.model(**extract_model_inputs(sub_batch))
            loss = outputs.loss

            log_name = f"val_{subset_name}_{suffix}_loss"
            self.log(
                log_name,
                loss,
                batch_size=mask.sum(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            results[log_name] = loss

            # Run generation with suffix
            self._run_split_validation_generation(sub_batch, f"{subset_name}_{suffix}")

        return results

    def _validation_dataloader_count(self) -> int:
        """Return number of validation dataloaders tracked by this module."""
        return max(len(self._validation_subset_names), 1)

    def _run_split_validation_generation(self, batch: Dict[str, Any], key: str) -> None:
        """Run validation generation for a specific split key."""
        if not self._validation_generation_enabled:
            return

        # Determine remaining generations for this split
        generated_so_far = self.split_generation_cnt.get(key, 0)
        remaining = max(self.val_generation_num - generated_so_far, 0)
        if remaining <= 0 or not self.is_global_zero:
            return

        generation_batch = self._build_generation_batch(batch, remaining)
        if generation_batch is None:
            return

        prompts, targets, predictions = self._generate_validation_texts(
            generation_batch
        )

        # Update ROUGE
        if key in self.split_rouge_metrics:
            self.split_rouge_metrics[key].update(predictions, targets)

        # Update Semantic
        if key in self.split_semantic_metrics:
            self.split_semantic_metrics[key].update(prompts, predictions, targets)

        # Collect generations
        generations = self.split_generations.setdefault(key, [])
        for prompt, target, prediction in zip(prompts, targets, predictions):
            from torchmetrics.functional.text.rouge import rouge_score

            rouge_scores = rouge_score(
                prediction,
                target,
                rouge_keys=("rougeL",),
                use_stemmer=True,
            )
            rouge_l_value = float(rouge_scores["rougeL_fmeasure"].item())
            generations.append(
                {
                    "prompt": prompt,
                    "target": target,
                    "prediction": prediction,
                    "rougeL_fmeasure": rouge_l_value,
                    "subset": key,
                }
            )
            if 0 < self.val_samples_to_log <= len(generations):
                break

        self.split_generation_cnt[key] = generated_so_far + len(predictions)

    @torch.no_grad()
    def on_validation_start(self) -> None:
        """Prepare validation metrics."""
        # Sync seen indices across ranks if distributed
        if self.trainer.world_size > 1:
            local_indices = list(self.seen_retain_indices)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                object_list = [None for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(object_list, local_indices)
                all_indices = set()
                for indices in object_list:
                    if indices:
                        all_indices.update(indices)
                self.seen_retain_indices = all_indices

        super().on_validation_start()

        # Setup split metrics for retain set
        retain_name = self.retain_subset_name
        for suffix in ["seen", "unseen"]:
            key = f"{retain_name}_{suffix}"
            self.split_rouge_metrics[key] = RougeMetric(rouge_keys=("rougeL",))
            self.split_generation_cnt[key] = 0
            self.split_generations[key] = []

            if self._semantic_metric_config and self._semantic_metric_config.enabled:
                self.split_semantic_metrics[key] = SemanticAccuracyMetric(
                    config=self._semantic_metric_config,
                    global_cfg=self._validation_global_cfg,
                )

            # Reset
            self.split_rouge_metrics[key].reset()
            if key in self.split_semantic_metrics:
                self.split_semantic_metrics[key].reset()

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        """Log metrics."""
        super().on_validation_epoch_end()

        if not self._validation_generation_enabled:
            return

        for key, rouge_metric in self.split_rouge_metrics.items():
            rouge_scores = rouge_metric.compute_recall()
            for metric_name, value in rouge_scores.items():
                log_name = f"val/{key}/{metric_name}"
                self.log(
                    log_name,
                    float(value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=metric_name.endswith("rougeL_recall"),
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
            rouge_metric.reset()

        semantic_results_by_key: Dict[str, Dict[str, Tuple[str, str, bool]]] = {}
        for key, semantic_metric in self.split_semantic_metrics.items():
            semantic_accuracy = semantic_metric.compute()
            if semantic_accuracy is not None:
                log_name = f"val/{key}/semantic_accuracy"
                self.log(
                    log_name,
                    semantic_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                semantic_results_by_key[key] = (
                    semantic_metric.get_last_eval_results_in_dic()
                )
            semantic_metric.reset()

        if self.is_global_zero:
            for key, generations in self.split_generations.items():
                if generations:
                    self._persist_split_generation_artifact(
                        key,
                        semantic_results_by_key.get(key),
                    )

    def _persist_split_generation_artifact(
        self,
        key: str,
        semantic_eval_results: Optional[Dict[str, Tuple[str, str, bool]]],
    ) -> None:
        """Persist split qualitative samples to a local artifact file."""
        generations = self.split_generations.get(key, [])
        if not generations:
            return

        records = build_generation_artifact_records(
            generations,
            semantic_eval_results=semantic_eval_results,
        )
        artifact_path = write_generation_artifact(
            self.log_dir or ".",
            category="split_generations",
            subset_name=key,
            epoch=self.current_epoch,
            records=records,
        )
        logger.info(
            "Saved %d split generation artifact record(s) to %s",
            len(records),
            artifact_path,
        )
