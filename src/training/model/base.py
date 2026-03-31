"""Base Lightning Module with common functionality."""

import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from torchmetrics.functional.text.rouge import rouge_score
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src.metrics import (
    RougeMetric,
    SemanticAccuracyMetric,
    SemanticMetricConfig,
)
from src.training.artifacts import (
    build_generation_artifact_records,
    write_generation_artifact,
)
from src.utils.generation import build_greedy_generation_kwargs
from src.utils.logging import get_logger
from src.utils.string import sanitize_identifier

logger = get_logger(__name__)


class BaseLightningModule(pl.LightningModule):
    """
    Base Lightning Module with common training functionality.

    Provides:
    - Model and tokenizer management
    - Optimizer and scheduler configuration
    - Common training step structure
    - Checkpointing support
    """

    _GENERATION_BATCH_KEYS: Tuple[str, ...] = (
        "prompt_input_ids",
        "prompt_attention_mask",
        "prompt_length",
        "prompt_text",
        "target_text",
    )

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize base module.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of warmup steps to total steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm

        # Save hyperparameters (excluding model and tokenizer)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        # Validation generation defaults (opt-in via _init_validation_generation).
        self._validation_generation_enabled: bool = False
        self._validation_global_cfg: Optional[Any] = None
        self.val_generation_num: int = 0
        self.val_samples_to_log: int = 0
        self.val_max_new_tokens: int = 0
        self._semantic_metric_config: Optional[SemanticMetricConfig] = None
        self._val_generation_cnt: Dict[int, int] = {}
        self._validation_generations: Dict[int, List[Dict[str, Any]]] = {}
        self.rouge_metrics: List[RougeMetric] = []
        self.semantic_metrics: List[SemanticAccuracyMetric] = []

    @property
    def is_global_zero(self) -> bool:
        """Check if current process is the global zero rank."""
        if self.trainer is None:
            return True
        return getattr(self.trainer, "is_global_zero", True)

    @property
    def val_dataloader_num(self) -> int:
        """Return the number of validation dataloaders."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return 0

        val_datasets = getattr(datamodule, "val_datasets", None)
        if val_datasets is not None:
            return len(val_datasets)

        val_loader = datamodule.val_dataloader()
        if val_loader is None:
            return 0
        if isinstance(val_loader, list):
            return len(val_loader)
        return 1

    @property
    def log_dir(self) -> str:
        """Return the logging directory."""

        # TODO: Need to check this method
        def _coerce_to_path(value: Any) -> Optional[Path]:
            if value is None:
                return None
            if isinstance(value, Path):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                return Path(stripped) if stripped else None
            try:
                return Path(value)
            except TypeError:
                return None

        if self.logger is not None:
            for attr in ("log_dir", "dir"):
                log_attr = getattr(self.logger, attr, None)
                log_path = _coerce_to_path(log_attr)
                if log_path is not None:
                    return str(log_path)

        trainer_log_dir = _coerce_to_path(getattr(self.trainer, "log_dir", None))
        if trainer_log_dir is not None:
            return str(trainer_log_dir)

        default_root_dir = _coerce_to_path(
            getattr(self.trainer, "default_root_dir", None)
        )
        if default_root_dir is not None:
            return str(default_root_dir)
        return ""

    @functools.cache
    def _get_val_subset_name(self, dataloader_idx: int) -> str:
        """Return the subset name of the validation dataset at the given index."""
        subset_name = self.trainer.datamodule.val_subset_names[dataloader_idx]  # type: ignore
        return sanitize_identifier(subset_name)

    @functools.cache
    def _get_val_metric_key(
        self,
        metric_name: str,
        dataloader_idx: int,
    ) -> str:
        """Construct a metric key scoped to the validation subset."""
        subset_label = self._get_val_subset_name(dataloader_idx)
        return f"val/{subset_label}/{metric_name}"

    def forward(self, **inputs) -> Any:
        """Forward pass through the model."""
        return self.model(**inputs)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Prepare optimizer parameters
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )

        # Calculate total steps
        if self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * self.warmup_ratio)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict."""
        if stage == "fit":
            # Lightning warns if modules start in eval mode; flip to train early.
            self.model.train()

    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step. Used for gradient clipping."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    def save_model(self, output_dir: str):
        """
        Save model and tokenizer to directory.

        Args:
            output_dir: Directory to save to
        """
        logger.info("Saving model and tokenizer to %s", output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")

    # ------------------------------------------------------------------ #
    # Validation generation helpers (opt-in)
    # ------------------------------------------------------------------ #
    def _init_validation_generation(
        self,
        *,
        cfg: Any,
        val_samples_to_log: int,
        val_max_new_tokens: int,
    ) -> None:
        """
        Enable generation-based validation utilities shared across modules.

        Args:
            cfg: Run configuration containing evaluation settings.
            val_samples_to_log: Number of samples to retain per dataloader.
            val_max_new_tokens: Max new tokens to decode during validation.
        """
        evaluation_cfg: Any = getattr(cfg, "evaluation", None)
        semantic_cfg: Any = getattr(evaluation_cfg, "semantic_equivalence", {}) or {}

        self._validation_generation_enabled = True
        self._validation_global_cfg = cfg
        self.val_samples_to_log = max(int(val_samples_to_log), 0)
        self.val_max_new_tokens = max(int(val_max_new_tokens), 1)
        self._semantic_metric_config = SemanticMetricConfig(**semantic_cfg)
        self._val_generation_cnt = {}
        self._validation_generations = {}
        self.rouge_metrics = []
        self.semantic_metrics = []

    def _validation_dataloader_count(self) -> int:
        """
        Return number of validation dataloaders.

        Subclasses enabling validation generation must override this.
        """
        raise NotImplementedError(
            "Modules using validation generation must implement "
            "_validation_dataloader_count()."
        )

    def _get_validation_subset_name(self, dataloader_idx: int) -> str:
        """
        Return sanitized subset name for validation logging.

        Subclasses may override to customize naming.
        """
        return self._get_val_subset_name(dataloader_idx)

    def _build_validation_metric_key(
        self,
        metric_name: str,
        dataloader_idx: int,
    ) -> str:
        """Construct standardized validation metric key."""
        subset_name = self._get_validation_subset_name(dataloader_idx)
        return f"val/{subset_name}/{metric_name}"

    def _remaining_generations(self, dataloader_idx: int) -> int:
        """Compute remaining qualitative generations allowed."""
        if not self._validation_generation_enabled or not getattr(
            self, "val_generation_num", 0
        ):
            return 0
        generated_so_far = self._val_generation_cnt.get(dataloader_idx, 0)
        remaining = max(self.val_generation_num - generated_so_far, 0)
        if not self.is_global_zero:
            return 0
        return remaining

    def _build_generation_batch(
        self,
        batch: Dict[str, Any],
        limit: int,
    ) -> Optional[Dict[str, Any]]:
        """Slice generation-relevant fields down to the requested limit."""
        if not self._validation_generation_enabled or limit <= 0:
            return None
        filtered: Dict[str, Any] = {}
        for key in self._GENERATION_BATCH_KEYS:
            value = batch.get(key)
            if value is None:
                return None
            filtered[key] = value[:limit]
        return filtered

    def _generate_validation_texts(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Decode prompts, targets, and model generations for a mini-batch."""
        prompt_inputs: torch.Tensor = batch["prompt_input_ids"]  # type: ignore[assignment]
        prompt_attention_mask: torch.Tensor = batch[
            "prompt_attention_mask"
        ]  # type: ignore[assignment]
        prompt_lengths_tensor: torch.Tensor = batch[
            "prompt_length"
        ]  # type: ignore[assignment]
        prompts: List[str] = batch["prompt_text"]
        targets: List[str] = [item.strip() for item in batch["target_text"]]

        batch_size = prompt_inputs.size(0)
        generation_kwargs: Dict[str, Any] = {
            "attention_mask": prompt_attention_mask,
            **build_greedy_generation_kwargs(
                self.model,
                self.tokenizer,
                input_length=int(prompt_inputs.size(1)),
                max_new_tokens=self.val_max_new_tokens,
                use_cache=True,
            ),
        }

        generated = self.model.generate(  # type: ignore[assignment]
            input_ids=prompt_inputs,
            **generation_kwargs,
        )

        predictions: List[str] = []
        for idx in range(batch_size):
            prompt_len = int(prompt_lengths_tensor[idx].item())
            generated_ids = generated[idx].detach().cpu()
            answer_start = min(prompt_len, generated_ids.size(0))
            prediction_ids = generated_ids[answer_start:]
            prediction_text = self.tokenizer.decode(
                prediction_ids, skip_special_tokens=True
            ).strip()
            predictions.append(prediction_text)

        return prompts, targets, predictions

    def _collect_validation_generations(
        self,
        prompts: List[str],
        targets: List[str],
        predictions: List[str],
        dataloader_idx: int,
    ) -> None:
        """Cache qualitative validation samples for later visualization."""
        if not self._validation_generation_enabled:
            return None
        subset_name = self._get_validation_subset_name(dataloader_idx)
        generations = self._validation_generations.setdefault(dataloader_idx, [])

        for prompt, target, prediction in zip(prompts, targets, predictions):
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
                    "subset": subset_name,
                }
            )
            if 0 < self.val_samples_to_log <= len(generations):
                break
        return None

    def _update_rouge_metric(
        self,
        predictions: List[str],
        targets: List[str],
        dataloader_idx: int,
    ) -> None:
        """Accumulate ROUGE statistics for later aggregation."""
        if not self._validation_generation_enabled or dataloader_idx >= len(
            self.rouge_metrics
        ):
            return None
        self.rouge_metrics[dataloader_idx].update(predictions, targets)
        return None

    def _update_semantic_metric(
        self,
        prompts: List[str],
        predictions: List[str],
        targets: List[str],
        dataloader_idx: int,
    ) -> None:
        """Accumulate semantic-accuracy statistics if enabled."""
        if (
            not self._validation_generation_enabled
            or not self._semantic_metric_config
            or not self._semantic_metric_config.enabled
            or dataloader_idx >= len(self.semantic_metrics)
        ):
            return None
        self.semantic_metrics[dataloader_idx].update(prompts, predictions, targets)
        return None

    def _maybe_run_validation_generation(
        self,
        batch: Dict[str, Any],
        dataloader_idx: int,
    ) -> None:
        """Optionally run qualitative generation pass for a validation batch."""
        if not self._validation_generation_enabled:
            return None
        remaining = self._remaining_generations(dataloader_idx)
        generation_batch = self._build_generation_batch(batch, remaining)
        if generation_batch is None:
            return None

        prompts, targets, predictions = self._generate_validation_texts(
            generation_batch
        )
        self._update_rouge_metric(predictions, targets, dataloader_idx)
        self._update_semantic_metric(
            prompts,
            predictions,
            targets,
            dataloader_idx,
        )
        self._collect_validation_generations(
            prompts,
            targets,
            predictions,
            dataloader_idx,
        )

        if self.val_generation_num > 0:
            current_total = self._val_generation_cnt.get(dataloader_idx, 0)
            self._val_generation_cnt[dataloader_idx] = current_total + len(predictions)
        return None

    def _log_validation_metrics(self) -> None:
        """Aggregate and log metrics collected over the validation epoch."""
        if not self._validation_generation_enabled:
            return None

        for dataloader_idx, rouge_metric in enumerate(self.rouge_metrics):
            rouge_scores = rouge_metric.compute_recall()
            for metric_name, value in rouge_scores.items():
                log_name = self._build_validation_metric_key(
                    metric_name,
                    dataloader_idx,
                )
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

        if (
            self._semantic_metric_config
            and self._semantic_metric_config.enabled
            and self.semantic_metrics
            and self.is_global_zero
        ):
            semantic_results_by_idx: Dict[int, Dict[str, Tuple[str, str, bool]]] = {}
            for dataloader_idx, semantic_metric in enumerate(self.semantic_metrics):
                semantic_accuracy = semantic_metric.compute()
                if semantic_accuracy is None:
                    continue
                log_name = self._build_validation_metric_key(
                    "semantic_accuracy",
                    dataloader_idx,
                )
                self.log(
                    log_name,
                    semantic_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                semantic_results_by_idx[dataloader_idx] = (
                    semantic_metric.get_last_eval_results_in_dic()
                )
                semantic_metric.reset()
            for dataloader_idx, generations in self._validation_generations.items():
                if generations:
                    self._persist_validation_generation_artifact(
                        dataloader_idx,
                        semantic_results_by_idx.get(dataloader_idx),
                    )
        elif self.is_global_zero:
            for dataloader_idx, generations in self._validation_generations.items():
                if generations:
                    self._persist_validation_generation_artifact(
                        dataloader_idx,
                        None,
                    )

        for metric in self.semantic_metrics:
            metric.reset()
        return None

    def _persist_validation_generation_artifact(
        self,
        dataloader_idx: int,
        semantic_eval_results: Optional[Dict[str, Tuple[str, str, bool]]],
    ) -> None:
        """Persist qualitative validation samples to a local artifact file."""
        if not self._validation_generation_enabled:
            return None

        generations = self._validation_generations.get(dataloader_idx, [])
        if not generations:
            return None

        subset_name = self._get_validation_subset_name(dataloader_idx)
        records = build_generation_artifact_records(
            generations,
            semantic_eval_results=semantic_eval_results,
        )
        artifact_path = write_generation_artifact(
            self.log_dir or ".",
            category="validation_generations",
            subset_name=subset_name,
            epoch=self.current_epoch,
            records=records,
        )
        logger.info(
            "Saved %d validation generation artifact record(s) to %s",
            len(records),
            artifact_path,
        )
        return None

    @torch.no_grad()
    def on_validation_start(self) -> None:
        """Prepare validation metric accumulators each epoch."""
        super().on_validation_start()
        if not self._validation_generation_enabled:
            return None

        dataloader_count = max(self._validation_dataloader_count(), 1)

        while len(self.rouge_metrics) < dataloader_count:
            self.rouge_metrics.append(RougeMetric(rouge_keys=("rougeL",)))

        semantic_enabled = (
            self._semantic_metric_config is not None
            and self._semantic_metric_config.enabled
        )
        if semantic_enabled:
            while len(self.semantic_metrics) < dataloader_count:
                self.semantic_metrics.append(
                    SemanticAccuracyMetric(
                        config=self._semantic_metric_config,  # type: ignore[arg-type]
                        global_cfg=self._validation_global_cfg,
                    )
                )

        for metric in self.rouge_metrics:
            metric.reset()
        for metric in self.semantic_metrics:
            metric.reset()

        self._val_generation_cnt = {idx: 0 for idx in range(dataloader_count)}
        self._validation_generations = {idx: [] for idx in range(dataloader_count)}
        return None

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        """Log aggregated validation metrics once per epoch."""
        super().on_validation_epoch_end()
        if not self._validation_generation_enabled:
            return None
        self._log_validation_metrics()
        return None
