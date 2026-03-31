"""Evaluation callbacks for unlearning assessment."""

import json
import os
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from src.evaluation import (
    MUSEEvaluator,
    RAQUELEvalConfig,
    build_muse_core_metric_logs,
    build_raquel_metric_logs,
    evaluate_muse_core_metrics,
    evaluate_raquel_splits,
    load_evaluation_data,
    load_raquel_examples,
    resolve_generation_device,
)
from src.metrics import SemanticMetricConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MUSEEvaluationCallback(Callback):
    """
    Callback to run MUSE core metrics evaluation during training.

    This callback runs MUSE core metrics (1-4 only) for efficient evaluation:
    1. No Verbatim Memorization (always)
    2. No Knowledge Memorization (if paraphrased data provided)
    3. No Privacy Leakage (if non-training data provided)
    4. Utility Preservation (always)

    Note: Metrics 5 (Scalability) and 6 (Sustainability) are excluded as they are
    computationally expensive. Use separate scripts for comprehensive evaluation.
    """

    def __init__(
        self,
        forget_path: str,
        retain_path: str,
        output_dir: str,
        paraphrased_path: Optional[str] = None,
        non_training_path: Optional[str] = None,
        run_on_train_end: bool = True,
        run_on_epoch_end: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize MUSE evaluation callback.

        Args:
            forget_path: Forget-set dataset path for evaluation
            retain_path: Retain-set dataset path for utility assessment
            output_dir: Directory to save evaluation results
            paraphrased_path: Optional paraphrased dataset path for knowledge memorization
            non_training_path: Optional non-training dataset path for privacy leakage
            run_on_train_end: Whether to run evaluation at end of training
            run_on_epoch_end: Whether to run evaluation at end of each epoch
            device: Device for evaluation ('cuda' or 'cpu')
        """
        super().__init__()
        self.forget_path = forget_path
        self.retain_path = retain_path
        self.paraphrased_path = paraphrased_path
        self.non_training_path = non_training_path
        self.output_dir = output_dir
        self.run_on_train_end = run_on_train_end
        self.run_on_epoch_end = run_on_epoch_end
        self.device = device
        self._forget_examples: Optional[List[Dict[str, Any]]] = None
        self._retain_examples: Optional[List[Dict[str, Any]]] = None
        self._paraphrased_examples: Optional[List[Dict[str, Any]]] = None
        self._non_training_examples: Optional[List[Dict[str, Any]]] = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _ensure_examples_loaded(self) -> None:
        """Load MUSE datasets once on first evaluation run."""
        if self._forget_examples is not None and self._retain_examples is not None:
            return

        (
            self._forget_examples,
            self._retain_examples,
            self._paraphrased_examples,
            self._non_training_examples,
        ) = load_evaluation_data(
            self.forget_path,
            self.retain_path,
            self.paraphrased_path,
            self.non_training_path,
        )
        logger.info(
            "Loaded MUSE callback datasets (forget=%d, retain=%d, paraphrased=%d, non_training=%d)",
            len(self._forget_examples),
            len(self._retain_examples),
            len(self._paraphrased_examples) if self._paraphrased_examples else 0,
            len(self._non_training_examples) if self._non_training_examples else 0,
        )

    def _run_evaluation(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """
        Run MUSE core metrics (1-4) evaluation.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module being trained
            stage: Stage identifier (e.g., 'epoch_0', 'epoch_1', 'final')
        """
        logger.info("=" * 60)
        logger.info("Running MUSE Core Metrics Evaluation at %s", stage)
        logger.info("=" * 60)

        self._ensure_examples_loaded()

        # Create evaluator
        evaluator = MUSEEvaluator(
            model=pl_module.model,
            tokenizer=pl_module.tokenizer,
            device=resolve_generation_device(pl_module.model, self.device),
        )

        results = evaluate_muse_core_metrics(
            evaluator,
            self._forget_examples or [],
            self._retain_examples or [],
            self._paraphrased_examples,
            self._non_training_examples,
        )

        # Add metadata
        results["metadata"] = {
            "stage": stage,
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch if trainer.current_epoch is not None else None,
            "metrics_evaluated": list(results.keys()),
        }

        # Save results
        output_file = os.path.join(self.output_dir, f"muse_results_{stage}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("MUSE results saved to %s", output_file)

        # Log all metrics to Lightning logger (TensorBoard/WandB)
        if trainer.logger is not None:
            metrics_to_log = build_muse_core_metric_logs(results, prefix=stage)
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)

        # Print summary
        logger.info("=" * 60)
        logger.info("MUSE Core Metrics Summary (%s)", stage)
        logger.info("=" * 60)
        logger.info(
            "[Metric 1] Verbatim Memorization ROUGE-L:  %.4f (lower is better)",
            results["verbatim"]["verbatim_rouge_mean"],
        )
        if "knowledge" in results:
            logger.info(
                "[Metric 2] Knowledge Memorization ROUGE-L:  %.4f (lower is better)",
                results["knowledge"]["knowledge_rouge_mean"],
            )
        if "privacy" in results:
            logger.info(
                "[Metric 3] Privacy Leakage (MIA Accuracy): %.4f (~0.50 is ideal)",
                results["privacy"]["mia_accuracy"],
            )
        logger.info(
            "[Metric 4] Utility Preservation ROUGE-L:    %.4f (higher is better)",
            results["utility"]["utility_rouge_mean"],
        )
        logger.info("=" * 60)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of training.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if self.run_on_train_end:
            self._run_evaluation(trainer, pl_module, "final")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called at the end of each training epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if self.run_on_epoch_end and trainer.current_epoch is not None:
            stage = f"epoch_{trainer.current_epoch}"
            self._run_evaluation(trainer, pl_module, stage)


class RAQUELEvaluationCallback(Callback):
    """
    Callback to evaluate RAQUEL affected/unaffected sets during training.
    """

    def __init__(
        self,
        affected_path: str,
        unaffected_path: str,
        output_dir: str,
        *,
        batch_size: int = 8,
        max_new_tokens: int = 64,
        max_prompt_length: Optional[int] = None,
        max_examples: Optional[int] = None,
        run_on_train_end: bool = True,
        run_on_epoch_end: bool = False,
        device: Optional[str] = None,
        semantic_cfg: Optional[Dict[str, Any]] = None,
        save_predictions: bool = False,
    ) -> None:
        super().__init__()
        self.affected_path = affected_path
        self.unaffected_path = unaffected_path
        self.output_dir = output_dir
        self.eval_config = RAQUELEvalConfig(
            batch_size=max(int(batch_size), 1),
            max_new_tokens=max(int(max_new_tokens), 1),
            max_prompt_length=(
                int(max_prompt_length) if max_prompt_length else None
            ),
            max_examples=int(max_examples) if max_examples else None,
            save_predictions=bool(save_predictions),
        )
        self.run_on_train_end = run_on_train_end
        self.run_on_epoch_end = run_on_epoch_end
        self.device = device
        self._affected_examples: Optional[List[Dict[str, str]]] = None
        self._unaffected_examples: Optional[List[Dict[str, str]]] = None
        if isinstance(semantic_cfg, SemanticMetricConfig):
            self.semantic_cfg = semantic_cfg
        else:
            semantic_kwargs = dict(semantic_cfg or {})
            semantic_kwargs.setdefault("enabled", False)
            self.semantic_cfg = SemanticMetricConfig(**semantic_kwargs)

        os.makedirs(output_dir, exist_ok=True)

    def _ensure_examples_loaded(self) -> None:
        """Load RAQUEL datasets once on first evaluation run."""
        if self._affected_examples is not None and self._unaffected_examples is not None:
            return

        self._affected_examples = load_raquel_examples(self.affected_path)
        self._unaffected_examples = load_raquel_examples(self.unaffected_path)
        logger.info(
            "Loaded RAQUEL callback datasets (affected=%d, unaffected=%d)",
            len(self._affected_examples),
            len(self._unaffected_examples),
        )

    def _run_evaluation(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        logger.info("=" * 60)
        logger.info("Running RAQUEL Evaluation at %s", stage)
        logger.info("=" * 60)

        self._ensure_examples_loaded()

        split_results = evaluate_raquel_splits(
            split_examples={
                "affected": self._affected_examples or [],
                "unaffected": self._unaffected_examples or [],
            },
            model=pl_module.model,
            tokenizer=pl_module.tokenizer,
            config=self.eval_config,
            device=self.device,
            semantic_cfg=self.semantic_cfg,
            global_cfg=getattr(pl_module, "_validation_global_cfg", None),
        )
        affected = split_results["affected"]
        unaffected = split_results["unaffected"]

        results = {
            "stage": stage,
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch if trainer.current_epoch is not None else None,
            "affected": affected,
            "unaffected": unaffected,
        }

        output_file = os.path.join(self.output_dir, f"raquel_eval_{stage}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        if trainer.logger is not None:
            metrics_to_log = build_raquel_metric_logs(split_results)
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)

        logger.info("RAQUEL evaluation saved to %s", output_file)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.run_on_train_end:
            self._run_evaluation(trainer, pl_module, "final")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.run_on_epoch_end and trainer.current_epoch is not None:
            stage = f"epoch_{trainer.current_epoch}"
            self._run_evaluation(trainer, pl_module, stage)
