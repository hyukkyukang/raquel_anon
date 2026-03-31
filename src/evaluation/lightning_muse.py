"""
Lightning-based MUSE Evaluator for Distributed Multi-GPU Evaluation.

This module provides PyTorch Lightning modules for running MUSE evaluation
metrics across multiple GPUs using DistributedDataParallel (DDP).
"""

import logging
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
from rouge_score import rouge_scorer
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.generation import build_greedy_generation_kwargs
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MUSELightningModule(pl.LightningModule):
    """
    Lightning module for distributed MUSE evaluation.

    Supports parallel evaluation of MUSE metrics across multiple GPUs.
    Each metric type (verbatim, knowledge, privacy, utility) can be
    evaluated independently using this module.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        metric_type: str,
        max_new_tokens: int = 100,
    ):
        """
        Initialize MUSE Lightning module.

        Args:
            model: Pre-trained model to evaluate
            tokenizer: Tokenizer for the model
            metric_type: Type of metric ("verbatim", "knowledge", "privacy", "utility")
            max_new_tokens: Maximum tokens to generate in responses
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.metric_type = metric_type
        self.max_new_tokens = max_new_tokens

        # ROUGE scorer for text similarity
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        # Storage for results across batches
        self.test_outputs = []

    def forward(self, input_ids, attention_mask):
        """Generate model response."""
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **build_greedy_generation_kwargs(
                    self.model,
                    self.tokenizer,
                    input_length=int(input_ids.size(1)),
                    max_new_tokens=self.max_new_tokens,
                ),
            )
        return outputs

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Evaluation step for a single batch.

        Args:
            batch: Batch containing questions and ground truth answers
            batch_idx: Index of current batch

        Returns:
            Dictionary with scores for this batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        ground_truths = batch["ground_truth"]

        # Generate responses
        generated_ids = self.forward(input_ids, attention_mask)

        # Decode only generated portion (exclude prompt)
        prompt_lengths = attention_mask.sum(dim=1)
        responses = []
        for i, gen_seq in enumerate(generated_ids):
            response_tokens = gen_seq[prompt_lengths[i] :]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response.strip())

        # Compute ROUGE-L scores
        rouge_scores = []
        for response, ground_truth in zip(responses, ground_truths):
            score = self.rouge_scorer.score(ground_truth, response)
            rouge_scores.append(score["rougeL"].fmeasure)

        # For privacy metric, also compute confidence scores
        if self.metric_type == "privacy":
            confidence_scores = self._compute_confidence_scores(batch)
            return {
                "rouge_scores": rouge_scores,
                "confidence_scores": confidence_scores,
                "is_member": batch.get("is_member", [1] * len(rouge_scores)),
            }

        return {"rouge_scores": rouge_scores}

    def _compute_confidence_scores(self, batch: Dict[str, Any]) -> List[float]:
        """
        Compute confidence scores for privacy metric (MIA).

        Args:
            batch: Batch with input_ids and labels

        Returns:
            List of confidence scores (negative log likelihood)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate per-example NLL
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            losses = losses.view(shift_labels.shape)

            # Average per sequence, return negative (lower loss = higher confidence)
            confidence_scores = -losses.mean(dim=1).cpu().tolist()

        return confidence_scores

    def on_test_epoch_end(self):
        """
        Aggregate results from all batches and compute final metrics.
        Called after all test batches have been processed.
        """
        # Gather outputs from all GPUs
        all_outputs = self.all_gather(self.test_outputs)

        # Only rank 0 computes final metrics
        if self.trainer.is_global_zero:
            if self.metric_type == "privacy":
                results = self._aggregate_privacy_results(all_outputs)
            else:
                results = self._aggregate_rouge_results(all_outputs)

            # Log results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    self.log(key, value, rank_zero_only=True)

            logger.info(f"MUSE {self.metric_type} evaluation complete: {results}")

    def _aggregate_rouge_results(self, outputs: List[Dict]) -> Dict[str, float]:
        """Aggregate ROUGE scores from all batches."""
        import numpy as np

        all_scores = []
        for output in outputs:
            if isinstance(output, list):
                for item in output:
                    all_scores.extend(item.get("rouge_scores", []))
            else:
                all_scores.extend(output.get("rouge_scores", []))

        return {
            f"{self.metric_type}_rouge_mean": float(np.mean(all_scores)),
            f"{self.metric_type}_rouge_std": float(np.std(all_scores)),
            f"{self.metric_type}_num_examples": len(all_scores),
        }

    def _aggregate_privacy_results(self, outputs: List[Dict]) -> Dict[str, float]:
        """Aggregate privacy (MIA) results from all batches."""
        import numpy as np

        all_confidence = []
        all_is_member = []

        for output in outputs:
            if isinstance(output, list):
                for item in output:
                    all_confidence.extend(item.get("confidence_scores", []))
                    all_is_member.extend(item.get("is_member", []))
            else:
                all_confidence.extend(output.get("confidence_scores", []))
                all_is_member.extend(output.get("is_member", []))

        # MIA: Use median as threshold
        threshold = np.median(all_confidence)
        predictions = [1 if score > threshold else 0 for score in all_confidence]
        mia_accuracy = np.mean(
            [pred == label for pred, label in zip(predictions, all_is_member)]
        )

        # Separate forget vs non-training confidence
        forget_confidence = [c for c, m in zip(all_confidence, all_is_member) if m == 1]
        non_training_confidence = [
            c for c, m in zip(all_confidence, all_is_member) if m == 0
        ]

        return {
            "privacy_mia_accuracy": float(mia_accuracy),
            "privacy_forget_confidence_mean": (
                float(np.mean(forget_confidence)) if forget_confidence else 0.0
            ),
            "privacy_non_training_confidence_mean": (
                float(np.mean(non_training_confidence))
                if non_training_confidence
                else 0.0
            ),
            "privacy_confidence_threshold": float(threshold),
        }

    def configure_optimizers(self):
        """Not used for evaluation, but required by Lightning."""
        return None


class MUSEEvaluationRunner:
    """
    High-level runner for MUSE evaluation using PyTorch Lightning.

    Handles setup of trainer and execution of different MUSE metrics
    with automatic multi-GPU support.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        accelerator: str = "auto",
        devices: int = 1,
        precision: int = 16,
    ):
        """
        Initialize MUSE evaluation runner.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            accelerator: Accelerator type ('gpu', 'cpu', 'auto')
            devices: Number of devices to use
            precision: Precision for evaluation (16 or 32)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision

    def evaluate_metric(
        self,
        datamodule: pl.LightningDataModule,
        metric_type: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate a single MUSE metric.

        Args:
            datamodule: DataModule with test data
            metric_type: Type of metric to evaluate
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with evaluation results
        """
        # Create Lightning module for this metric
        lightning_module = MUSELightningModule(
            model=self.model,
            tokenizer=self.tokenizer,
            metric_type=metric_type,
            max_new_tokens=max_new_tokens,
        )

        # Create trainer
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        # Run evaluation
        logger.info(f"Evaluating MUSE metric: {metric_type}")
        trainer.test(lightning_module, datamodule=datamodule)

        # Return aggregated results (stored in module during test)
        return {}  # Results are logged, can be retrieved from callbacks if needed

    def evaluate_all(
        self,
        forget_datamodule: pl.LightningDataModule,
        retain_datamodule: pl.LightningDataModule,
        paraphrased_datamodule: Optional[pl.LightningDataModule] = None,
        non_training_datamodule: Optional[pl.LightningDataModule] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all available MUSE metrics.

        Args:
            forget_datamodule: DataModule for forget set
            retain_datamodule: DataModule for retain set
            paraphrased_datamodule: Optional DataModule for paraphrased questions
            non_training_datamodule: Optional DataModule for non-training data

        Returns:
            Dictionary with all evaluation results
        """
        results = {}

        # Metric 1: Verbatim Memorization
        logger.info("Evaluating Metric 1: Verbatim Memorization")
        results["verbatim"] = self.evaluate_metric(forget_datamodule, "verbatim")

        # Metric 2: Knowledge Memorization
        if paraphrased_datamodule:
            logger.info("Evaluating Metric 2: Knowledge Memorization")
            results["knowledge"] = self.evaluate_metric(
                paraphrased_datamodule, "knowledge"
            )

        # Metric 3: Privacy Leakage (requires combined dataset)
        if non_training_datamodule:
            logger.info("Evaluating Metric 3: Privacy Leakage")
            # Note: This requires a combined datamodule with both forget and non-training
            # The datamodule should mark examples with is_member flag
            results["privacy"] = self.evaluate_metric(
                non_training_datamodule, "privacy"
            )

        # Metric 4: Utility Preservation
        logger.info("Evaluating Metric 4: Utility Preservation")
        results["utility"] = self.evaluate_metric(retain_datamodule, "utility")

        return results
