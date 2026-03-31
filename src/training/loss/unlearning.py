"""Unlearning loss functions for machine unlearning."""

import abc
import logging
import math
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.batch import extract_model_inputs
from src.training.data import create_idk_dataset
from src.training.loss.registry import register_unlearning_loss
from src.training.loss.utils import build_masked_batch, resolve_model_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


class UnlearningLoss(nn.Module, abc.ABC):
    """
    Abstract base class for unlearning loss functions.

    All unlearning loss classes should inherit from this base class and
    implement the forward method.
    """

    def __init__(self):
        """Initialize the unlearning loss."""
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        model: PreTrainedModel,
        forget_batch: Dict[str, Any],
        forget_examples_raw: Optional[List[Dict[str, str]]] = None,
        idk_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute the unlearning loss.

        Args:
            model: The model being trained
            forget_batch: Batch from forget dataset with tokenized inputs
            forget_examples_raw: Optional raw forget examples with question/answer keys
            idk_batch: Optional aligned IDK batch for targeted losses

        Returns:
            Loss value (scalar tensor)
        """
        raise NotImplementedError

    @staticmethod
    def _sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute summed log probabilities of target sequences, ignoring positions with label -100.

        Args:
            logits: FloatTensor[B, T, V] - model logits
            labels: LongTensor[B, T] - token ids; -100 for ignored positions

        Returns:
            logps: FloatTensor[B] - Sum of token log-probabilities per sequence
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~mask, 0)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * mask

        return token_log_probs.sum(dim=-1)


@register_unlearning_loss("ga", "gradient_ascent")
class GradientAscentLoss(UnlearningLoss):
    """
    Gradient Ascent unlearning loss.

    Maximizes the loss on forget data to encourage the model to unlearn.
    This is the simplest untargeted unlearning method.
    """

    def __init__(self):
        """Initialize gradient ascent loss."""
        super().__init__()

    def forward(
        self,
        model: PreTrainedModel,
        forget_batch: Dict[str, Any],
        forget_examples_raw: Optional[List[Dict[str, str]]] = None,
        idk_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute gradient ascent loss.

        Args:
            model: The model being trained
            forget_batch: Batch from forget dataset
            forget_examples_raw: Not used in this loss (kept for interface compatibility)

        Returns:
            Negative of the standard loss (to maximize original loss)
        """
        outputs = model(**extract_model_inputs(forget_batch))
        forget_loss = outputs.loss
        # Negative loss for gradient ascent
        return -forget_loss


@register_unlearning_loss("npo")
class NPOLoss(UnlearningLoss):
    r"""
    Negative Preference Optimization (NPO) loss.

    Let Δ(x, y) = log π_θ(y|x) − log π_ref(y|x).

    The per-example NPO loss is:

        L = -(2/β) * log σ(-β Δ)

    where β>0 and σ is the logistic sigmoid.

    Notes
    -----
    * Gradient w.r.t. log π_θ is 2 * σ(β Δ) ∈ (0, 2), which:

        - approaches ~2 when the current model over-remembers (Δ≫0),
        - smoothly turns off as forgetting succeeds (Δ≪0),
        - reduces to gradient ascent on log π_θ when β→0 (weight → 1).

    * We add an optional constant (-2/β * log 2) to keep the loss well-scaled
      near Δ≈0 without changing gradients (set `normalize=True`).
    """

    def __init__(
        self,
        reference_model: PreTrainedModel,
        beta: float = 2.0,
        reduction: str = "mean",
        normalize: bool = True,
    ):
        """
        Initialize NPO loss.

        Args:
            reference_model: Reference model (typically base model) for computing log-probs
            beta: Inverse temperature β controlling selectivity (default: 2.0)
            reduction: Reduction over the batch ("mean", "sum", or "none")
            normalize: If True, subtract (2/β)*log 2 for numerical stability
        """
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean','sum','none'}")

        self.reference_model = reference_model
        self.beta = float(beta)
        self.reduction = reduction
        self.normalize = normalize

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

    def forward(
        self,
        model: PreTrainedModel,
        forget_batch: Dict[str, Any],
        forget_examples_raw: Optional[List[Dict[str, str]]] = None,
        idk_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute NPO loss.

        Args:
            model: The model being trained
            forget_batch: Batch from forget dataset with 'input_ids' and 'labels'
            forget_examples_raw: Not used in this loss (kept for interface compatibility)

        Returns:
            NPO loss value (scalar if reduced, else per-example)
        """
        model_inputs = extract_model_inputs(forget_batch)

        # Compute sequence log-probabilities for current model
        current_outputs = model(**model_inputs)
        current_logps = self._sequence_log_probs(
            current_outputs.logits, model_inputs["labels"]
        )

        # Compute sequence log-probabilities for reference model (frozen)
        with torch.no_grad():
            ref_outputs = self.reference_model(**model_inputs)
            ref_logps = self._sequence_log_probs(
                ref_outputs.logits, model_inputs["labels"]
            )

        # Detach the reference to prevent gradients flowing into the frozen model
        ref_logps = ref_logps.detach()

        # Δ = log π_θ(y|x) - log π_ref(y|x)
        delta = (current_logps - ref_logps).to(torch.float32)
        scaled = -self.beta * delta  # -βΔ

        # Stable log-sigmoid
        log_sig = torch.nn.functional.logsigmoid(scaled)  # = -softplus(βΔ)

        # Optional constant shift keeps loss ~0 near Δ≈0 (does not change gradients)
        if self.normalize:
            log_sig = log_sig + math.log(2.0)

        loss_vec = -(2.0 / self.beta) * log_sig  # [B]

        if self.reduction == "mean":
            loss = loss_vec.mean()
        elif self.reduction == "sum":
            loss = loss_vec.sum()
        else:
            loss = loss_vec

        return loss


@register_unlearning_loss("idk")
class IDKLoss(UnlearningLoss):
    """
    IDK Fine-tuning loss.

    Directly fine-tunes the model to produce "I don't know" responses for forget queries.
    This is a simple targeted unlearning approach.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        idk_variation: str = "random",
        max_length: int = 1024,
    ):
        """
        Initialize IDK loss.

        Args:
            tokenizer: Tokenizer for processing IDK examples
            idk_variation: Type of IDK response variation
            max_length: Maximum tokenization length for fallback IDK batches
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.idk_variation = idk_variation
        self.max_length = max_length

    def forward(
        self,
        model: PreTrainedModel,
        forget_batch: Dict[str, Any],
        forget_examples_raw: Optional[List[Dict[str, str]]] = None,
        idk_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute IDK fine-tuning loss.

        Args:
            model: The model being trained
            forget_batch: Batch from forget dataset (used for batch size)
            forget_examples_raw: Raw forget examples with question/answer keys

        Returns:
            Cross-entropy loss on IDK responses
        """
        if idk_batch is not None:
            outputs = model(**extract_model_inputs(idk_batch))
            return outputs.loss

        forget_model_inputs = extract_model_inputs(forget_batch)

        if forget_examples_raw is None or len(forget_examples_raw) == 0:
            # Fallback if no raw examples
            logger.warning("No raw examples provided for IDK loss, using default loss")
            outputs = model(**forget_model_inputs)
            return outputs.loss

        batch_size = forget_model_inputs["input_ids"].size(0)

        # Create IDK examples for current batch
        # Sample by index to handle both Lists and Datasets (which random.sample might reject)
        num_examples = len(forget_examples_raw)
        sample_size = min(batch_size, num_examples)
        indices = random.sample(range(num_examples), sample_size)
        sampled_examples = [forget_examples_raw[i] for i in indices]

        idk_examples = create_idk_dataset(sampled_examples, self.idk_variation)

        device = resolve_model_device(model)
        idk_batch = build_masked_batch(
            self.tokenizer,
            idk_examples,
            device,
            max_length=self.max_length,
        )

        outputs = model(**idk_batch)
        return outputs.loss


@register_unlearning_loss("dpo")
class DPOLoss(UnlearningLoss):
    """
    Direct Preference Optimization (DPO) loss for unlearning.

    Uses preference learning to make the model prefer "I don't know" responses
    over original answers for forget data.
    """

    def __init__(
        self,
        reference_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        beta: float = 0.1,
        idk_variation: str = "random",
        max_length: int = 1024,
    ):
        """
        Initialize DPO loss.

        Args:
            reference_model: Original/reference model for computing preference
            tokenizer: Tokenizer for processing examples
            beta: DPO temperature parameter
            idk_variation: Type of IDK response variation
            max_length: Maximum tokenization length for fallback DPO batches
        """
        super().__init__()
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.idk_variation = idk_variation
        self.max_length = max_length

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

    def forward(
        self,
        model: PreTrainedModel,
        forget_batch: Dict[str, Any],
        forget_examples_raw: Optional[List[Dict[str, str]]] = None,
        idk_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        Args:
            model: The model being trained
            forget_batch: Batch from forget dataset
            forget_examples_raw: Raw forget examples with question/answer keys

        Returns:
            DPO loss value
        """
        dispreferred_batch = extract_model_inputs(forget_batch)

        if idk_batch is not None:
            preferred_batch = extract_model_inputs(idk_batch)
        elif forget_examples_raw is None or len(forget_examples_raw) == 0:
            logger.warning("No raw examples provided for DPO loss, using default loss")
            outputs = model(**dispreferred_batch)
            return -outputs.loss
        else:
            batch_size = dispreferred_batch["input_ids"].size(0)

            # Sample by index to handle both Lists and Datasets
            num_examples = len(forget_examples_raw)
            sample_size = min(batch_size, num_examples)
            indices = random.sample(range(num_examples), sample_size)
            sampled_examples = [forget_examples_raw[i] for i in indices]

            # Create preferred (IDK) and dispreferred (original) responses
            idk_examples = create_idk_dataset(sampled_examples, self.idk_variation)

            device = resolve_model_device(model)

            preferred_batch = build_masked_batch(
                self.tokenizer,
                idk_examples,
                device,
                max_length=self.max_length,
            )
            dispreferred_batch = build_masked_batch(
                self.tokenizer,
                sampled_examples,
                device,
                max_length=self.max_length,
            )

        if preferred_batch["input_ids"].size(0) != dispreferred_batch["input_ids"].size(
            0
        ):
            raise ValueError("Preferred and dispreferred DPO batches must be aligned.")

        # Compute per-example sequence log probabilities for the current model
        preferred_outputs = model(**preferred_batch)
        dispreferred_outputs = model(**dispreferred_batch)

        preferred_logprob = self._sequence_log_probs(
            preferred_outputs.logits, preferred_batch["labels"]
        )
        dispreferred_logprob = self._sequence_log_probs(
            dispreferred_outputs.logits, dispreferred_batch["labels"]
        )

        # Reference model is frozen; evaluate without gradients
        with torch.no_grad():
            ref_preferred_outputs = self.reference_model(**preferred_batch)
            ref_dispreferred_outputs = self.reference_model(**dispreferred_batch)

            ref_preferred_logprob = self._sequence_log_probs(
                ref_preferred_outputs.logits, preferred_batch["labels"]
            )
            ref_dispreferred_logprob = self._sequence_log_probs(
                ref_dispreferred_outputs.logits, dispreferred_batch["labels"]
            )

        logits = self.beta * (
            (preferred_logprob - dispreferred_logprob)
            - (ref_preferred_logprob - ref_dispreferred_logprob)
        )
        dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()

        return dpo_loss
