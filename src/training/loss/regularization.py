"""Regularization loss functions for unlearning."""

import abc
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from src.training.batch import extract_model_inputs
from src.training.loss.registry import register_regularization_loss
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegularizationLoss(nn.Module, abc.ABC):
    """
    Abstract base class for regularization loss functions.

    All regularization loss classes should inherit from this base class and
    implement the forward method.
    """

    def __init__(self):
        """Initialize the regularization loss."""
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        model: PreTrainedModel,
        retain_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute the regularization loss.

        Args:
            model: The model being trained
            retain_batch: Optional batch from retain dataset with tokenized inputs

        Returns:
            Loss value (scalar tensor)
        """
        raise NotImplementedError


@register_regularization_loss("gd", "grad_descent")
class GradDescentLoss(RegularizationLoss):
    """
    Gradient Descent regularization loss.

    Minimizes the loss on retain data to preserve model utility.
    This is the standard supervised learning loss.
    """

    def __init__(self):
        """Initialize gradient descent loss."""
        super().__init__()

    def forward(
        self,
        model: PreTrainedModel,
        retain_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute gradient descent loss on retain set.

        Args:
            model: The model being trained
            retain_batch: Batch from retain dataset

        Returns:
            Standard cross-entropy loss on retain data
        """
        if retain_batch is None:
            logger.warning("No retain batch provided for GradDescentLoss")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        outputs = model(**extract_model_inputs(retain_batch))
        return outputs.loss


@register_regularization_loss("kl", "kl_divergence")
class KLDivergenceLoss(RegularizationLoss):
    """
    KL Divergence regularization loss.

    Encourages the unlearned model to stay close to the original model
    on the retain set, preserving utility while unlearning forget data.
    """

    def __init__(self, reference_model: PreTrainedModel, reduction: str = "batchmean"):
        """
        Initialize KL Divergence loss.

        Args:
            reference_model: Original model to stay close to
            reduction: Reduction method for KL divergence ('batchmean', 'sum', 'mean')
        """
        super().__init__()
        self.reference_model = reference_model
        self.reduction = reduction

        # Freeze reference model parameters
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Set reference model to eval mode
        self.reference_model.eval()

    def forward(
        self,
        model: PreTrainedModel,
        retain_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference model on retain set.

        Args:
            model: The model being trained
            retain_batch: Batch from retain dataset

        Returns:
            KL divergence loss
        """
        if retain_batch is None:
            logger.warning("No retain batch provided for KLDivergenceLoss")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        model_inputs = extract_model_inputs(retain_batch)

        # Current model outputs (with gradients)
        current_outputs = model(**model_inputs)
        current_logits = current_outputs.logits

        # Reference model outputs (no gradients)
        with torch.no_grad():
            reference_outputs = self.reference_model(**model_inputs)
            reference_logits = reference_outputs.logits

        # Compute KL divergence
        # KL(P||Q) where P = reference, Q = current
        # We want current model to match reference model
        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(current_logits, dim=-1),
            nn.functional.softmax(reference_logits, dim=-1),
            reduction=self.reduction,
        )

        return kl_loss
