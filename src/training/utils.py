"""Utility functions for training."""

from typing import Any, Dict

from lightning.pytorch.utilities import rank_zero_only
import torch

from src.training.model_loading import load_model_and_tokenizer
from src.training.model_paths import check_model_exists, get_base_model_dir_component
from src.training.methods import (
    LABEL_IGNORE_INDEX,
    METHOD_DESCRIPTIONS,
    REGULARIZATION_METHODS,
    UNLEARNING_LOSS_METHODS,
    UNLEARNING_METHODS,
    check_unlearning_method,
    needs_idk_dataset,
    needs_reference_model,
    parse_unlearning_method,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@rank_zero_only
def log_if_rank_zero(msg: str, *args) -> None:
    """Log info messages only from the main process in distributed setups."""
    logger.info(msg, *args)
