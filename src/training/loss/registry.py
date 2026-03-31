"""Registry for unlearning and regularization loss classes."""

import inspect
import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

from transformers import PreTrainedModel, PreTrainedTokenizer

if TYPE_CHECKING:
    from .regularization import (
        GradDescentLoss,
        KLDivergenceLoss,
        RegularizationLoss,
    )
    from .unlearning import (
        DPOLoss,
        GradientAscentLoss,
        IDKLoss,
        NPOLoss,
        UnlearningLoss,
    )

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic decorators
T_UnlearningLoss = TypeVar("T_UnlearningLoss", bound="UnlearningLoss")
T_RegularizationLoss = TypeVar("T_RegularizationLoss", bound="RegularizationLoss")

# Registry mapping loss names to classes and metadata
_UNLEARNING_LOSS_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGULARIZATION_LOSS_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRATIONS_READY = False


class _LossClassMetadata:
    """Metadata for a registered loss class."""

    def __init__(
        self,
        cls: Type,
        names: tuple[str, ...],
        required_params: set[str],
        default_values: Dict[str, Any],
    ):
        """
        Initialize loss class metadata.

        Args:
            cls: The loss class
            names: Registered name aliases
            required_params: Set of required parameter names
            default_values: Dictionary of default parameter values
        """
        self.cls = cls
        self.names = names
        self.required_params = required_params
        self.default_values = default_values


def ensure_loss_registrations() -> None:
    """Load decorator-bearing loss modules so registry lookups work under lazy imports."""
    global _REGISTRATIONS_READY

    if _REGISTRATIONS_READY:
        return

    import_module(".unlearning", __package__)
    import_module(".regularization", __package__)
    _REGISTRATIONS_READY = True


def _extract_class_metadata(cls: Type) -> tuple[set[str], Dict[str, Any]]:
    """
    Extract parameter requirements and defaults from a class's __init__ method.

    Args:
        cls: The class to inspect

    Returns:
        Tuple of (required_params, default_values)
    """
    sig = inspect.signature(cls.__init__)
    required_params: set[str] = set()
    default_values: Dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        if param.default == inspect.Parameter.empty:
            required_params.add(param_name)
        else:
            default_values[param_name] = param.default

    return required_params, default_values


def _create_instance_from_params(
    cls: Type,
    required_params: set[str],
    default_values: Dict[str, Any],
    **kwargs: Any,
) -> Any:
    """
    Create an instance of a class by extracting relevant parameters.

    Args:
        cls: The class to instantiate
        required_params: Set of required parameter names
        default_values: Dictionary of default parameter values
        **kwargs: Available parameters

    Returns:
        Instance of the class

    Raises:
        ValueError: If required parameters are missing
    """
    # Extract only parameters that the class accepts
    sig = inspect.signature(cls.__init__)
    class_params: Dict[str, Any] = {}

    for param_name in sig.parameters:
        if param_name == "self":
            continue

        if param_name in kwargs and kwargs[param_name] is not None:
            class_params[param_name] = kwargs[param_name]
        elif param_name in default_values:
            class_params[param_name] = default_values[param_name]

    # Check required parameters
    missing_params = required_params - set(class_params.keys())
    if missing_params:
        raise ValueError(
            f"{cls.__name__} requires the following parameters: {', '.join(missing_params)}"
        )

    return cls(**class_params)


def register_unlearning_loss(*names: str):
    """
    Decorator to register an unlearning loss class.

    Args:
        *names: One or more name aliases for the loss (e.g., "ga", "gradient_ascent")

    Returns:
        Decorator function

    Example:
        @register_unlearning_loss("ga", "gradient_ascent")
        class GradientAscentLoss(UnlearningLoss):
            ...
    """

    def decorator(cls: Type[T_UnlearningLoss]) -> Type[T_UnlearningLoss]:
        """Register the class in the unlearning loss registry."""
        if not names:
            raise ValueError("At least one name must be provided for registration")

        required_params, default_values = _extract_class_metadata(cls)
        metadata = _LossClassMetadata(cls, names, required_params, default_values)

        # Register all name aliases
        for name in names:
            name_lower = name.lower()
            if name_lower in _UNLEARNING_LOSS_REGISTRY:
                logger.warning(
                    f"Overwriting existing unlearning loss registration: {name_lower}"
                )
            _UNLEARNING_LOSS_REGISTRY[name_lower] = {
                "metadata": metadata,
            }
            logger.debug(f"Registered unlearning loss: {name_lower} -> {cls.__name__}")

        return cls

    return decorator


def register_regularization_loss(*names: str):
    """
    Decorator to register a regularization loss class.

    Args:
        *names: One or more name aliases for the loss (e.g., "gd", "grad_descent")

    Returns:
        Decorator function

    Example:
        @register_regularization_loss("gd", "grad_descent")
        class GradDescentLoss(RegularizationLoss):
            ...
    """

    def decorator(cls: Type[T_RegularizationLoss]) -> Type[T_RegularizationLoss]:
        """Register the class in the regularization loss registry."""
        if not names:
            raise ValueError("At least one name must be provided for registration")

        required_params, default_values = _extract_class_metadata(cls)
        metadata = _LossClassMetadata(cls, names, required_params, default_values)

        # Register all name aliases
        for name in names:
            name_lower = name.lower()
            if name_lower in _REGULARIZATION_LOSS_REGISTRY:
                logger.warning(
                    f"Overwriting existing regularization loss registration: {name_lower}"
                )
            _REGULARIZATION_LOSS_REGISTRY[name_lower] = {
                "metadata": metadata,
            }
            logger.debug(
                f"Registered regularization loss: {name_lower} -> {cls.__name__}"
            )

        return cls

    return decorator


def get_unlearning_loss(
    name: str,
    reference_model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    beta: Optional[float] = None,
    idk_variation: Optional[str] = None,
    max_length: Optional[int] = None,
    reduction: Optional[str] = None,
    normalize: Optional[bool] = None,
) -> "UnlearningLoss":
    """
    Get an unlearning loss instance by name.

    Args:
        name: Name of the loss (e.g., "ga", "npo", "idk", "dpo")
        reference_model: Optional reference model (required for NPO and DPO)
        tokenizer: Optional tokenizer (required for IDK and DPO)
        beta: Optional beta parameter (used by NPO and DPO)
        idk_variation: Optional IDK variation string (used by IDK and DPO)
        max_length: Optional max sequence length (used by IDK and DPO fallbacks)
        reduction: Optional reduction method (used by NPO)
        normalize: Optional normalize flag (used by NPO)

    Returns:
        An instance of the requested unlearning loss

    Raises:
        ValueError: If the loss name is not registered or required parameters are missing
    """
    ensure_loss_registrations()
    name_lower = name.lower()
    if name_lower not in _UNLEARNING_LOSS_REGISTRY:
        available = ", ".join(sorted(_UNLEARNING_LOSS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown unlearning loss: {name}. Available losses: {available}"
        )

    registry_entry = _UNLEARNING_LOSS_REGISTRY[name_lower]
    metadata: _LossClassMetadata = registry_entry["metadata"]

    # Handle class-specific defaults that were in the old factory functions
    # These defaults are applied before parameter extraction
    params: Dict[str, Any] = {
        "reference_model": reference_model,
        "tokenizer": tokenizer,
        "beta": beta,
        "idk_variation": idk_variation,
        "max_length": max_length,
        "reduction": reduction,
        "normalize": normalize,
    }

    # Apply class-specific defaults based on the class name
    # This maintains backward compatibility with the old factory functions
    if metadata.cls.__name__ == "NPOLoss":
        if params["beta"] is None:
            params["beta"] = 2.0
        if params["reduction"] is None:
            params["reduction"] = "mean"
        if params["normalize"] is None:
            params["normalize"] = True
    elif metadata.cls.__name__ == "IDKLoss":
        if params["idk_variation"] is None:
            params["idk_variation"] = "random"
        if params["max_length"] is None:
            params["max_length"] = 1024
    elif metadata.cls.__name__ == "DPOLoss":
        if params["beta"] is None:
            params["beta"] = 0.1
        if params["idk_variation"] is None:
            params["idk_variation"] = "random"
        if params["max_length"] is None:
            params["max_length"] = 1024

    return _create_instance_from_params(
        metadata.cls,
        metadata.required_params,
        metadata.default_values,
        **params,
    )


def list_unlearning_losses() -> list[str]:
    """
    List all registered unlearning loss names.

    Returns:
        List of registered loss names
    """
    ensure_loss_registrations()
    return sorted(_UNLEARNING_LOSS_REGISTRY.keys())


def get_regularization_loss(
    name: str,
    reference_model: Optional[PreTrainedModel] = None,
    reduction: Optional[str] = None,
) -> "RegularizationLoss":
    """
    Get a regularization loss instance by name.

    Args:
        name: Name of the loss (e.g., "gd", "kl")
        reference_model: Optional reference model (required for KL)
        reduction: Optional reduction method (used by KL)

    Returns:
        An instance of the requested regularization loss

    Raises:
        ValueError: If the loss name is not registered or required parameters are missing
    """
    ensure_loss_registrations()
    name_lower = name.lower()
    if name_lower not in _REGULARIZATION_LOSS_REGISTRY:
        available = ", ".join(sorted(_REGULARIZATION_LOSS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown regularization loss: {name}. Available losses: {available}"
        )

    registry_entry = _REGULARIZATION_LOSS_REGISTRY[name_lower]
    metadata: _LossClassMetadata = registry_entry["metadata"]

    # Handle class-specific defaults that were in the old factory functions
    params: Dict[str, Any] = {
        "reference_model": reference_model,
        "reduction": reduction,
    }

    # Apply class-specific defaults based on the class name
    # This maintains backward compatibility with the old factory functions
    if metadata.cls.__name__ == "KLDivergenceLoss":
        if params["reduction"] is None:
            params["reduction"] = "batchmean"

    return _create_instance_from_params(
        metadata.cls,
        metadata.required_params,
        metadata.default_values,
        **params,
    )


def list_regularization_losses() -> list[str]:
    """
    List all registered regularization loss names.

    Returns:
        List of registered loss names
    """
    ensure_loss_registrations()
    return sorted(_REGULARIZATION_LOSS_REGISTRY.keys())
