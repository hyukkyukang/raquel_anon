"""Loss package exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DPOLoss": (".unlearning", "DPOLoss"),
    "get_regularization_loss": (".registry", "get_regularization_loss"),
    "get_unlearning_loss": (".registry", "get_unlearning_loss"),
    "GradDescentLoss": (".regularization", "GradDescentLoss"),
    "GradientAscentLoss": (".unlearning", "GradientAscentLoss"),
    "IDKLoss": (".unlearning", "IDKLoss"),
    "KLDivergenceLoss": (".regularization", "KLDivergenceLoss"),
    "list_regularization_losses": (".registry", "list_regularization_losses"),
    "list_unlearning_losses": (".registry", "list_unlearning_losses"),
    "NPOLoss": (".unlearning", "NPOLoss"),
    "register_regularization_loss": (".registry", "register_regularization_loss"),
    "register_unlearning_loss": (".registry", "register_unlearning_loss"),
    "RegularizationLoss": (".regularization", "RegularizationLoss"),
    "UnlearningLoss": (".unlearning", "UnlearningLoss"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _EXPORTS[name]
    module = import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
