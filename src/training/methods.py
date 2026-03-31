"""Lightweight training method metadata and compatibility helpers."""

from typing import Optional, Tuple

UNLEARNING_LOSS_METHODS = [
    "ga",
    "npo",
    "idk",
    "dpo",
]

REGULARIZATION_METHODS = [
    "gd",
    "kl",
]

METHOD_DESCRIPTIONS = {
    "ga_gd": "Gradient Ascent (maximize forget loss) + Grad Descent (minimize retain loss)",
    "ga_kl": "Gradient Ascent (maximize forget loss) + KL Divergence (stay close to original)",
    "npo_gd": "Negative Preference Optimization (forget answers relative to a reference model) + Grad Descent (minimize retain loss)",
    "npo_kl": "Negative Preference Optimization (forget answers relative to a reference model) + KL Divergence (stay close to original)",
    "idk_gd": "IDK Fine-tune (replace answers with 'I don't know') + Grad Descent (minimize retain loss)",
    "idk_kl": "IDK Fine-tune (replace answers with 'I don't know') + KL Divergence (stay close to original)",
    "dpo_gd": "Direct Preference Optimization (prefer IDK over original) + Grad Descent (minimize retain loss)",
    "dpo_kl": "Direct Preference Optimization (prefer IDK over original) + KL Divergence (stay close to original)",
}

UNLEARNING_METHODS = list(METHOD_DESCRIPTIONS.keys())

LABEL_IGNORE_INDEX = -100


def parse_unlearning_method(method: str) -> Tuple[str, str]:
    """Parse a combined unlearning method string into its components."""
    if method not in UNLEARNING_METHODS:
        raise ValueError(
            f"Unknown unlearning method: {method}. "
            f"Choose from: {', '.join(UNLEARNING_METHODS)}"
        )

    return tuple(method.split("_", maxsplit=1))  # type: ignore[return-value]


def check_unlearning_method(
    unlearning_method: str, regularization_method: Optional[str] = None
) -> Tuple[str, str]:
    """Validate bare or combined unlearning method inputs."""
    if regularization_method is None:
        return parse_unlearning_method(unlearning_method)

    if unlearning_method not in UNLEARNING_LOSS_METHODS:
        raise ValueError(
            f"Unknown unlearning method: {unlearning_method}. "
            f"Choose from: {', '.join(UNLEARNING_LOSS_METHODS)}"
        )
    if regularization_method not in REGULARIZATION_METHODS:
        raise ValueError(
            f"Unknown regularization method: {regularization_method}. "
            f"Choose from: {', '.join(REGULARIZATION_METHODS)}"
        )

    return unlearning_method, regularization_method


def needs_idk_dataset(
    unlearning_method: str, regularization_method: Optional[str] = None
) -> bool:
    """Return whether the method needs IDK-style auxiliary data."""
    parsed_unlearning_method, _ = check_unlearning_method(
        unlearning_method, regularization_method
    )
    return parsed_unlearning_method in ["idk", "dpo"]


def needs_reference_model(
    unlearning_method: str, regularization_method: Optional[str] = None
) -> bool:
    """Return whether the method needs a frozen reference model."""
    parsed_unlearning_method, parsed_regularization_method = check_unlearning_method(
        unlearning_method, regularization_method
    )
    return (
        parsed_regularization_method == "kl"
        or parsed_unlearning_method in ["dpo", "npo"]
    )
