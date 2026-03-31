"""Evaluation package exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "build_muse_core_metric_logs": (".muse", "build_muse_core_metric_logs"),
    "MUSEEvaluator": (".muse", "MUSEEvaluator"),
    "RAQUELEvalConfig": (".raquel", "RAQUELEvalConfig"),
    "TraditionalEvalConfig": (".traditional_data", "TraditionalEvalConfig"),
    "build_raquel_metric_logs": (".raquel", "build_raquel_metric_logs"),
    "compute_prob": (".traditional", "compute_prob"),
    "compute_rouge_l_recall": (".traditional", "compute_rouge_l_recall"),
    "compute_truth_ratio": (".traditional", "compute_truth_ratio"),
    "evaluate_raquel_split": (".raquel", "evaluate_raquel_split"),
    "evaluate_raquel_splits": (".raquel", "evaluate_raquel_splits"),
    "evaluate_muse_core_metrics": (".muse", "evaluate_muse_core_metrics"),
    "evaluate_traditional": (".traditional", "evaluate_traditional"),
    "evaluate_with_muse": (".traditional", "evaluate_with_muse"),
    "generate_answer": (".traditional", "generate_answer"),
    "load_evaluation_data": (".muse", "load_evaluation_data"),
    "load_raquel_examples": (".raquel", "load_raquel_examples"),
    "load_traditional_examples": (".traditional_data", "load_traditional_examples"),
    "resolve_generation_device": (".raquel", "resolve_generation_device"),
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
