"""Data package exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CustomDataCollator": (".collator", "CustomDataCollator"),
    "UnlearningDataLoader": (".dataloader", "UnlearningDataLoader"),
    "QATokenizedDataset": (".datasets", "QATokenizedDataset"),
    "BaseDataModule": (".pl_module", "BaseDataModule"),
    "FineTuneDataModule": (".pl_module", "FineTuneDataModule"),
    "UnlearningDataModule": (".pl_module", "UnlearningDataModule"),
    "create_idk_dataset": (".transforms", "create_idk_dataset"),
    "format_examples": (".transforms", "format_examples"),
    "get_idk_response": (".transforms", "get_idk_response"),
    "tokenize_function": (".transforms", "tokenize_function"),
    "load_dataset": (".utils", "load_json_dataset"),
    "load_json_dataset": (".utils", "load_json_dataset"),
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
