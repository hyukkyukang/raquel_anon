"""Training package exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple


class _LazyModuleProxy:
    """Module-like proxy that imports the real package on first attribute access."""

    def __init__(self, module_path: str, package: str):
        self._module_path = module_path
        self._package = package
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = import_module(self._module_path, self._package)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())

    def __repr__(self) -> str:
        return f"<lazy module proxy {self._package}{self._module_path}>"


_MODULE_EXPORTS: Dict[str, str] = {
    "callback": ".callback",
    "data": ".data",
    "datamodules": ".data",
    "logger": ".logger",
    "loss": ".loss",
    "model": ".model",
}

_VALUE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "METHOD_DESCRIPTIONS": (".methods", "METHOD_DESCRIPTIONS"),
    "UNLEARNING_METHODS": (".methods", "UNLEARNING_METHODS"),
    "check_model_exists": (".model_paths", "check_model_exists"),
    "get_base_model_dir_component": (".model_paths", "get_base_model_dir_component"),
    "load_model_and_tokenizer": (".model_loading", "load_model_and_tokenizer"),
    "needs_idk_dataset": (".methods", "needs_idk_dataset"),
    "needs_reference_model": (".methods", "needs_reference_model"),
    "parse_unlearning_method": (".methods", "parse_unlearning_method"),
    "check_unlearning_method": (".methods", "check_unlearning_method"),
}

__all__ = list(_MODULE_EXPORTS) + list(_VALUE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        proxy = _LazyModuleProxy(_MODULE_EXPORTS[name], __name__)
        globals()[name] = proxy
        return proxy

    if name in _VALUE_EXPORTS:
        module_path, attr_name = _VALUE_EXPORTS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
