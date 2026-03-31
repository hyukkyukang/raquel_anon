"""Aligned database exports with lazy loading."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CascadeDeleteHandler": (".cascade_delete", "CascadeDeleteHandler"),
    "SchemaHeuristicChecker": (".schema_heuristics", "SchemaHeuristicChecker"),
    "EntityRegistry": (".entity_registry", "EntityRegistry"),
    "ColumnInfo": (".schema_registry", "ColumnInfo"),
    "ForeignKeyConstraint": (".schema_registry", "ForeignKeyConstraint"),
    "TableSchema": (".schema_registry", "TableSchema"),
    "SchemaRegistry": (".schema_registry", "SchemaRegistry"),
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
