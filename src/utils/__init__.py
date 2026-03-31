"""Utility modules for RAQUEL.

This package provides common utilities including:
- data_loaders: Loading QA datasets, SQL queries, and schema files
- database: Database connections and path building utilities
- db_operations: PostgreSQL shell command wrappers
- logging: Logger configuration utilities
- table_data: Table data retrieval and formatting
- table_cache: In-memory table data caching for efficient upserts
- results_saver: Intermediate results saving utilities
- checkpoint: Checkpoint management for resumable builds
- batches: Batch iteration utility for processing items in chunks
"""

from importlib import import_module
from typing import Any, Dict, Generator, List, Tuple, TypeVar

# Type variable for generic batching
T = TypeVar("T")


def batches(items: List[T], batch_size: int) -> Generator[List[T], None, None]:
    """Yield successive batches from items list.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        Batches of items as lists
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BuildCheckpoint": (".checkpoint", "BuildCheckpoint"),
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "load_insert_statements": (".data_loaders", "load_insert_statements"),
    "load_qa_dataset": (".data_loaders", "load_qa_dataset"),
    "load_schema": (".data_loaders", "load_schema"),
    "load_schema_statements": (".data_loaders", "load_schema_statements"),
    "load_sql_queries": (".data_loaders", "load_sql_queries"),
    "load_translated_queries": (".data_loaders", "load_translated_queries"),
    "PathBuilder": (".database", "PathBuilder"),
    "PathBuilderMixin": (".database", "PathBuilderMixin"),
    "PostgresConnectionMixin": (".database", "PostgresConnectionMixin"),
    "PostgresShellOperations": (".db_operations", "PostgresShellOperations"),
    "build_greedy_generation_kwargs": (".generation", "build_greedy_generation_kwargs"),
    "get_logger": (".logging", "get_logger"),
    "IntermediateResultsSaver": (".results_saver", "IntermediateResultsSaver"),
    "SQLStatementSaver": (".results_saver", "SQLStatementSaver"),
    "TableDataCache": (".table_cache", "TableDataCache"),
    "extract_table_names_from_schema": (".table_data", "extract_table_names_from_schema"),
    "extract_table_names_from_schema_str": (
        ".table_data",
        "extract_table_names_from_schema_str",
    ),
    "fetch_table_data": (".table_data", "fetch_table_data"),
    "format_table_data_for_prompt": (".table_data", "format_table_data_for_prompt"),
    "get_all_table_data_from_schema": (".table_data", "get_all_table_data_from_schema"),
    "get_all_table_data_from_schema_str": (
        ".table_data",
        "get_all_table_data_from_schema_str",
    ),
}


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

# Note: json_utils is imported lazily to avoid circular imports
# (json_utils -> llm.json_repair -> llm.api -> utils.env)
# Import directly: from src.utils.json_utils import safe_json_loads

__all__ = [
    # Batch utilities
    "batches",
    # Data loaders
    "load_qa_dataset",
    "load_sql_queries",
    "load_translated_queries",
    "load_schema",
    "load_schema_statements",
    "load_insert_statements",
    # Database utilities
    "PathBuilder",
    "PathBuilderMixin",
    "PostgresConnectionMixin",
    "PostgresShellOperations",
    "build_greedy_generation_kwargs",
    # Logging
    "get_logger",
    # Table data utilities
    "extract_table_names_from_schema",
    "extract_table_names_from_schema_str",
    "fetch_table_data",
    "format_table_data_for_prompt",
    "get_all_table_data_from_schema",
    "get_all_table_data_from_schema_str",
    # Table data caching
    "TableDataCache",
    # Results saving
    "IntermediateResultsSaver",
    "SQLStatementSaver",
    # Checkpoint management
    "BuildCheckpoint",
    "CheckpointManager",
    # JSON utilities (import directly from src.utils.json_utils to avoid circular imports)
]
