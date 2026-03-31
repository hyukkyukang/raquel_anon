"""Centralized data loading utilities.

This module consolidates data loading functions that were previously duplicated
across multiple stage modules, providing a single source of truth for:
- Loading QA datasets from HuggingFace
- Loading SQL queries from files
- Loading schema definitions
- Loading insert statements
"""

import os
from typing import List, Tuple

from datasets import load_dataset
from omegaconf import DictConfig


def load_qa_dataset(cfg: DictConfig, name: str) -> List[Tuple[str, str]]:
    """Load QA pairs from a HuggingFace dataset.

    Args:
        cfg: Hydra configuration containing dataset settings
        name: Dataset subset name (e.g., "retain90", "forget10")

    Returns:
        List of (question, answer) tuples
    """
    dataset = load_dataset(
        path=cfg.dataset.huggingface_path,
        name=name,
        split=cfg.dataset.split,
    )
    qa_pairs: List[Tuple[str, str]] = [
        (item["question"], item["answer"]) for item in dataset  # type: ignore
    ]
    return qa_pairs


def load_sql_queries(file_path: str, separator: str = "\n\n") -> List[str]:
    """Load SQL queries from a file separated by a delimiter.

    Args:
        file_path: Path to the SQL queries file
        separator: String delimiter between queries (default: double newline)

    Returns:
        List of SQL query strings

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "r") as f:
        text: str = f.read()

    # Split by separator and filter empty entries
    sql_queries: List[str] = []
    for line in text.split(separator + "\n"):
        stripped = line.strip()
        if stripped:
            sql_queries.append(stripped)
    return sql_queries


def load_translated_queries(file_path: str, separator: str = "\n\n") -> List[str]:
    """Load translated natural language queries from a file.

    Args:
        file_path: Path to the translated queries file
        separator: String delimiter between queries (default: double newline)

    Returns:
        List of translated query strings

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(file_path, "r") as f:
        text: str = f.read()

    # Split by separator and filter empty entries
    translated_queries: List[str] = []
    for line in text.split(separator):
        stripped = line.strip()
        if stripped:
            translated_queries.append(stripped)
    return translated_queries


def load_schema(schema_path: str) -> str:
    """Load schema content from a SQL file.

    Args:
        schema_path: Path to the schema SQL file

    Returns:
        Schema content as a single string

    Raises:
        FileNotFoundError: If the schema file does not exist
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        return f.read()


def load_schema_statements(schema_path: str) -> List[str]:
    """Load schema as a list of individual SQL statements.

    Args:
        schema_path: Path to the schema SQL file

    Returns:
        List of SQL statements (split by semicolon)

    Raises:
        FileNotFoundError: If the schema file does not exist
    """
    schema_content: str = load_schema(schema_path)
    # Split by semicolon and filter out empty statements
    schema_statements: List[str] = [
        stmt.strip() for stmt in schema_content.split(";") if stmt.strip()
    ]
    return schema_statements


def load_insert_statements(insert_path: str) -> List[str]:
    """Load INSERT statements from a SQL file.

    Args:
        insert_path: Path to the inserts SQL file

    Returns:
        List of INSERT SQL statements (split by semicolon)

    Raises:
        FileNotFoundError: If the insert file does not exist
    """
    if not os.path.exists(insert_path):
        raise FileNotFoundError(f"Insert file not found: {insert_path}")

    with open(insert_path, "r") as f:
        insert_content: str = f.read()

    # Split by semicolon and filter out empty statements
    insert_statements: List[str] = [
        stmt.strip() for stmt in insert_content.split(";") if stmt.strip()
    ]
    return insert_statements
