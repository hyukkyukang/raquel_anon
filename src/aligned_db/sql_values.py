"""SQL value and attribute-normalization helpers for aligned DB upserts."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from src.aligned_db.schema_registry import ForeignKeyConstraint, SchemaRegistry, TableSchema

logger = logging.getLogger("AlignedDB")


def get_self_referential_fk_columns(table: TableSchema) -> Set[str]:
    """Identify FK columns that reference the same table."""
    self_ref_cols: Set[str] = set()
    for fk in table.foreign_keys:
        if fk.references_table.lower() == table.name.lower():
            self_ref_cols.add(fk.column_name)
    return self_ref_cols


def build_self_ref_fk_update(
    *,
    entity_type: str,
    entity: Dict[str, Any],
    self_ref_fk_values: Dict[str, str],
    fk_column_map: Dict[str, ForeignKeyConstraint],
    conflict_key: Optional[str],
    schema_registry: SchemaRegistry,
    build_fk_subquery: Callable[[ForeignKeyConstraint, str, Optional[SchemaRegistry]], str],
    escape_value: Callable[[Any, Optional[str]], str],
) -> Optional[str]:
    """Build an UPDATE statement to set self-referential FK values."""
    if not self_ref_fk_values or not conflict_key:
        return None

    set_clauses: List[str] = []
    for col, value in self_ref_fk_values.items():
        if col not in fk_column_map:
            continue
        fk_subquery = build_fk_subquery(fk_column_map[col], value, schema_registry)
        set_clauses.append(f"{col} = {fk_subquery}")

    if not set_clauses:
        return None

    entity_key_value = entity.get(conflict_key)
    if not entity_key_value:
        return None

    escaped_key = escape_value(entity_key_value, None)
    return (
        f"UPDATE {entity_type} SET {', '.join(set_clauses)} "
        f"WHERE {conflict_key} = {escaped_key};"
    )


def escape_value(value: Any, column_type: Optional[str] = None) -> str:
    """Escape a value for SQL, with optional type-aware coercion."""
    if value is None:
        return "NULL"

    upper_type = column_type.upper() if column_type else None

    if upper_type == "BOOLEAN":
        return coerce_to_boolean(value)
    if upper_type == "DATE":
        return coerce_to_date(value)
    if upper_type in ("INTEGER", "INT", "SERIAL", "BIGINT", "SMALLINT"):
        return coerce_to_integer(value)
    if upper_type in ("FLOAT", "REAL", "DOUBLE PRECISION", "NUMERIC", "DECIMAL"):
        return coerce_to_float(value)

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)

    str_value = str(value).strip()
    if not str_value and column_type and upper_type in (
        "DATE",
        "INTEGER",
        "FLOAT",
        "BOOLEAN",
        "SERIAL",
    ):
        return "NULL"

    escaped = str_value.replace("'", "''")
    return f"'{escaped}'"


def coerce_to_boolean(value: Any) -> str:
    """Coerce a value to PostgreSQL BOOLEAN format."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return "TRUE" if value else "FALSE"

    str_val = str(value).strip().lower()
    if not str_val:
        return "NULL"
    if str_val in ("true", "yes", "1", "t", "y", "on"):
        return "TRUE"
    if str_val in ("false", "no", "0", "f", "n", "off", "none"):
        return "FALSE"

    logger.debug("Cannot coerce %r to BOOLEAN, returning NULL", value)
    return "NULL"


def coerce_to_date(value: Any) -> str:
    """Coerce a value to PostgreSQL DATE format."""
    if value is None:
        return "NULL"

    if isinstance(value, int):
        if 1000 <= value <= 9999:
            return f"'{value}-01-01'"
        logger.debug("Integer %r out of year range, returning NULL", value)
        return "NULL"

    if isinstance(value, float):
        year = int(value)
        if 1000 <= year <= 9999:
            return f"'{year}-01-01'"
        return "NULL"

    str_val = str(value).strip()
    if not str_val:
        return "NULL"

    escaped = normalize_date_value(str_val).replace("'", "''")
    return f"'{escaped}'"


def coerce_to_integer(value: Any) -> str:
    """Coerce a value to PostgreSQL INTEGER format."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))

    str_val = str(value).strip()
    if not str_val:
        return "NULL"

    try:
        return str(int(str_val))
    except ValueError:
        pass

    try:
        return str(int(float(str_val)))
    except ValueError:
        logger.debug("Cannot coerce %r to INTEGER, returning NULL", value)
        return "NULL"


def coerce_to_float(value: Any) -> str:
    """Coerce a value to PostgreSQL FLOAT format."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1.0" if value else "0.0"
    if isinstance(value, (int, float)):
        return str(float(value))

    str_val = str(value).strip()
    if not str_val:
        return "NULL"

    try:
        return str(float(str_val))
    except ValueError:
        logger.debug("Cannot coerce %r to FLOAT, returning NULL", value)
        return "NULL"


def normalize_date_value(value: str) -> str:
    """Normalize a date string to PostgreSQL DATE format."""
    value = value.strip()
    if re.match(r"^\d{4}$", value):
        return f"{value}-01-01"
    if re.match(r"^\d{4}-\d{2}$", value):
        return f"{value}-01"
    return value


def normalize_entity_attributes(
    *,
    entity: Dict[str, Any],
    entity_type: str,
    valid_columns: List[str],
) -> Dict[str, Any]:
    """Normalize extracted entity attributes to match schema column names."""
    normalized: Dict[str, Any] = {}
    entity_name_col = f"{entity_type}_name"
    pk_col = f"{entity_type}_id"
    explicit_keys = set(entity.keys())
    fk_columns: Set[str] = {
        column for column in valid_columns if column.endswith("_id") and column != pk_col
    }

    for key, value in entity.items():
        fk_col = f"{key}_id"

        if key == "name":
            if "name" in valid_columns:
                normalized["name"] = value
            elif entity_type == "work" and "title" in valid_columns:
                normalized["title"] = value
            elif entity_name_col in valid_columns:
                normalized[entity_name_col] = value
            else:
                normalized[key] = value
        elif key == "full_name":
            if "name" in valid_columns:
                normalized["name"] = value
            elif key in valid_columns:
                normalized[key] = value
            elif entity_name_col in valid_columns:
                normalized[entity_name_col] = value
            else:
                normalized["name"] = value
        elif key == entity_name_col:
            if "name" in valid_columns:
                normalized["name"] = value
            elif key in valid_columns:
                normalized[key] = value
            else:
                normalized["name"] = value
        elif key in valid_columns:
            normalized[key] = value
            if (
                fk_col in fk_columns
                and isinstance(value, str)
                and fk_col not in explicit_keys
            ):
                normalized[fk_col] = value
        elif fk_col in fk_columns and isinstance(value, str):
            normalized[fk_col] = value
        else:
            normalized[key] = value

    return normalized
