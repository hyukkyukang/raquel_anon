"""Cleanup helpers for aligned DB tables and columns."""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import hkkang_utils.pg as pg_utils

from src.aligned_db.schema_execution import quote_table
from src.aligned_db.schema_registry import SchemaRegistry, _quote_identifier

logger = logging.getLogger("AlignedDB")


def get_existing_tables(
    *,
    pg_client: pg_utils.PostgresConnector,
) -> Set[str]:
    """Query the database for tables that actually exist."""
    try:
        cursor = pg_client.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
            """
        )
        return {row[0] for row in cursor.fetchall()}
    except Exception as exc:
        logger.warning("Could not query existing tables: %s", exc)
        return set()


def get_existing_columns(
    *,
    pg_client: pg_utils.PostgresConnector,
) -> Dict[str, Set[str]]:
    """Query the database for columns that actually exist in each table."""
    try:
        cursor = pg_client.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
        )
        columns: Dict[str, Set[str]] = {}
        for table_name, column_name in cursor.fetchall():
            columns.setdefault(table_name, set()).add(column_name)
        return columns
    except Exception as exc:
        logger.warning("Could not query existing columns: %s", exc)
        return {}


def get_reverse_dependency_order(
    *,
    schema_registry: SchemaRegistry,
    tables_to_order: List[str],
) -> List[str]:
    """Get tables in reverse dependency order for safe dropping."""
    dependencies: Dict[str, set] = {table: set() for table in tables_to_order}

    for table_name in tables_to_order:
        table = schema_registry.get_table(table_name)
        if table and table.foreign_keys:
            for fk in table.foreign_keys:
                if fk.references_table in tables_to_order:
                    dependencies[table_name].add(fk.references_table)

    result: List[str] = []
    visited: set = set()
    temp_visited: set = set()

    def visit(table: str) -> None:
        if table in temp_visited or table in visited:
            return

        temp_visited.add(table)
        for dependency in dependencies.get(table, set()):
            visit(dependency)
        temp_visited.discard(table)
        visited.add(table)
        result.append(table)

    for table in tables_to_order:
        if table not in visited:
            visit(table)

    result.reverse()
    return result


def remove_empty_tables(
    *,
    pg_client: pg_utils.PostgresConnector,
    schema_registry: SchemaRegistry,
) -> Tuple[int, List[str]]:
    """Remove tables with zero rows from the database and schema registry."""
    logger.info("Checking for empty tables to remove...")

    existing_tables = get_existing_tables(pg_client=pg_client)
    registry_tables = schema_registry.get_table_names()

    non_existent = [table for table in registry_tables if table not in existing_tables]
    if non_existent:
        for table_name in non_existent:
            if table_name in schema_registry.tables:
                del schema_registry.tables[table_name]
        logger.debug(
            "  Removed %d non-existent tables from schema registry",
            len(non_existent),
        )

    table_names = [table for table in registry_tables if table in existing_tables]
    empty_tables: List[str] = []

    for table_name in table_names:
        try:
            cursor = pg_client.execute(f"SELECT COUNT(*) FROM {quote_table(table_name)};")
            row = cursor.fetchone()
            row_count = row[0] if row else 0

            if row_count == 0:
                empty_tables.append(table_name)
                logger.debug("  Table '%s' is empty (0 rows)", table_name)
            else:
                logger.debug("  Table '%s' has %d rows", table_name, row_count)
        except Exception as exc:
            logger.warning("  Could not check table '%s': %s", table_name, exc)

    if not empty_tables:
        logger.info("No empty tables found, nothing to remove")
        return 0, []

    logger.info(
        "Found %d empty tables to remove: %s",
        len(empty_tables),
        empty_tables,
    )

    ordered_tables = get_reverse_dependency_order(
        schema_registry=schema_registry,
        tables_to_order=empty_tables,
    )

    removed_tables: List[str] = []
    for table_name in ordered_tables:
        try:
            pg_client.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            pg_client.conn.commit()

            if table_name in schema_registry.tables:
                del schema_registry.tables[table_name]

            removed_tables.append(table_name)
            logger.info("  Removed empty table: %s", table_name)
        except Exception as exc:
            logger.warning("  Failed to remove table '%s': %s", table_name, exc)
            pg_client.conn.rollback()

    logger.info(
        "Empty table cleanup complete: %d tables removed",
        len(removed_tables),
    )
    return len(removed_tables), removed_tables


def remove_null_only_columns(
    *,
    pg_client: pg_utils.PostgresConnector,
    schema_registry: SchemaRegistry,
) -> Tuple[int, Dict[str, List[str]]]:
    """Remove columns where all values are NULL."""
    logger.info("Checking for null-only columns to remove...")

    existing_tables = get_existing_tables(pg_client=pg_client)
    existing_columns = get_existing_columns(pg_client=pg_client)
    table_names = [
        table_name
        for table_name in schema_registry.get_table_names()
        if table_name in existing_tables
    ]
    removed_columns: Dict[str, List[str]] = {}
    total_removed = 0

    for table_name in table_names:
        table = schema_registry.get_table(table_name)
        if not table:
            continue

        actual_columns = existing_columns.get(table_name, set())
        if actual_columns:
            missing_registry_columns = [
                column.name for column in table.columns if column.name not in actual_columns
            ]
            if missing_registry_columns:
                table.columns = [
                    column for column in table.columns if column.name in actual_columns
                ]
                logger.debug(
                    "  Reconciled %d missing columns from schema registry for table '%s': %s",
                    len(missing_registry_columns),
                    table_name,
                    missing_registry_columns,
                )

        pk_col = table.get_primary_key()
        fk_columns = {fk.column_name for fk in table.foreign_keys}
        columns_to_check = [
            col.name
            for col in table.columns
            if col.name != pk_col
            and col.name not in fk_columns
            and col.name in actual_columns
        ]

        if not columns_to_check:
            continue

        null_only_columns: List[str] = []
        for col_name in columns_to_check:
            try:
                quoted_col = _quote_identifier(col_name)
                cursor = pg_client.execute(
                    f"SELECT EXISTS(SELECT 1 FROM {quote_table(table_name)} "
                    f"WHERE {quoted_col} IS NOT NULL LIMIT 1);"
                )
                row = cursor.fetchone()
                has_non_null = row[0] if row else False

                if not has_non_null:
                    null_only_columns.append(col_name)
                    logger.debug(
                        "  Column '%s.%s' contains only NULL values",
                        table_name,
                        col_name,
                    )
            except Exception as exc:
                logger.warning(
                    "  Could not check column '%s.%s': %s",
                    table_name,
                    col_name,
                    exc,
                )

        if not null_only_columns:
            continue

        table_removed_cols: List[str] = []
        for col_name in null_only_columns:
            try:
                pg_client.execute(
                    f"ALTER TABLE {quote_table(table_name)} "
                    f"DROP COLUMN {_quote_identifier(col_name)};"
                )
                pg_client.conn.commit()
                table.columns = [column for column in table.columns if column.name != col_name]
                table_removed_cols.append(col_name)
                total_removed += 1
                logger.debug("  Removed null-only column: %s.%s", table_name, col_name)
            except Exception as exc:
                logger.warning(
                    "  Failed to remove column '%s.%s': %s",
                    table_name,
                    col_name,
                    exc,
                )
                pg_client.conn.rollback()

        if table_removed_cols:
            removed_columns[table_name] = table_removed_cols
            logger.info(
                "  %s: removed %d null-only columns: %s",
                table_name,
                len(table_removed_cols),
                table_removed_cols,
            )

    if total_removed == 0:
        logger.info("No null-only columns found, nothing to remove")
    else:
        logger.info(
            "Null-only column cleanup complete: %d columns removed from %d tables",
            total_removed,
            len(removed_columns),
        )

    return total_removed, removed_columns
