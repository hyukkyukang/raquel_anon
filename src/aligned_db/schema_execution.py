"""Schema execution helpers for aligned database construction."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Tuple

import hkkang_utils.pg as pg_utils
import tqdm

from src.aligned_db.schema_registry import (
    ForeignKeyConstraint,
    SchemaRegistry,
    TableSchema,
    _quote_identifier,
)

logger = logging.getLogger("AlignedDB")

SCHEMA_CONTEXT_ERROR_PATTERNS: List[str] = [
    "foreign key",
    "references",
    "relation",
    "does not exist",
    "constraint",
    "type mismatch",
    "incompatible",
    "violates",
]


def is_create_table_statement(stmt: str) -> bool:
    """Return whether a statement is a CREATE TABLE statement."""
    return bool(re.match(r"^\s*CREATE\s+TABLE\b", stmt, re.IGNORECASE))


def is_alter_table_add_fk_statement(stmt: str) -> bool:
    """Return whether a statement is a deferred ALTER TABLE foreign key statement."""
    return bool(
        re.match(
            r"^\s*ALTER\s+TABLE\b.*\bFOREIGN\s+KEY\b",
            stmt,
            re.IGNORECASE | re.DOTALL,
        )
    )


def quote_columns(columns: List[str]) -> str:
    """Join column names with SQL identifier quoting where needed."""
    return ", ".join(_quote_identifier(col) for col in columns)


def quote_table(table_name: str) -> str:
    """Quote a table name if it is a reserved identifier."""
    return _quote_identifier(table_name)


def strip_foreign_keys_from_statement(
    stmt: str,
) -> Tuple[str, str, List[ForeignKeyConstraint]]:
    """Remove FK constraints from a CREATE TABLE statement for two-phase creation."""
    table_match = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\(", stmt, re.IGNORECASE)
    if not table_match:
        return ("", stmt, [])

    table_name = table_match.group(1)
    paren_start = stmt.find("(")
    if paren_start == -1:
        return (table_name, stmt, [])

    paren_depth = 0
    paren_end = -1
    for idx, char in enumerate(stmt[paren_start:], start=paren_start):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                paren_end = idx
                break

    if paren_end == -1:
        return (table_name, stmt, [])

    content = stmt[paren_start + 1 : paren_end]

    parts: List[str] = []
    current_part = ""
    paren_depth = 0
    for char in content:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "," and paren_depth == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    if current_part.strip():
        parts.append(current_part.strip())

    non_fk_parts: List[str] = []
    fk_constraints: List[ForeignKeyConstraint] = []
    for part in parts:
        part_upper = part.upper().strip()
        if part_upper.startswith("FOREIGN KEY"):
            fk_match = re.search(
                r"FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)",
                part,
                re.IGNORECASE,
            )
            if fk_match:
                fk_constraints.append(
                    ForeignKeyConstraint(
                        column_name=fk_match.group(1),
                        references_table=fk_match.group(2),
                        references_column=fk_match.group(3),
                    )
                )
        else:
            non_fk_parts.append(part)

    prefix = stmt[: paren_start + 1]
    suffix = stmt[paren_end:]
    stripped_stmt = prefix + ",\n    ".join(non_fk_parts) + suffix
    return (table_name, stripped_stmt, fk_constraints)


def needs_schema_context(error_message: str) -> bool:
    """Determine whether SQL correction should receive schema context."""
    error_lower = error_message.lower()
    return any(pattern in error_lower for pattern in SCHEMA_CONTEXT_ERROR_PATTERNS)


def correct_schema_sql(
    *,
    sql_generator: Any,
    sql: str,
    error_message: str,
    full_schema: Optional[List[str]] = None,
) -> Optional[str]:
    """Attempt to correct one schema statement using the configured SQL generator."""
    try:
        schema_context: Optional[List[str]] = None
        if full_schema and needs_schema_context(error_message):
            schema_context = [statement for statement in full_schema if statement != sql]
            logger.info(
                "Including %d related tables as schema context",
                len(schema_context),
            )

        corrected_list = sql_generator.sql_syntax_correction(
            sql,
            error_message=error_message,
            schema_context=schema_context,
        )
        if corrected_list:
            corrected = corrected_list[0]
            logger.info(
                "LLM corrected SQL based on error: %s...",
                error_message[:100],
            )
            logger.debug("Original SQL:\n%s", sql)
            logger.debug("Corrected SQL:\n%s", corrected)
            return corrected
        return None
    except Exception as exc:
        logger.warning("LLM SQL correction failed: %s", exc)
        return None


def execute_schema_statements(
    *,
    pg_client: pg_utils.PostgresConnector,
    sql_generator: Any,
    schema: List[str],
    max_correction_retries: int = 2,
    clear_lookup_column_cache: Optional[Callable[[], None]] = None,
) -> None:
    """Execute CREATE TABLE statements using a two-phase FK-safe flow."""
    create_statements = [stmt for stmt in schema if is_create_table_statement(stmt)]
    deferred_fk_statements = [
        stmt for stmt in schema if is_alter_table_add_fk_statement(stmt)
    ]
    passthrough_statements = [
        stmt
        for stmt in schema
        if not is_create_table_statement(stmt)
        and not is_alter_table_add_fk_statement(stmt)
    ]

    logger.info(
        "Executing %d CREATE TABLE statements (two-phase)...",
        len(create_statements),
    )
    if deferred_fk_statements:
        logger.info(
            "Skipping %d pre-generated ALTER TABLE foreign key statements; "
            "foreign keys will be recreated from CREATE TABLE definitions",
            len(deferred_fk_statements),
        )
    if passthrough_statements:
        logger.info(
            "Deferring %d non-table DDL statements until after table creation",
            len(passthrough_statements),
        )

    tables_without_fks: List[Tuple[str, str]] = []
    deferred_fks: List[Tuple[str, ForeignKeyConstraint]] = []
    for stmt in create_statements:
        table_name, stripped_stmt, fk_constraints = strip_foreign_keys_from_statement(
            stmt
        )
        tables_without_fks.append((table_name, stripped_stmt))
        for fk in fk_constraints:
            deferred_fks.append((table_name, fk))

    if deferred_fks:
        logger.info(
            "  Phase 1: Creating %d tables (deferring %d foreign key constraints)",
            len(tables_without_fks),
            len(deferred_fks),
        )
    else:
        logger.info("  Phase 1: Creating %d tables", len(tables_without_fks))

    for idx, (table_name, stmt) in enumerate(
        tqdm.tqdm(tables_without_fks, desc="Phase 1: Creating tables")
    ):
        current_stmt = stmt
        for attempt in range(max_correction_retries + 1):
            try:
                pg_client.execute(current_stmt)
                pg_client.conn.commit()
                if attempt > 0:
                    logger.info(
                        "  [%d/%d] Created table: %s (after %d correction(s))",
                        idx + 1,
                        len(tables_without_fks),
                        table_name,
                        attempt,
                    )
                else:
                    logger.info(
                        "  [%d/%d] Created table: %s",
                        idx + 1,
                        len(tables_without_fks),
                        table_name,
                    )
                break
            except Exception as exc:
                if attempt < max_correction_retries:
                    logger.warning(
                        "Error executing schema statement (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )
                    logger.info("Attempting LLM-based SQL correction...")
                    corrected = correct_schema_sql(
                        sql_generator=sql_generator,
                        sql=current_stmt,
                        error_message=str(exc),
                        full_schema=create_statements + passthrough_statements,
                    )
                    if corrected and corrected != current_stmt:
                        logger.info("LLM returned corrected SQL, retrying...")
                        current_stmt = corrected
                    else:
                        logger.warning(
                            "LLM correction did not produce a different SQL"
                        )
                else:
                    logger.error("Error executing schema statement: %s", current_stmt)
                    logger.error("Error: %s", exc)
                    raise

    logger.info("  Phase 1 complete: Created %d tables", len(tables_without_fks))

    if deferred_fks:
        logger.info(
            "  Phase 2: Adding %d foreign key constraints...",
            len(deferred_fks),
        )
        fk_errors: List[Tuple[str, ForeignKeyConstraint, str]] = []

        for idx, (table_name, fk) in enumerate(
            tqdm.tqdm(deferred_fks, desc="Phase 2: Adding foreign keys")
        ):
            quoted_table = _quote_identifier(table_name)
            quoted_col = _quote_identifier(fk.column_name)
            quoted_ref_table = _quote_identifier(fk.references_table)
            quoted_ref_col = _quote_identifier(fk.references_column)
            constraint_name = f"fk_{table_name}_{fk.column_name}"

            alter_stmt = (
                f"ALTER TABLE {quoted_table} ADD CONSTRAINT {constraint_name} "
                f"FOREIGN KEY ({quoted_col}) "
                f"REFERENCES {quoted_ref_table}({quoted_ref_col})"
            )

            try:
                pg_client.execute(alter_stmt)
                pg_client.conn.commit()
                logger.debug(
                    "  [%d/%d] Added FK: %s.%s -> %s.%s",
                    idx + 1,
                    len(deferred_fks),
                    table_name,
                    fk.column_name,
                    fk.references_table,
                    fk.references_column,
                )
            except Exception as exc:
                fk_errors.append((table_name, fk, str(exc)))
                logger.warning(
                    "  [%d/%d] Failed to add FK %s.%s -> %s: %s",
                    idx + 1,
                    len(deferred_fks),
                    table_name,
                    fk.column_name,
                    fk.references_table,
                    exc,
                )
                pg_client.conn.rollback()

        if fk_errors:
            logger.warning(
                "  Phase 2 complete: Added %d FKs, %d failed (likely due to missing referenced tables)",
                len(deferred_fks) - len(fk_errors),
                len(fk_errors),
            )
        else:
            logger.info(
                "  Phase 2 complete: Added all %d foreign key constraints",
                len(deferred_fks),
            )
    else:
        logger.info("  Phase 2: No foreign key constraints to add")

    if passthrough_statements:
        logger.info(
            "  Phase 3: Executing %d additional DDL statements...",
            len(passthrough_statements),
        )
        for idx, stmt in enumerate(
            tqdm.tqdm(passthrough_statements, desc="Phase 3: Executing DDL")
        ):
            current_stmt = stmt
            for attempt in range(max_correction_retries + 1):
                try:
                    pg_client.execute(current_stmt)
                    pg_client.conn.commit()
                    if attempt > 0:
                        logger.info(
                            "  [%d/%d] Executed DDL (after %d correction(s))",
                            idx + 1,
                            len(passthrough_statements),
                            attempt,
                        )
                    else:
                        logger.info(
                            "  [%d/%d] Executed DDL",
                            idx + 1,
                            len(passthrough_statements),
                        )
                    break
                except Exception as exc:
                    if attempt < max_correction_retries:
                        logger.warning(
                            "Error executing DDL statement (attempt %d): %s",
                            attempt + 1,
                            exc,
                        )
                        logger.info("Attempting LLM-based SQL correction...")
                        corrected = correct_schema_sql(
                            sql_generator=sql_generator,
                            sql=current_stmt,
                            error_message=str(exc),
                            full_schema=create_statements + passthrough_statements,
                        )
                        if corrected and corrected != current_stmt:
                            logger.info("LLM returned corrected SQL, retrying...")
                            current_stmt = corrected
                        else:
                            logger.warning(
                                "LLM correction did not produce a different SQL"
                            )
                    else:
                        logger.error("Error executing DDL statement: %s", current_stmt)
                        logger.error("Error: %s", exc)
                        raise
    else:
        logger.info("  Phase 3: No additional DDL statements to execute")

    logger.info("Successfully created all %d tables", len(create_statements))
    if clear_lookup_column_cache is not None:
        clear_lookup_column_cache()


def add_unique_constraints(
    *,
    pg_client: pg_utils.PostgresConnector,
    schema_registry: SchemaRegistry,
) -> None:
    """Add UNIQUE constraints for conflict keys to enable UPSERT operations."""
    logger.info("Adding UNIQUE constraints for UPSERT conflict keys...")
    added_count = 0
    skipped_count = 0

    for table_name in schema_registry.get_table_names():
        table: Optional[TableSchema] = schema_registry.get_table(table_name)
        if not table or table.is_junction_table():
            continue

        conflict_key = table.get_conflict_key()
        if not conflict_key or conflict_key == table.get_primary_key():
            continue
        if conflict_key not in table.get_column_names():
            continue

        constraint_name = f"{table_name}_{conflict_key}_unique"
        try:
            pg_client.execute(
                f"ALTER TABLE {quote_table(table_name)} ADD CONSTRAINT {constraint_name} "
                f"UNIQUE ({_quote_identifier(conflict_key)});"
            )
            pg_client.conn.commit()
            added_count += 1

            column = table.get_column(conflict_key)
            if column:
                column.is_unique = True

            logger.debug(
                "  Added UNIQUE constraint on %s.%s",
                table_name,
                conflict_key,
            )
        except Exception as exc:
            pg_client.conn.rollback()
            skipped_count += 1

            column = table.get_column(conflict_key)
            if column and "already exists" in str(exc).lower():
                column.is_unique = True

            logger.debug(
                "  Skipped UNIQUE constraint on %s.%s: %s",
                table_name,
                conflict_key,
                exc,
            )

    logger.info("UNIQUE constraints: %d added, %d skipped", added_count, skipped_count)
