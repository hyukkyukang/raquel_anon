"""Entity lookup and FK-resolution helpers for aligned DB upserts."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import hkkang_utils.pg as pg_utils

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_execution import quote_table
from src.aligned_db.schema_registry import (
    ForeignKeyConstraint,
    SchemaRegistry,
    _quote_identifier,
)
from src.aligned_db.upsert_support import normalize_fk_value

logger = logging.getLogger("AlignedDB")


def get_lookup_column_from_db(
    *,
    pg_client: pg_utils.PostgresConnector,
    table_name: str,
    cache: Dict[str, Optional[str]],
) -> Optional[str]:
    """Query the actual database to find the best lookup column for a table."""
    if table_name in cache:
        return cache[table_name]

    result: Optional[str] = None
    try:
        with pg_client.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            columns = cursor.fetchall()

            if not columns:
                cache[table_name] = None
                return None

            text_columns = [
                col[0]
                for col in columns
                if col[1].lower() in ("text", "character varying", "varchar")
            ]
            pk_col = f"{table_name}_id"

            cursor.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = 'public'
                    AND tc.table_name = %s
                    AND tc.constraint_type = 'UNIQUE'
                """,
                (table_name,),
            )
            unique_columns = {row[0] for row in cursor.fetchall()}

            unique_text_cols = [col for col in text_columns if col in unique_columns]
            if unique_text_cols:
                for pattern in [
                    "full_name",
                    f"{table_name}_name",
                    "name",
                    "title",
                    "label",
                ]:
                    if pattern in unique_text_cols:
                        result = pattern
                        break
                if not result:
                    result = unique_text_cols[0]

            if not result:
                for pattern in [
                    "full_name",
                    f"{table_name}_name",
                    "name",
                    "title",
                    "label",
                ]:
                    if pattern in text_columns:
                        result = pattern
                        break

            if not result and text_columns:
                for col in text_columns:
                    if col != pk_col:
                        result = col
                        break
    except Exception as exc:
        logger.debug("Failed to query DB schema for %s: %s", table_name, exc)

    cache[table_name] = result
    return result


def get_entity_lookup_column(
    *,
    pg_client: pg_utils.PostgresConnector,
    entity_type: str,
    schema_registry: Optional[SchemaRegistry],
    cache: Dict[str, Optional[str]],
) -> str:
    """Get the lookup column for an entity type."""
    db_lookup_col = get_lookup_column_from_db(
        pg_client=pg_client,
        table_name=entity_type,
        cache=cache,
    )
    if db_lookup_col:
        return db_lookup_col

    if schema_registry:
        table = schema_registry.get_table(entity_type)
        if table:
            column_names = table.get_column_names()
            pk_name = table.get_primary_key() or f"{entity_type}_id"

            for candidate in [
                "name",
                "title",
                "full_name",
                f"{entity_type}_name",
                "label",
            ]:
                if candidate in column_names:
                    return candidate

            for col in table.columns:
                if col.name != pk_name and col.data_type.upper() in ("TEXT", "VARCHAR"):
                    return col.name

    return "name"


def build_fk_subquery(
    *,
    fk_constraint: ForeignKeyConstraint,
    value: str,
    schema_registry: Optional[SchemaRegistry],
    get_entity_lookup_column_fn: Any,
    escape_value_fn: Any,
) -> str:
    """Build a subquery to resolve a FK string value to an integer ID."""
    ref_table = fk_constraint.references_table
    ref_column = fk_constraint.references_column
    lookup_col = get_entity_lookup_column_fn(ref_table, schema_registry)
    lookup_value = normalize_fk_value(value, ref_table)
    escaped_value = escape_value_fn(lookup_value)

    return (
        f"(SELECT t.{_quote_identifier(ref_column)} FROM {quote_table(ref_table)} t "
        f"WHERE t.{_quote_identifier(lookup_col)} = {escaped_value} LIMIT 1)"
    )


def auto_create_missing_referenced_entities(
    *,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    get_entity_lookup_column_fn: Any,
) -> None:
    """Auto-create missing entities that are referenced by FK-bearing tables."""
    created_count = 0

    for entity_type in list(entity_registry.get_entity_types()):
        table = schema_registry.get_table(entity_type)
        if not table:
            continue

        fk_columns = list(table.foreign_keys)
        if not fk_columns:
            continue

        entities = entity_registry.get_entities(entity_type)
        for entity in entities:
            for fk in fk_columns:
                fk_col_name = fk.column_name
                ref_table = fk.references_table
                ref_value = entity.get(fk_col_name)

                if not ref_value or not isinstance(ref_value, str):
                    continue

                if not schema_registry.get_table(ref_table):
                    continue

                normalized_ref_value = normalize_fk_value(ref_value, ref_table)
                lookup_col = get_entity_lookup_column_fn(ref_table, schema_registry)

                existing_entities = entity_registry.get_entities(ref_table)
                entity_exists = any(
                    existing.get(lookup_col) == normalized_ref_value
                    or existing.get("name") == normalized_ref_value
                    or existing.get(lookup_col) == ref_value
                    or existing.get("name") == ref_value
                    for existing in existing_entities
                )

                if not entity_exists:
                    stub_entity: Dict[str, Any] = {lookup_col: normalized_ref_value}
                    if lookup_col != "name":
                        stub_entity["name"] = normalized_ref_value

                    entity_registry.add_entity(ref_table, stub_entity)
                    created_count += 1
                    logger.info(
                        "  Auto-created missing %s: %s='%s' (referenced by %s)",
                        ref_table,
                        lookup_col,
                        normalized_ref_value,
                        entity_type,
                    )

    if created_count > 0:
        logger.info("Auto-created %d missing referenced entities", created_count)
