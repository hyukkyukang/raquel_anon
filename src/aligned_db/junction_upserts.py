"""Junction-table helpers for aligned DB upsert generation."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set, Tuple

import hkkang_utils.pg as pg_utils

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_execution import quote_columns, quote_table
from src.aligned_db.schema_registry import (
    ColumnInfo,
    ForeignKeyConstraint,
    SchemaRegistry,
    TableSchema,
    _quote_identifier,
)
from src.aligned_db.upsert_support import (
    get_entity_types_from_junction_table,
    infer_entities_from_junction_name,
    infer_entities_from_relationships,
)

logger = logging.getLogger("AlignedDB")


def create_junction_table_dynamic(
    *,
    pg_client: pg_utils.PostgresConnector,
    schema_registry: SchemaRegistry,
    table_name: str,
    source_entity: str,
    target_entity: str,
) -> None:
    """Create a junction table in the schema registry and database."""
    source_col = f"{source_entity}_id"
    target_col = f"{target_entity}_id"

    columns = [
        ColumnInfo(name=source_col, data_type="INTEGER", is_nullable=False),
        ColumnInfo(name=target_col, data_type="INTEGER", is_nullable=False),
    ]
    foreign_keys = [
        ForeignKeyConstraint(source_col, source_entity, source_col),
        ForeignKeyConstraint(target_col, target_entity, target_col),
    ]

    table = TableSchema(
        name=table_name,
        columns=columns,
        foreign_keys=foreign_keys,
        primary_key_columns=[source_col, target_col],
    )

    schema_registry.add_table(table)
    pg_client.execute(table.to_create_sql())
    pg_client.conn.commit()


def create_missing_junction_tables(
    *,
    pg_client: pg_utils.PostgresConnector,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    missing_tables: Set[str],
) -> int:
    """Dynamically create missing junction tables."""
    created_count = 0
    entity_types = set(entity_registry.get_entity_types())

    for junction_table in missing_tables:
        source, target = infer_entities_from_junction_name(junction_table, entity_types)
        if not source or not target:
            source, target = infer_entities_from_relationships(
                junction_table,
                entity_registry,
            )

        if source and target:
            if schema_registry.get_table(source) and schema_registry.get_table(target):
                create_junction_table_dynamic(
                    pg_client=pg_client,
                    schema_registry=schema_registry,
                    table_name=junction_table,
                    source_entity=source,
                    target_entity=target,
                )
                created_count += 1
                logger.info(
                    "Created junction table: %s (%s <-> %s)",
                    junction_table,
                    source,
                    target,
                )
            else:
                logger.warning(
                    "Cannot create %s: entity table(s) missing (%s exists: %s, %s exists: %s)",
                    junction_table,
                    source,
                    schema_registry.get_table(source) is not None,
                    target,
                    schema_registry.get_table(target) is not None,
                )
        else:
            logger.warning("Cannot infer entities for junction: %s", junction_table)

    return created_count


def generate_junction_table_upserts(
    *,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    runtime: Any,
) -> Tuple[List[str], Set[str]]:
    """Generate INSERT statements for junction tables using subqueries."""
    upserts: List[str] = []
    missing_tables: Set[str] = set()
    junction_tables = entity_registry.get_junction_tables()

    if not junction_tables:
        return upserts, missing_tables

    logger.info(
        "Generating junction table upserts for %d tables...",
        len(junction_tables),
    )
    grounding_resolver = runtime.build_grounding_resolver(
        schema_registry,
        entity_registry,
    )

    for junction_table in junction_tables:
        relationships = entity_registry.get_relationships(junction_table)
        table = schema_registry.get_table(junction_table)

        if not table:
            missing_tables.add(junction_table)
            continue
        if not relationships:
            continue

        entity_types = get_entity_types_from_junction_table(table, schema_registry)
        if not entity_types or len(entity_types) != 2:
            logger.warning(
                "Cannot determine entity types for junction table: %s",
                junction_table,
            )
            continue

        entity1_type, entity2_type = entity_types
        entity1_lookup_col = runtime.get_entity_lookup_column(
            entity1_type,
            schema_registry,
        )
        entity2_lookup_col = runtime.get_entity_lookup_column(
            entity2_type,
            schema_registry,
        )

        entity1_fk_col: Optional[str] = None
        entity2_fk_col: Optional[str] = None
        additional_fk_cols: List[Tuple[str, ForeignKeyConstraint]] = []

        for fk in table.foreign_keys:
            if fk.references_table == entity1_type and not entity1_fk_col:
                entity1_fk_col = fk.column_name
            elif fk.references_table == entity2_type and not entity2_fk_col:
                entity2_fk_col = fk.column_name
            else:
                additional_fk_cols.append((fk.column_name, fk))

        if not entity1_fk_col:
            entity1_fk_col = f"{entity1_type}_id"
        if not entity2_fk_col:
            entity2_fk_col = f"{entity2_type}_id"

        type_count = 0
        skipped_count = 0
        for rel in relationships:
            raw_entity1_name = (
                rel.get(f"{entity1_type}_name")
                or rel.get("source")
                or rel.get(f"{entity1_type}")
                or rel.get("name")
            )
            raw_entity2_name = (
                rel.get(f"{entity2_type}_name")
                or rel.get("target")
                or rel.get(f"{entity2_type}")
                or (rel.get("title") if entity2_type == "work" else None)
            )

            if not raw_entity1_name or not raw_entity2_name:
                skipped_count += 1
                continue

            entity1_result = grounding_resolver.resolve(
                ref_table=entity1_type,
                raw_value=str(raw_entity1_name),
                owner_type=entity2_type,
                owner_value=str(raw_entity2_name),
            )
            entity2_result = grounding_resolver.resolve(
                ref_table=entity2_type,
                raw_value=str(raw_entity2_name),
                owner_type=entity1_type,
                owner_value=str(raw_entity1_name),
            )
            entity1_name = entity1_result.resolved_value or str(raw_entity1_name).strip()
            entity2_name = entity2_result.resolved_value or str(raw_entity2_name).strip()

            entity1_escaped = runtime.escape_value(entity1_name)
            entity2_escaped = runtime.escape_value(entity2_name)

            columns = [entity1_fk_col, entity2_fk_col]
            select_parts = [
                f"(SELECT {_quote_identifier(entity1_type + '_id')} FROM {quote_table(entity1_type)} WHERE {_quote_identifier(entity1_lookup_col)} = {entity1_escaped} LIMIT 1)",
                f"(SELECT {_quote_identifier(entity2_type + '_id')} FROM {quote_table(entity2_type)} WHERE {_quote_identifier(entity2_lookup_col)} = {entity2_escaped} LIMIT 1)",
            ]
            exists_parts = [
                f"EXISTS (SELECT 1 FROM {quote_table(entity1_type)} WHERE {_quote_identifier(entity1_lookup_col)} = {entity1_escaped})",
                f"EXISTS (SELECT 1 FROM {quote_table(entity2_type)} WHERE {_quote_identifier(entity2_lookup_col)} = {entity2_escaped})",
            ]

            attr_value = rel.get("attr", "")
            for fk_col, fk_constraint in additional_fk_cols:
                if not attr_value:
                    continue

                clean_attr = attr_value
                if "=" in clean_attr:
                    clean_attr = clean_attr.split("=", 1)[-1]
                if ":" in clean_attr and not clean_attr.startswith("role"):
                    clean_attr = clean_attr.split(":", 1)[-1]
                clean_attr = clean_attr.strip()

                if clean_attr and clean_attr.lower() not in ("null", "unknown", ""):
                    ref_table = fk_constraint.references_table
                    ref_col = fk_constraint.references_column
                    lookup_col = runtime.get_entity_lookup_column(
                        ref_table,
                        schema_registry,
                    )
                    attr_result = grounding_resolver.resolve(
                        ref_table=ref_table,
                        raw_value=clean_attr,
                        owner_type=entity1_type,
                        owner_value=entity1_name,
                    )
                    resolved_attr = attr_result.resolved_value or clean_attr
                    escaped_attr = runtime.escape_value(resolved_attr)

                    columns.append(fk_col)
                    select_parts.append(
                        f"(SELECT {_quote_identifier(ref_col)} FROM {quote_table(ref_table)} WHERE {_quote_identifier(lookup_col)} ILIKE {escaped_attr} LIMIT 1)"
                    )

            sql = (
                f"INSERT INTO {quote_table(junction_table)} ({quote_columns(columns)}) "
                f"SELECT {', '.join(select_parts)} "
                f"WHERE {' AND '.join(exists_parts)} "
                f"ON CONFLICT DO NOTHING;"
            )

            upserts.append(sql)
            type_count += 1

        if skipped_count > 0:
            logger.debug(
                "  %s: skipped %d relations (missing entity names)",
                junction_table,
                skipped_count,
            )
        logger.info("  %s: %d INSERT statements", junction_table, type_count)

    if missing_tables:
        logger.debug("Missing junction tables: %s", missing_tables)

    return upserts, missing_tables
