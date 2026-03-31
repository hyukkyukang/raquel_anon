"""Entity and relationship upsert generation for aligned DB builds."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_execution import quote_columns, quote_table
from src.aligned_db.schema_registry import ForeignKeyConstraint, SchemaRegistry, _quote_identifier

logger = logging.getLogger("AlignedDB")


def generate_upserts_from_entities(
    *,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    runtime: Any,
    enable_dynamic_junction: bool,
    max_junction_fix_iterations: int,
) -> List[str]:
    """Generate entity and junction-table upsert statements."""
    logger.info("Generating upsert statements from entities...")
    upserts: List[str] = []
    skipped_entities = 0

    runtime.ground_entity_references(schema_registry, entity_registry)
    runtime.auto_create_missing_referenced_entities(schema_registry, entity_registry)

    available_entity_types: Set[str] = set(entity_registry.get_entity_types())
    sorted_tables = schema_registry._topological_sort()
    ordered_entity_types: List[str] = [
        table.name for table in sorted_tables if table.name in available_entity_types
    ]
    for entity_type in entity_registry.get_entity_types():
        if entity_type not in ordered_entity_types:
            ordered_entity_types.append(entity_type)

    for entity_type in ordered_entity_types:
        entities = entity_registry.get_entities(entity_type)
        table = schema_registry.get_table(entity_type)

        if not table:
            logger.warning("No table found for entity type: %s", entity_type)
            continue

        pk = table.get_primary_key()
        type_count = 0

        conflict_key: Optional[str] = runtime.get_conflict_key(entity_type, table)
        composite_conflict_keys: Optional[List[str]] = None
        if table.is_junction_table() and len(table.primary_key_columns) >= 2:
            composite_conflict_keys = table.primary_key_columns

        fk_column_map: Dict[str, ForeignKeyConstraint] = {
            fk.column_name: fk for fk in table.foreign_keys
        }

        self_ref_fk_cols: Set[str] = runtime.get_self_referential_fk_columns(table)
        if self_ref_fk_cols:
            logger.info(
                "  %s: detected self-referential FKs: %s",
                entity_type,
                self_ref_fk_cols,
            )

        entities_needing_self_ref_update: List[Tuple[Dict[str, Any], Dict[str, str]]] = []

        for entity in entities:
            valid_columns = table.get_column_names()
            columns: List[str] = []
            values: List[str] = []
            self_ref_fk_values: Dict[str, str] = {}

            normalized_entity = runtime.normalize_entity_attributes(
                entity,
                entity_type,
                valid_columns,
            )

            for key, value in normalized_entity.items():
                if value is None or key not in valid_columns:
                    continue

                if isinstance(value, str) and not value.strip():
                    col_info = table.get_column(key)
                    if col_info and col_info.data_type.upper() != "TEXT":
                        continue

                if key == pk:
                    continue

                if key in fk_column_map:
                    if key in self_ref_fk_cols:
                        if isinstance(value, str):
                            self_ref_fk_values[key] = value
                        continue

                    if isinstance(value, str):
                        columns.append(key)
                        fk_subquery = runtime.build_fk_subquery(
                            fk_column_map[key],
                            value,
                            schema_registry,
                        )
                        values.append(fk_subquery)
                    elif isinstance(value, bool):
                        logger.warning(
                            "Skipping FK column %r with boolean value %r",
                            key,
                            value,
                        )
                    elif isinstance(value, (int, float)):
                        logger.warning(
                            "Skipping FK column %r with raw integer value %r",
                            key,
                            value,
                        )
                    else:
                        logger.warning(
                            "Skipping FK column %r with unsupported type %s",
                            key,
                            type(value).__name__,
                        )
                    continue

                col_info = table.get_column(key)
                col_type = col_info.data_type if col_info else None
                columns.append(key)
                values.append(runtime.escape_value(value, col_type))

            if not columns:
                skipped_entities += 1
                continue

            if self_ref_fk_values:
                entities_needing_self_ref_update.append((entity, self_ref_fk_values))
            has_subquery_values = any(value.startswith("(SELECT ") for value in values)

            actual_conflict_key = runtime.resolve_conflict_key(
                conflict_key,
                columns,
                entity_type,
            )
            if actual_conflict_key and actual_conflict_key in columns:
                update_cols = [column for column in columns if column != actual_conflict_key]
                quoted_conflict_key = _quote_identifier(actual_conflict_key)
                if update_cols:
                    update_clause = ", ".join(
                        f"{_quote_identifier(column)} = EXCLUDED.{_quote_identifier(column)}"
                        for column in update_cols
                    )
                    if has_subquery_values:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"SELECT {', '.join(values)} "
                            f"ON CONFLICT ({quoted_conflict_key}) DO UPDATE SET {update_clause};"
                        )
                    else:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"VALUES ({', '.join(values)}) "
                            f"ON CONFLICT ({quoted_conflict_key}) DO UPDATE SET {update_clause};"
                        )
                else:
                    if has_subquery_values:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"SELECT {', '.join(values)} "
                            f"ON CONFLICT ({quoted_conflict_key}) DO NOTHING;"
                        )
                    else:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"VALUES ({', '.join(values)}) "
                            f"ON CONFLICT ({quoted_conflict_key}) DO NOTHING;"
                        )
            elif composite_conflict_keys and all(
                key in columns for key in composite_conflict_keys
            ):
                composite_key_str = quote_columns(composite_conflict_keys)
                update_cols = [
                    column for column in columns if column not in composite_conflict_keys
                ]
                if update_cols:
                    update_clause = ", ".join(
                        f"{_quote_identifier(column)} = EXCLUDED.{_quote_identifier(column)}"
                        for column in update_cols
                    )
                    if has_subquery_values:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"SELECT {', '.join(values)} "
                            f"ON CONFLICT ({composite_key_str}) DO UPDATE SET {update_clause};"
                        )
                    else:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"VALUES ({', '.join(values)}) "
                            f"ON CONFLICT ({composite_key_str}) DO UPDATE SET {update_clause};"
                        )
                else:
                    if has_subquery_values:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"SELECT {', '.join(values)} "
                            f"ON CONFLICT ({composite_key_str}) DO NOTHING;"
                        )
                    else:
                        sql = (
                            f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                            f"VALUES ({', '.join(values)}) "
                            f"ON CONFLICT ({composite_key_str}) DO NOTHING;"
                        )
            else:
                if has_subquery_values:
                    sql = (
                        f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                        f"SELECT {', '.join(values)};"
                    )
                else:
                    sql = (
                        f"INSERT INTO {quote_table(entity_type)} ({quote_columns(columns)}) "
                        f"VALUES ({', '.join(values)});"
                    )

            upserts.append(sql)
            type_count += 1

        self_ref_update_count = 0
        for entity, self_ref_fk_values in entities_needing_self_ref_update:
            update_sql = runtime.build_self_ref_fk_update(
                entity_type,
                entity,
                self_ref_fk_values,
                fk_column_map,
                conflict_key,
                schema_registry,
            )
            if update_sql:
                upserts.append(update_sql)
                self_ref_update_count += 1

        if self_ref_update_count > 0:
            logger.info(
                "  %s: %d INSERT + %d UPDATE statements (self-ref FKs)",
                entity_type,
                type_count,
                self_ref_update_count,
            )
        else:
            logger.info("  %s: %d UPSERT statements", entity_type, type_count)

    entity_statement_count = len(upserts)
    junction_upserts: List[str] = []
    for iteration in range(max_junction_fix_iterations + 1):
        new_upserts, missing_tables = runtime.generate_junction_table_upserts(
            schema_registry,
            entity_registry,
        )
        junction_upserts.extend(new_upserts)

        if not missing_tables:
            break

        if enable_dynamic_junction and iteration < max_junction_fix_iterations:
            logger.info(
                "Creating %d missing junction tables (iteration %d)...",
                len(missing_tables),
                iteration + 1,
            )
            created = runtime.create_missing_junction_tables(
                schema_registry,
                entity_registry,
                missing_tables,
            )
            if created == 0:
                logger.warning(
                    "Could not create any of %d missing junction tables",
                    len(missing_tables),
                )
                break
        else:
            logger.warning(
                "%d junction tables still missing after %d iterations",
                len(missing_tables),
                iteration + 1,
            )

    upserts.extend(junction_upserts)

    logger.info(
        "Upsert generation complete:\n"
        "  Entity statements: %d\n"
        "  Junction table statements: %d\n"
        "  Total statements: %d\n"
        "  Skipped (empty) entities: %d",
        entity_statement_count,
        len(junction_upserts),
        len(upserts),
        skipped_entities,
    )
    return upserts
