"""Persistence and debug helpers for aligned DB builds."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import hkkang_utils.pg as pg_utils

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_execution import quote_table
from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("AlignedDB")


def save_upserts_log(
    *,
    upserts: List[str],
    entity_registry: EntityRegistry,
    upsert_log_dir_path: str,
    grounding_summary: Dict[str, Any] | None = None,
) -> None:
    """Save generated upsert statements and summary metadata."""
    log_dir = os.path.join(upsert_log_dir_path, "upserts")
    os.makedirs(log_dir, exist_ok=True)

    upserts_path = os.path.join(log_dir, "all_upserts.sql")
    with open(upserts_path, "w", encoding="utf-8") as handle:
        for idx, sql in enumerate(upserts, start=1):
            handle.write(f"-- Statement {idx}\n")
            handle.write(sql)
            handle.write("\n\n")

    summary_path = os.path.join(log_dir, "upsert_summary.json")
    summary: Dict[str, Any] = {
        "total_statements": len(upserts),
        "entity_counts": {
            entity_type: len(entity_registry.get_entities(entity_type))
            for entity_type in entity_registry.get_entity_types()
        },
        "relationship_counts": {
            relation_type: len(entity_registry.get_relationships(relation_type))
            for relation_type in entity_registry.get_junction_tables()
        },
    }
    summary["insert_statements"] = sum(
        1 for statement in upserts if "INSERT INTO" in statement and "SELECT" not in statement
    )
    summary["select_insert_statements"] = sum(
        1 for statement in upserts if "INSERT INTO" in statement and "SELECT" in statement
    )

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if grounding_summary is not None:
        grounding_path = os.path.join(log_dir, "grounding_summary.json")
        with open(grounding_path, "w", encoding="utf-8") as handle:
            json.dump(grounding_summary, handle, indent=2)
        grounding_audit_path = os.path.join(log_dir, "grounding_audit.json")
        with open(grounding_audit_path, "w", encoding="utf-8") as handle:
            json.dump(grounding_summary, handle, indent=2)

    logger.info("Saved %d upsert statements to %s", len(upserts), upserts_path)


def verify_data_insertion(
    *,
    pg_client: pg_utils.PostgresConnector,
    entity_registry: EntityRegistry,
) -> None:
    """Verify that representative entity and junction rows reached the database."""
    logger.info("Verifying data insertion...")
    key_types = ["person", "work", "award", "location"]

    for entity_type in key_types:
        entities = entity_registry.get_entities(entity_type)
        if not entities:
            continue

        try:
            with pg_client.conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {quote_table(entity_type)}")
                db_count = cursor.fetchone()[0]

            if db_count < len(entities) * 0.9:
                logger.warning(
                    "  %s: Registry has %d, but DB only has %d rows!",
                    entity_type,
                    len(entities),
                    db_count,
                )
            else:
                logger.info(
                    "  %s: %d rows (registry: %d)",
                    entity_type,
                    db_count,
                    len(entities),
                )
        except Exception as exc:
            logger.warning("  %s: Failed to verify - %s", entity_type, exc)

    for junction_table in entity_registry.get_junction_tables()[:5]:
        relationships = entity_registry.get_relationships(junction_table)
        if not relationships:
            continue

        try:
            with pg_client.conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {quote_table(junction_table)}")
                db_count = cursor.fetchone()[0]

            if db_count < len(relationships) * 0.5:
                logger.warning(
                    "  %s: Registry has %d relations, but DB only has %d rows!",
                    junction_table,
                    len(relationships),
                    db_count,
                )
            else:
                logger.info(
                    "  %s: %d rows (registry: %d)",
                    junction_table,
                    db_count,
                    len(relationships),
                )
        except Exception as exc:
            logger.debug("  %s: Failed to verify - %s", junction_table, exc)

    logger.info("Data insertion verification complete")


def save_build_results(
    *,
    save_dir_path: str,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    qa_extractions: Any,
    verification_results: List[Any],
    extraction_cleanup_stats: Any | None = None,
) -> None:
    """Persist aligned DB artifacts for later nullification and debugging."""
    logger.info("Saving pipeline results to %s...", save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)

    schema_path = os.path.join(save_dir_path, "schema.sql")
    with open(schema_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(schema_registry.to_sql_list()))
    logger.info("  Saved schema to %s", schema_path)

    schema_registry_path = os.path.join(save_dir_path, "schema_registry.json")
    with open(schema_registry_path, "w", encoding="utf-8") as handle:
        json.dump(schema_registry.to_dict(), handle, indent=2)
    logger.info("  Saved schema registry to %s", schema_registry_path)

    entities_path = os.path.join(save_dir_path, "entities.json")
    with open(entities_path, "w", encoding="utf-8") as handle:
        handle.write(entity_registry.to_json())
    logger.info("  Saved entities to %s", entities_path)

    extractions_path = os.path.join(save_dir_path, "qa_extractions.json")
    with open(extractions_path, "w", encoding="utf-8") as handle:
        json.dump(qa_extractions.to_dict(), handle, indent=2)
    forget_count = len(qa_extractions.get_forget_extractions())
    retain_count = len(qa_extractions.get_retain_extractions())
    logger.info(
        "  Saved QA extractions to %s (%d retain, %d forget)",
        extractions_path,
        retain_count,
        forget_count,
    )

    if extraction_cleanup_stats is not None:
        cleanup_path = os.path.join(save_dir_path, "qa_extraction_cleanup.json")
        with open(cleanup_path, "w", encoding="utf-8") as handle:
            json.dump(extraction_cleanup_stats.to_dict(), handle, indent=2)
        logger.info("  Saved QA extraction cleanup stats to %s", cleanup_path)

    verification_path = os.path.join(save_dir_path, "verification_summary.json")
    summary = {
        "total": len(verification_results),
        "needs_fix": sum(1 for result in verification_results if result.needs_fix),
        "avg_similarity": (
            sum(result.similarity_score for result in verification_results)
            / len(verification_results)
            if verification_results
            else 0
        ),
    }
    with open(verification_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("  Saved verification summary to %s", verification_path)

    logger.info("All results saved successfully to %s", save_dir_path)
