"""Materialize an aligned DB build from saved stage logs.

This script is designed for interrupted or bounded regeneration runs where the
expensive discovery/extraction stages have already completed and been written to
``log/aligned_db_pipeline`` but the final aligned DB persistence step did not
finish. It reconstructs the schema and per-QA extractions from stage artifacts,
repopulates the target aligned database, applies the normal cleanup steps, and
saves the top-level artifacts required by downstream nullification.
"""

from __future__ import annotations

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import json
import os
from pathlib import Path
from typing import List, Tuple

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.aligned_db.db import AlignedDB
from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.verification_flow import run_iterative_verification
from src.generator.relation_normalization import filter_qa_extractions_to_schema_relations
from src.utils.db_operations import PostgresShellOperations
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


def _load_schema_registry(stage3_path: Path) -> SchemaRegistry:
    with stage3_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    tables = payload.get("tables")
    if isinstance(tables, list):
        return SchemaRegistry.from_dict(
            {"tables": {table["name"]: table for table in tables if "name" in table}}
        )
    if isinstance(tables, dict):
        return SchemaRegistry.from_dict({"tables": tables})

    sql_statements = payload.get("sql_statements")
    if isinstance(sql_statements, list) and sql_statements:
        return SchemaRegistry.from_sql_list(sql_statements)

    raise ValueError(f"Unsupported schema artifact format in {stage3_path}")


def _relax_non_junction_foreign_keys(schema_registry: SchemaRegistry) -> SchemaRegistry:
    """Drop non-junction FK metadata for stage-log materialization.

    The stage-3 saved schema is stricter than the runtime state normally used
    during a fresh build, and replaying it verbatim can suppress most entity
    inserts when referenced entities were not extracted strongly enough. For the
    materializer we keep junction-table FK metadata, but relax entity-table FKs
    so interrupted runs can still be recovered into a usable aligned DB.
    """
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None:
            continue
        if table.is_junction_table():
            continue
        table.foreign_keys = []
    return schema_registry


def _load_qa_extractions(per_qa_dir: Path) -> QAExtractionRegistry:
    extraction_paths = sorted(per_qa_dir.glob("extraction_qa_*.json"))
    if not extraction_paths:
        raise FileNotFoundError(f"No per-QA extraction artifacts found in {per_qa_dir}")

    registry = QAExtractionRegistry.empty()
    for path in extraction_paths:
        with path.open("r", encoding="utf-8") as handle:
            registry.add(QAExtraction.from_dict(json.load(handle)))
    return registry


def _load_entity_registry(stage5_path: Path) -> EntityRegistry:
    with stage5_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EntityRegistry.from_dict(payload)


def _filter_entity_registry_to_schema(
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
) -> EntityRegistry:
    valid_tables = set(schema_registry.get_table_names())
    filtered_entities = {
        entity_type: entities
        for entity_type, entities in entity_registry.entities.items()
        if entity_type in valid_tables
    }
    filtered_relationships = {
        rel_type: rows
        for rel_type, rows in entity_registry.relationships.items()
        if rel_type in valid_tables
    }
    return EntityRegistry(
        entities=filtered_entities,
        relationships=filtered_relationships,
    )


def _load_saved_qa_pairs(qa_pairs_path: Path) -> List[Tuple[str, str]]:
    with qa_pairs_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [(question, answer) for question, answer in payload]


def _ensure_target_database(cfg: DictConfig, overwrite: bool) -> None:
    db_ops = PostgresShellOperations(cfg)
    if overwrite:
        db_ops.drop_database(
            db_name=cfg.database.db_id,
            host=cfg.database.host,
            port=cfg.database.port,
            if_exists=True,
        )
        db_ops.create_database(
            db_name=cfg.database.db_id,
            host=cfg.database.host,
            port=cfg.database.port,
        )
        return

    try:
        db_ops.create_database(
            db_name=cfg.database.db_id,
            host=cfg.database.host,
            port=cfg.database.port,
        )
    except Exception:
        logger.info(
            "Database %s already exists; reusing it for materialization.",
            cfg.database.db_id,
        )


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    overwrite: bool = cfg.get("overwrite", False)
    max_retries: int = cfg.model.aligned_db.max_retries
    save_dir = Path(cfg.project_path) / cfg.model.dir_path

    stage3_path = save_dir / "log" / "aligned_db_pipeline" / "stage3_schema" / "schema.json"
    per_qa_dir = save_dir / "log" / "aligned_db_pipeline" / "stage4_extraction" / "per_qa"
    stage5_entity_registry_path = (
        save_dir
        / "log"
        / "aligned_db_pipeline"
        / "stage5_deduplication"
        / "entity_registry.json"
    )
    qa_pairs_path = save_dir / "qa_pairs.jsonl"
    verification_summary_path = save_dir / "verification_summary.json"

    if not overwrite and verification_summary_path.exists():
        logger.info(
            "Top-level build artifacts already exist at %s; use overwrite=True to rebuild.",
            save_dir,
        )
        return

    if not stage3_path.exists():
        raise FileNotFoundError(f"Missing stage 3 schema artifact: {stage3_path}")
    if not per_qa_dir.exists():
        raise FileNotFoundError(f"Missing stage 4 per-QA extraction directory: {per_qa_dir}")
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"Missing saved QA pairs: {qa_pairs_path}")

    logger.info("Loading schema and extraction artifacts from %s", save_dir)
    schema_registry = _relax_non_junction_foreign_keys(_load_schema_registry(stage3_path))
    qa_extractions = _load_qa_extractions(per_qa_dir)
    qa_extractions, removed_relations = filter_qa_extractions_to_schema_relations(
        schema_registry,
        qa_extractions,
    )
    qa_pairs = _load_saved_qa_pairs(qa_pairs_path)
    if stage5_entity_registry_path.exists():
        logger.info("Loading Stage 5 deduplicated entity registry from %s", stage5_entity_registry_path)
        entity_registry = _filter_entity_registry_to_schema(
            schema_registry,
            _load_entity_registry(stage5_entity_registry_path),
        )
    else:
        logger.info("Stage 5 entity registry artifact not found; rebuilding from per-QA extractions")
        entity_registry = EntityRegistry.from_qa_extractions(qa_extractions).deduplicate()
    _ensure_target_database(cfg, overwrite=overwrite)

    aligned_db = AlignedDB(global_cfg=cfg)
    schema_sql = [table.to_create_sql() for table in schema_registry.tables.values()]

    logger.info(
        "Materializing aligned DB from stage logs:\n"
        "  QA pairs: %d\n"
        "  Extractions: %d\n"
        "  Tables: %d\n"
        "  Relations filtered to stage-3 schema: %d",
        len(qa_pairs),
        qa_extractions.count,
        len(schema_registry.get_table_names()),
        removed_relations,
    )

    aligned_db._reset_database()
    aligned_db._execute_schema_statements(schema_sql)
    aligned_db._add_unique_constraints(schema_registry)

    upserts = aligned_db._generate_upserts_from_entities(schema_registry, entity_registry)
    aligned_db._save_upserts_log(upserts, entity_registry)
    aligned_db._execute_upserts(upserts, max_retries=max_retries)
    aligned_db._verify_data_insertion(entity_registry)

    verification_results = []
    if cfg.model.aligned_db.get("enable_round_trip_verification", True):
        logger.info("Running Stage 6 round-trip verification from recovered artifacts...")
        verification_results, total_fixes_applied = aligned_db._run_iterative_verification(
            qa_pairs=qa_pairs,
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            qa_extractions=qa_extractions,
        )
        logger.info(
            "Recovered verification complete: %d checked, %d fixes applied",
            len(verification_results),
            total_fixes_applied,
        )
    else:
        logger.info("Stage 6 verification disabled by config; skipping verification")

    if cfg.model.aligned_db.get("remove_empty_tables", True):
        aligned_db._remove_empty_tables(schema_registry)
    if cfg.model.aligned_db.get("remove_null_only_columns", True):
        aligned_db._remove_null_only_columns(schema_registry)

    aligned_db._save_results(
        schema_registry=schema_registry,
        entity_registry=entity_registry,
        qa_extractions=qa_extractions,
        verification_results=verification_results,
    )

    logger.info(
        "Aligned build materialized successfully at %s with %d upserts executed.",
        save_dir,
        len(upserts),
    )


if __name__ == "__main__":
    run_as_main(main, logger.name)
