"""Collect local performance snapshots for optimized generator/nullification paths.

This script avoids external LLM calls. It benchmarks:
1. Query-synthesis preprocessing with the new sampled path versus the old DB-wide path
2. Round-trip verifier lookup-plan execution on representative DB-derived plans
3. End-to-end nullification build time on the current aligned/null databases
"""

from script.stages.utils import init_stage

init_stage()

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from script.hotfix import apply_python314_compatibility_patches

apply_python314_compatibility_patches()

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.aligned_db.nullified_db import NullifiedDBBuilder
from src.aligned_db.schema_registry import SchemaRegistry
from src.generator.round_trip_verifier import RoundTripVerifier
from src.generator.synthesizer import QuerySynthesizer
from src.utils.data_loaders import load_schema
from src.utils.db_operations import PostgresShellOperations
from src.utils.database import PathBuilder
from src.utils.logging import get_logger
from src.utils.table_data import (
    estimate_column_statistics_from_rows,
    extract_sample_values_from_rows,
    extract_table_names_from_schema_str,
    fetch_table_data,
    get_all_column_statistics,
    get_column_sample_values,
    get_valid_join_pairs,
)

logger = get_logger(__name__, __file__)


def _load_schema_registry(path: str) -> SchemaRegistry:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SchemaRegistry.from_dict(payload)


def _query_scalar_values(
    pg_client,
    table_name: str,
    column_name: str,
    limit: int,
) -> List[Any]:
    try:
        with pg_client.conn.cursor() as cursor:
            cursor.execute(
                f'SELECT "{column_name}" FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL LIMIT {limit}'
            )
            return [row[0] for row in cursor.fetchall() if row and row[0] is not None]
    except Exception as exc:
        logger.warning("Failed to fetch sample values for %s.%s: %s", table_name, column_name, exc)
        return []


def _build_representative_lookup_plans(pg_client) -> List[Dict[str, Any]]:
    person_names = _query_scalar_values(pg_client, "person", "name", limit=3)
    work_titles = _query_scalar_values(pg_client, "work", "title", limit=3)
    genre_names = _query_scalar_values(pg_client, "genre", "name", limit=2)

    plans: List[Dict[str, Any]] = []
    for name in person_names:
        plans.append(
            {
                "label": f"person_exact:{name}",
                "plan": [{"table": "person", "conditions": {"name": str(name)}}],
            }
        )
        plans.append(
            {
                "label": f"work_author_fallback:{name}",
                "plan": [{"table": "work", "conditions": {"author": str(name)}}],
            }
        )

    for title in work_titles:
        plans.append(
            {
                "label": f"work_title:{title}",
                "plan": [{"table": "work", "conditions": {"title": str(title)}}],
            }
        )

    for genre_name in genre_names:
        plans.append(
            {
                "label": f"genre_to_work_junction:{genre_name}",
                "plan": [
                    {"table": "genre", "conditions": {"name": str(genre_name)}},
                    {"table": "work", "conditions": {"genre_id": "<genre_id_from_lookup1>"}},
                ],
            }
        )

    return plans


def _benchmark_synthesizer_preprocessing(
    synthesizer: QuerySynthesizer,
    schema: str,
    cfg: DictConfig,
) -> Dict[str, Any]:
    table_names = extract_table_names_from_schema_str(schema)

    sampled_fetch_start = time.perf_counter()
    sampled_table_data = synthesizer._get_prompt_table_data(schema)
    sampled_fetch_seconds = time.perf_counter() - sampled_fetch_start

    sampled_profile_start = time.perf_counter()
    sampled_column_stats = estimate_column_statistics_from_rows(sampled_table_data)
    sampled_values = extract_sample_values_from_rows(sampled_table_data)
    sampled_join_pairs = get_valid_join_pairs(synthesizer.pg_client, table_names)
    sampled_profile_seconds = time.perf_counter() - sampled_profile_start

    legacy_fetch_start = time.perf_counter()
    legacy_table_data = fetch_table_data(
        synthesizer.pg_client,
        table_names,
        log_fetches=False,
        max_rows_per_table=None,
    )
    legacy_fetch_seconds = time.perf_counter() - legacy_fetch_start

    legacy_profile_start = time.perf_counter()
    legacy_column_stats = get_all_column_statistics(synthesizer.pg_client, table_names)
    legacy_values = get_column_sample_values(synthesizer.pg_client, table_names)
    legacy_join_pairs = get_valid_join_pairs(synthesizer.pg_client, table_names)
    legacy_profile_seconds = time.perf_counter() - legacy_profile_start

    query_path = os.path.join(
        cfg.project_path,
        cfg.paths.data_dir,
        cfg.paths.sql_queries,
    )
    history_slice: List[str] = []
    if os.path.exists(query_path):
        with open(query_path, "r", encoding="utf-8") as handle:
            history_slice = [
                query.strip()
                for query in handle.read().split(cfg.paths.separator)
                if query.strip()
            ][:25]

    bounded_history = synthesizer._build_prompt_history(history_slice)
    history_summary = synthesizer._build_history_summary(history_slice)

    return {
        "sampled": {
            "fetch_seconds": sampled_fetch_seconds,
            "profile_seconds": sampled_profile_seconds,
            "table_count": len(sampled_table_data),
            "row_count": sum(len(rows) for rows in sampled_table_data.values()),
            "sample_cap_per_table": synthesizer.table_sample_row_limit,
            "column_stats_tables": len(sampled_column_stats),
            "sample_values_tables": len(sampled_values),
            "join_pairs": len(sampled_join_pairs),
        },
        "legacy_like": {
            "fetch_seconds": legacy_fetch_seconds,
            "profile_seconds": legacy_profile_seconds,
            "table_count": len(legacy_table_data),
            "row_count": sum(len(rows) for rows in legacy_table_data.values()),
            "column_stats_tables": len(legacy_column_stats),
            "sample_values_tables": len(legacy_values),
            "join_pairs": len(legacy_join_pairs),
        },
        "history_compaction": {
            "raw_query_count": len(history_slice),
            "raw_history_characters": sum(len(query) for query in history_slice),
            "bounded_history_query_count": len(bounded_history),
            "bounded_history_characters": sum(len(query) for query in bounded_history),
            "summary_characters": len(history_summary),
        },
    }


def _benchmark_verifier_lookup_plans(
    cfg: DictConfig,
    schema_registry: SchemaRegistry,
    pg_client,
) -> Dict[str, Any]:
    verifier = RoundTripVerifier(cfg.llm, cfg)
    plans = _build_representative_lookup_plans(pg_client)
    verifier._reset_verification_run_state(schema_registry)
    execution_results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    try:
        for item in plans:
            plan_start = time.perf_counter()
            records = verifier._execute_lookup_plan(
                pg_client,
                item["plan"],
                schema_registry=schema_registry,
            )
            execution_results.append(
                {
                    "label": item["label"],
                    "elapsed_seconds": time.perf_counter() - plan_start,
                    "tables_returned": {
                        table_name: len(rows) for table_name, rows in records.items()
                    },
                }
            )
    finally:
        stats = verifier._snapshot_verification_stats()
        verifier._clear_verification_run_state()

    return {
        "plan_count": len(plans),
        "elapsed_seconds": time.perf_counter() - total_start,
        "stats": stats,
        "plans": execution_results,
    }


def _benchmark_nullification(cfg: DictConfig) -> Dict[str, Any]:
    path_builder = PathBuilder(cfg)
    schema_path = path_builder.build_data_path(cfg.paths.schema)
    cleaned_inserts_path = path_builder.build_data_path(cfg.paths.cleaned_inserts)
    summary_path = os.path.join(
        cfg.project_path,
        cfg.model.dir_path,
        "log",
        "nullify",
        "summary.json",
    )

    logger.info("Refreshing null database from aligned DB artifacts before benchmark...")
    reset_start = time.perf_counter()
    db_ops = PostgresShellOperations(cfg)
    db_ops.drop_database(
        db_name=cfg.database.null_db_id,
        host=cfg.database.host,
        port=cfg.database.null_port,
        if_exists=True,
    )
    db_ops.create_database(
        db_name=cfg.database.null_db_id,
        host=cfg.database.host,
        port=cfg.database.null_port,
    )
    db_ops.execute_sql_file(
        db_name=cfg.database.null_db_id,
        file_path=schema_path,
        host=cfg.database.host,
        port=cfg.database.null_port,
    )
    db_ops.execute_sql_file(
        db_name=cfg.database.null_db_id,
        file_path=cleaned_inserts_path,
        host=cfg.database.host,
        port=cfg.database.null_port,
    )
    if os.path.exists(summary_path):
        os.remove(summary_path)
    reset_seconds = time.perf_counter() - reset_start

    builder = NullifiedDBBuilder(global_cfg=cfg)
    start = time.perf_counter()
    result = builder.build(overwrite=False)
    return {
        "database_reset_seconds": reset_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "entities_removed": result.entities_removed,
        "relations_removed": result.relations_removed,
        "candidate_entity_keys": result.candidate_entity_keys,
        "skipped_absent_entity_keys": result.skipped_absent_entity_keys,
        "matched_entity_keys": result.matched_entity_keys,
        "planned_entity_keys": result.planned_entity_keys,
        "cleanup_rows_deleted": result.cleanup_rows_deleted,
        "retain_verified": result.retain_verified,
        "tables_affected": len(result.tables_affected),
        "errors": list(result.errors),
        "row_comparison_tables": len(result.row_comparison),
        "log_dir": builder.nullify_log_dir_path,
    }


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    path_builder = PathBuilder(cfg)
    schema_path = path_builder.build_data_path(cfg.paths.schema)
    schema_registry_path = os.path.join(
        cfg.project_path,
        cfg.paths.data_dir,
        "aligned_db",
        "schema_registry.json",
    )

    schema = load_schema(schema_path)
    schema_registry = _load_schema_registry(schema_registry_path)
    synthesizer = QuerySynthesizer(cfg.model.synthesizer, cfg)

    logger.info("Benchmarking synthesizer preprocessing paths...")
    synthesizer_metrics = _benchmark_synthesizer_preprocessing(
        synthesizer=synthesizer,
        schema=schema,
        cfg=cfg,
    )

    logger.info("Benchmarking verifier lookup-plan execution...")
    verifier_metrics = _benchmark_verifier_lookup_plans(
        cfg=cfg,
        schema_registry=schema_registry,
        pg_client=synthesizer.pg_client,
    )

    logger.info("Benchmarking end-to-end nullification rebuild...")
    nullification_metrics = _benchmark_nullification(cfg)

    snapshot = {
        "captured_at": datetime.now().isoformat(),
        "database": {
            "db_id": cfg.database.db_id,
            "null_db_id": cfg.database.null_db_id,
            "host": cfg.database.host,
            "port": cfg.database.port,
            "null_port": cfg.database.null_port,
        },
        "synthesizer_preprocessing": synthesizer_metrics,
        "verifier_lookup_execution": verifier_metrics,
        "nullification": nullification_metrics,
    }

    output_dir = os.path.join(cfg.project_path, "results", "performance_review")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"performance_snapshot_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)

    logger.info("Saved performance snapshot to %s", output_path)
    logger.info(
        "Synth preprocessing sampled rows=%s vs legacy rows=%s",
        synthesizer_metrics["sampled"]["row_count"],
        synthesizer_metrics["legacy_like"]["row_count"],
    )
    logger.info(
        "Verifier lookup plans=%s, total table queries=%s",
        verifier_metrics["plan_count"],
        verifier_metrics["stats"]["table_queries"],
    )
    logger.info(
        "Nullification rebuild: %.3fs, entities_removed=%s, relations_removed=%s",
        nullification_metrics["elapsed_seconds"],
        nullification_metrics["entities_removed"],
        nullification_metrics["relations_removed"],
    )


if __name__ == "__main__":
    main()
