"""Replay an aligned build from saved stages 1-3 discovery artifacts."""

from __future__ import annotations

from script.stages.utils import init_stage

init_stage()

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.aligned_db.build_flow import prepare_build_artifacts_from_frozen_discovery
from src.aligned_db.db import AlignedDB
from src.generator.frozen_discovery import (
    copy_frozen_discovery_artifacts,
    load_frozen_discovery_artifacts,
)
from src.llm.tracker import llm_call_tracker
from src.utils.db_operations import PostgresShellOperations
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_saved_qa_pairs(path: Path) -> List[Tuple[str, str]]:
    payload = _load_json(path)
    return [(str(question), str(answer)) for question, answer in payload]


def _load_qa_sources(source_dir: Path, qa_pair_count: int) -> List[str]:
    metadata_path = source_dir / "qa_pairs_metadata.json"
    if metadata_path.exists():
        payload = _load_json(metadata_path)
        records = payload.get("records", [])
        if isinstance(records, list):
            sources = [str(record.get("source", "unknown")) for record in records]
            if sources:
                return sources[:qa_pair_count]
    return ["unknown"] * qa_pair_count


def _copy_saved_qa_artifacts(source_dir: Path, target_dir: Path) -> None:
    for pattern in ("qa_pairs*.jsonl", "qa_pairs*.json"):
        for path in source_dir.glob(pattern):
            shutil.copy2(path, target_dir / path.name)


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
            "Database %s already exists; reusing it for replay.",
            cfg.database.db_id,
        )


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    replay_source_dir = cfg.get("replay_source_dir")
    if not replay_source_dir:
        raise ValueError(
            "Missing replay_source_dir. Pass +replay_source_dir=/path/to/aligned_build"
        )

    overwrite = bool(cfg.get("overwrite", False))
    max_retries = int(cfg.model.aligned_db.max_retries)
    source_dir = Path(str(replay_source_dir)).resolve()
    target_dir = (Path(cfg.project_path) / cfg.model.dir_path).resolve()

    if source_dir == target_dir:
        raise ValueError("Replay source and target directories must be different")

    source_stage_root = source_dir / "log" / "aligned_db_pipeline"
    target_stage_root = target_dir / "log" / "aligned_db_pipeline"

    qa_pairs_path = source_dir / "qa_pairs.jsonl"
    extraction_qa_pairs_path = source_dir / "qa_pairs_extraction.jsonl"
    verification_summary_path = target_dir / "verification_summary.json"

    if not source_dir.exists():
        raise FileNotFoundError(f"Replay source directory not found: {source_dir}")
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"Missing saved canonical QA pairs: {qa_pairs_path}")
    if not extraction_qa_pairs_path.exists():
        raise FileNotFoundError(
            f"Missing saved extraction QA pairs: {extraction_qa_pairs_path}"
        )
    if not overwrite and verification_summary_path.exists():
        logger.info(
            "Replay target already has top-level build artifacts at %s; use overwrite=true to rebuild.",
            target_dir,
        )
        return

    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Preparing frozen discovery replay:\n"
        "  Source aligned build: %s\n"
        "  Target aligned build: %s",
        source_dir,
        target_dir,
    )

    copy_frozen_discovery_artifacts(
        source_stage_root=source_stage_root,
        target_stage_root=target_stage_root,
    )
    _copy_saved_qa_artifacts(source_dir, target_dir)

    canonical_qa_pairs = _load_saved_qa_pairs(qa_pairs_path)
    extraction_qa_pairs = _load_saved_qa_pairs(extraction_qa_pairs_path)
    qa_sources = _load_qa_sources(source_dir, len(canonical_qa_pairs))

    aligned_db = AlignedDB(global_cfg=cfg)
    llm_call_tracker.reset()

    frozen_discovery = load_frozen_discovery_artifacts(
        stage_root=source_stage_root,
        attribute_normalizer=aligned_db.aligned_db_pipeline.attribute_normalizer,
    )

    prepared_artifacts = prepare_build_artifacts_from_frozen_discovery(
        qa_pairs=canonical_qa_pairs,
        extraction_qa_pairs=extraction_qa_pairs,
        qa_sources=qa_sources,
        frozen_discovery=frozen_discovery,
        aligned_db_pipeline=aligned_db.aligned_db_pipeline,
        schema_enrichment_cfg=dict(
            cfg.model.aligned_db.get("schema_enrichment", {})
        ),
    )

    schema_registry = prepared_artifacts.schema_registry
    entity_registry = prepared_artifacts.entity_registry
    qa_extractions = prepared_artifacts.qa_extractions
    schema_sql = prepared_artifacts.schema_sql

    _ensure_target_database(cfg, overwrite=overwrite)
    aligned_db._reset_database()
    aligned_db._execute_schema_statements(schema_sql)
    aligned_db._add_unique_constraints(schema_registry)

    upserts = aligned_db._generate_upserts_from_entities(schema_registry, entity_registry)
    aligned_db._save_upserts_log(upserts, entity_registry)
    aligned_db._execute_upserts(upserts, max_retries=max_retries)
    aligned_db._verify_data_insertion(entity_registry)

    verification_results: List[Any] = []
    total_fixes_applied = 0
    if cfg.model.aligned_db.get("enable_round_trip_verification", True):
        logger.info("Running Stage 6 round-trip verification from frozen discovery replay...")
        verification_results, total_fixes_applied = aligned_db._run_iterative_verification(
            qa_pairs=canonical_qa_pairs,
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            qa_extractions=qa_extractions,
        )

    if cfg.model.aligned_db.get("remove_empty_tables", True):
        logger.info("Phase 7: Removing empty tables...")
        aligned_db._remove_empty_tables(schema_registry)

    if cfg.model.aligned_db.get("remove_null_only_columns", True):
        logger.info("Phase 8: Removing null-only columns...")
        aligned_db._remove_null_only_columns(schema_registry)

    logger.info("Saving replayed build results...")
    aligned_db._save_results(
        schema_registry,
        entity_registry,
        qa_extractions,
        verification_results,
    )
    llm_call_tracker.log_summary(title="FrozenDiscoveryReplay LLM Statistics")

    logger.info(
        "Frozen discovery replay complete:\n"
        "  QA pairs: %d\n"
        "  Upserts executed: %d\n"
        "  Verification results: %d\n"
        "  Fixes applied: %d",
        len(canonical_qa_pairs),
        len(upserts),
        len(verification_results),
        total_fixes_applied,
    )


if __name__ == "__main__":
    run_as_main(main, logger.name)
