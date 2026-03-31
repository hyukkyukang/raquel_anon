"""High-level build orchestration helpers for aligned DB construction."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("AlignedDB")


@dataclass(frozen=True)
class CachedBuildArtifacts:
    """Previously saved build outputs that can be reused."""

    schema_sql: List[str]
    verification_summary: Dict[str, Any]


@dataclass(frozen=True)
class PreparedBuildArtifacts:
    """Artifacts produced by the generation stages before DB execution."""

    schema_registry: SchemaRegistry
    entity_registry: EntityRegistry
    qa_extractions: QAExtractionRegistry
    schema_sql: List[str]
    total_entities: int


@dataclass(frozen=True)
class VerificationSummaryStats:
    """Aggregated verification counts for final reporting."""

    needs_fix_count: int
    inconsistent_count: int
    passed_count: int
    avg_similarity: float


def load_cached_build_artifacts(save_dir_path: str) -> CachedBuildArtifacts:
    """Load previously saved schema and verification summary."""
    verification_summary_path = os.path.join(save_dir_path, "verification_summary.json")
    schema_path = os.path.join(save_dir_path, "schema.sql")

    cached_schema: List[str] = []
    if os.path.exists(schema_path):
        with open(schema_path, "r") as handle:
            schema_content = handle.read()
        cached_schema = [
            statement.strip()
            for statement in schema_content.split("\n\n")
            if statement.strip()
        ]

    with open(verification_summary_path, "r") as handle:
        summary = json.load(handle)

    return CachedBuildArtifacts(
        schema_sql=cached_schema,
        verification_summary=summary,
    )


def prepare_build_artifacts(
    *,
    qa_pairs: List[Tuple[str, str]],
    canonical_qa_pairs: List[Tuple[str, str]],
    normalized_qa_pairs: List[Tuple[str, str]] | None,
    naturalized_qa_pairs: List[Tuple[str, str]] | None,
    extraction_qa_pairs: List[Tuple[str, str]] | None,
    qa_sources: List[str] | None,
    aligned_db_pipeline: Any,
    save_qa_pairs_fn: Any,
    qa_pair_records: List[Dict[str, Any]] | None = None,
    qa_pair_normalization_summary: Dict[str, Any] | None = None,
    qa_pair_naturalization_summary: Dict[str, Any] | None = None,
) -> PreparedBuildArtifacts:
    """Run the discovery/extraction pipeline and prepare schema artifacts."""
    save_qa_pairs_fn(
        canonical_qa_pairs,
        normalized_qa_pairs=normalized_qa_pairs,
        naturalized_qa_pairs=naturalized_qa_pairs,
        extraction_qa_pairs=extraction_qa_pairs,
        qa_pair_records=qa_pair_records,
        qa_pair_normalization_summary=qa_pair_normalization_summary,
        qa_pair_naturalization_summary=qa_pair_naturalization_summary,
    )

    logger.info(
        "Phase 1~5: Running AlignedDBPipeline...\n"
        "  (Type discovery, attribute discovery, schema generation, extraction, deduplication)"
    )
    schema_registry, entity_registry, qa_extractions = aligned_db_pipeline.run(
        qa_pairs,
        extraction_qa_pairs=extraction_qa_pairs,
        qa_sources=qa_sources,
    )

    logger.info("Enriching schema with extracted entity attributes...")
    columns_added, _ = schema_registry.enrich_from_entities(entity_registry.entities)
    if columns_added > 0:
        logger.info("  Added %d columns discovered from extractions", columns_added)

    schema_sql = schema_registry.to_sql_list()
    logger.info("Schema generation complete: %d tables created", len(schema_sql))

    entity_types = entity_registry.get_entity_types()
    total_entities = sum(
        len(entity_registry.get_entities(entity_type)) for entity_type in entity_types
    )
    entity_details = "\n".join(
        f"    - {entity_type}: {len(entity_registry.get_entities(entity_type))} entities"
        for entity_type in entity_types
    )
    logger.info(
        "Entity extraction complete:\n"
        "  Entity types: %d\n"
        "  Total entities: %d\n"
        "%s",
        len(entity_types),
        total_entities,
        entity_details,
    )

    return PreparedBuildArtifacts(
        schema_registry=schema_registry,
        entity_registry=entity_registry,
        qa_extractions=qa_extractions,
        schema_sql=schema_sql,
        total_entities=total_entities,
    )


def summarize_verification_results(
    verification_results: List[Any],
) -> VerificationSummaryStats:
    """Compute final verification counters for build logging."""
    needs_fix_count = sum(1 for result in verification_results if result.needs_fix)
    inconsistent_count = sum(
        1 for result in verification_results if result.has_qa_inconsistency
    )
    passed_count = len(verification_results) - needs_fix_count - inconsistent_count
    avg_similarity = (
        sum(result.similarity_score for result in verification_results)
        / len(verification_results)
        if verification_results
        else 0.0
    )

    return VerificationSummaryStats(
        needs_fix_count=needs_fix_count,
        inconsistent_count=inconsistent_count,
        passed_count=passed_count,
        avg_similarity=avg_similarity,
    )
