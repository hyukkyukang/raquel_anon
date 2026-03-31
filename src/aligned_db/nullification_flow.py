"""Orchestration helpers for nullified DB builds."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("NullifiedDBBuilder")


@dataclass(frozen=True)
class PreparedNullificationArtifacts:
    """Artifacts prepared before nullification execution starts."""

    schema_registry: SchemaRegistry
    retain_entities: Set[Any]
    entities_to_remove: Set[Any]
    deletion_plan: List[Tuple[str, str, str]]
    candidate_entity_count: int
    skipped_absent_entity_count: int


def load_cached_nullification_summary(summary_path: str) -> Dict[str, Any]:
    """Load a cached nullification summary from disk."""
    with open(summary_path, "r") as handle:
        return json.load(handle)


def prepare_nullification_artifacts(
    *,
    load_qa_extractions_fn: Callable[[], Any],
    load_schema_registry_fn: Callable[[], Optional[SchemaRegistry]],
    identify_entities_by_source_fn: Callable[[Any, SchemaRegistry], tuple[Set[Any], Set[Any]]],
    compute_entities_to_remove_fn: Callable[[Set[Any], Set[Any]], Set[Any]],
    filter_existing_entities_fn: Callable[[Set[Any], SchemaRegistry], tuple[Set[Any], int]],
    compute_cascade_plan_fn: Callable[[Set[Any], SchemaRegistry], List[Tuple[str, str, str]]],
) -> Optional[PreparedNullificationArtifacts]:
    """Load prerequisite data and compute the nullification plan."""
    logger.info("Step 1: Loading QA extractions and schema registry...")
    qa_extractions = load_qa_extractions_fn()
    schema_registry = load_schema_registry_fn()

    if qa_extractions is None or schema_registry is None:
        return None

    logger.info("Step 2: Identifying forget entities...")
    forget_entities, retain_entities = identify_entities_by_source_fn(
        qa_extractions,
        schema_registry,
    )
    logger.info(
        "  Found %d forget entities, %d retain entities",
        len(forget_entities),
        len(retain_entities),
    )

    logger.info("Step 3: Computing entities to remove...")
    candidate_entities_to_remove = compute_entities_to_remove_fn(
        forget_entities,
        retain_entities,
    )
    logger.info(
        "  %d entities to remove (forget-only)",
        len(candidate_entities_to_remove),
    )

    logger.info("Step 3b: Filtering forget-only entities against aligned DB...")
    entities_to_remove, skipped_absent_entity_count = filter_existing_entities_fn(
        candidate_entities_to_remove,
        schema_registry,
    )
    logger.info(
        "  %d entities remain after aligned baseline filtering",
        len(entities_to_remove),
    )
    if skipped_absent_entity_count > 0:
        logger.info(
            "  Skipped %d forget-only entities absent from the aligned DB",
            skipped_absent_entity_count,
        )

    logger.info("Step 4: Computing FK cascade deletion order...")
    deletion_plan = compute_cascade_plan_fn(entities_to_remove, schema_registry)

    return PreparedNullificationArtifacts(
        schema_registry=schema_registry,
        retain_entities=retain_entities,
        entities_to_remove=entities_to_remove,
        deletion_plan=deletion_plan,
        candidate_entity_count=len(candidate_entities_to_remove),
        skipped_absent_entity_count=skipped_absent_entity_count,
    )
