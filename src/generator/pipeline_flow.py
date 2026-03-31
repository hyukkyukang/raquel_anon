"""Higher-level orchestration helpers for aligned DB pipeline stages."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import (
    AttributeType,
    EntityType,
    RelationType,
    TypeRegistry,
)
from src.generator.extraction_quality import build_extraction_quality_gate_summary
from src.generator.relation_normalization import filter_qa_extractions_to_schema_relations

logger = logging.getLogger("AlignedDBPipeline")


@dataclass(frozen=True)
class DiscoveryArtifacts:
    """Artifacts produced by pipeline stages 1-3."""

    entity_types: List[EntityType]
    attributes: Dict[str, List[AttributeType]]
    relations: List[RelationType]
    type_registry: TypeRegistry
    schema_registry: SchemaRegistry


@dataclass(frozen=True)
class ExtractionArtifacts:
    """Artifacts produced by pipeline stages 4-5."""

    qa_extractions: QAExtractionRegistry
    entity_registry: EntityRegistry


def run_discovery_and_schema_stages(
    *,
    qa_pairs: List[Tuple[str, str]],
    entity_type_discoverer: Any,
    attribute_normalizer: Any,
    schema_generator: Any,
    entity_type_batch_size: int,
    discovery_max_workers: int,
    discover_attributes_and_relations_parallel_fn: Callable[
        [List[Tuple[str, str]], List[EntityType]],
        Tuple[Dict[str, List[AttributeType]], List[RelationType]],
    ],
    save_stage1_results_fn: Callable[[List[EntityType]], None],
    save_stage2_results_fn: Callable[
        [List[EntityType], Dict[str, List[AttributeType]], List[RelationType], TypeRegistry],
        None,
    ],
    save_stage3_results_fn: Callable[[SchemaRegistry, TypeRegistry], None],
) -> DiscoveryArtifacts:
    """Run stages 1-3 and persist their intermediate outputs."""
    logger.info("\n[Stage 1] Entity Type Discovery (parallel)")
    entity_types = entity_type_discoverer.discover_all(
        qa_pairs,
        batch_size=entity_type_batch_size,
        max_workers=discovery_max_workers,
    )
    logger.info("  Discovered %d entity types", len(entity_types))
    save_stage1_results_fn(entity_types)

    logger.info("\n[Stage 2] Attribute & Relation Discovery (parallel)")
    del discovery_max_workers
    attributes, relations = discover_attributes_and_relations_parallel_fn(
        qa_pairs,
        entity_types,
    )
    logger.info(
        "  Discovered %d attributes, %d relations",
        sum(len(value) for value in attributes.values()),
        len(relations),
    )

    type_registry = attribute_normalizer.build_type_registry(
        entity_types, attributes, relations
    )
    logger.info("  Built %s", type_registry)
    save_stage2_results_fn(entity_types, attributes, relations, type_registry)

    logger.info("\n[Stage 3] Schema Generation")
    schema_registry = schema_generator.generate_from_registry(type_registry)
    logger.info(
        "  Generated schema with %d tables",
        len(schema_registry.get_table_names()),
    )
    save_stage3_results_fn(schema_registry, type_registry)

    return DiscoveryArtifacts(
        entity_types=entity_types,
        attributes=attributes,
        relations=relations,
        type_registry=type_registry,
        schema_registry=schema_registry,
    )


def run_extraction_and_registry_stages(
    *,
    qa_pairs: List[Tuple[str, str]],
    extraction_qa_pairs: Optional[List[Tuple[str, str]]] = None,
    qa_sources: Optional[List[str]],
    schema_registry: SchemaRegistry,
    type_registry: TypeRegistry,
    per_qa_extractor: Any,
    extraction_max_concurrency: int,
    validation_enabled: bool,
    run_extraction_validation_fn: Callable[[List[Tuple[str, str]], QAExtractionRegistry, SchemaRegistry, TypeRegistry], QAExtractionRegistry],
    save_stage4_results_fn: Callable[[List[Tuple[str, str]], QAExtractionRegistry, Dict[str, Any]], None],
    save_stage4_5_results_fn: Callable[[QAExtractionRegistry], None],
    save_stage5_results_fn: Callable[[EntityRegistry], None],
) -> ExtractionArtifacts:
    """Run stages 4-5 and persist their intermediate outputs."""
    extraction_input = extraction_qa_pairs or qa_pairs

    logger.info("\n[Stage 4] Per-QA Value Extraction")
    qa_extractions = asyncio.run(
        per_qa_extractor.extract_all(
            extraction_input,
            schema_registry,
            type_registry,
            max_concurrency=extraction_max_concurrency,
        )
    )
    logger.info("  Extracted from %d QA pairs", qa_extractions.count)

    if qa_sources:
        retain_count, forget_count = apply_source_labels(qa_extractions, qa_sources)
        logger.info(
            "  Source labels applied: %d retain, %d forget",
            retain_count,
            forget_count,
        )

    qa_extractions, removed_relations = filter_qa_extractions_to_schema_relations(
        schema_registry,
        qa_extractions,
    )
    if removed_relations:
        logger.info(
            "  Filtered %d stage-4 relations not backed by the finalized schema",
            removed_relations,
        )

    quality_gate_summary = build_extraction_quality_gate_summary(
        qa_pairs,
        qa_extractions,
        type_registry,
    )
    logger.info(
        "  Quality gate: %d abstract entities, %d abstract relation endpoints, %d template-style flags",
        quality_gate_summary.get("abstract_entity_count", 0),
        quality_gate_summary.get("abstract_relation_endpoint_count", 0),
        sum(quality_gate_summary.get("template_counts", {}).values()),
    )

    save_stage4_results_fn(qa_pairs, qa_extractions, quality_gate_summary)

    if validation_enabled:
        logger.info("\n[Stage 4.5] Fact-Based Extraction Validation")
        qa_extractions = run_extraction_validation_fn(
            qa_pairs,
            qa_extractions,
            schema_registry,
            type_registry,
        )
        qa_extractions, removed_relations = filter_qa_extractions_to_schema_relations(
            schema_registry,
            qa_extractions,
        )
        if removed_relations:
            logger.info(
                "  Filtered %d post-validation relations not backed by the finalized schema",
                removed_relations,
            )
        logger.info(
            "  Validated: %d valid, %d invalid",
            qa_extractions.valid_count,
            qa_extractions.count - qa_extractions.valid_count,
        )
        save_stage4_5_results_fn(qa_extractions)

    logger.info("\n[Stage 5] Deduplicate & Populate DB")
    entity_registry = EntityRegistry.from_qa_extractions(qa_extractions).deduplicate()
    logger.info(
        "  Entity registry: %d types, %d entities",
        len(entity_registry.get_entity_types()),
        sum(len(value) for value in entity_registry.entities.values()),
    )
    save_stage5_results_fn(entity_registry)

    return ExtractionArtifacts(
        qa_extractions=qa_extractions,
        entity_registry=entity_registry,
    )


def apply_source_labels(
    qa_extractions: QAExtractionRegistry,
    qa_sources: List[str],
) -> Tuple[int, int]:
    """Attach retain/forget source labels to per-QA extractions."""
    for extraction in qa_extractions:
        if extraction.qa_index < len(qa_sources):
            extraction.source = qa_sources[extraction.qa_index]

    retain_count = sum(1 for extraction in qa_extractions if extraction.source == "retain")
    forget_count = sum(1 for extraction in qa_extractions if extraction.source == "forget")
    return retain_count, forget_count
