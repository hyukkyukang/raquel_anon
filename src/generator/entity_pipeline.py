"""Six-stage pipeline for domain-agnostic database construction.

This module provides the AlignedDBPipeline that orchestrates all phases of
database construction: discovery, normalization, schema generation,
extraction, validation, and verification.

The pipeline implements a clear separation of concerns with 6 stages:
- Stage 1: Entity Type Discovery
- Stage 2: Attribute & Relation Discovery
- Stage 3: Schema Generation (from TypeRegistry)
- Stage 4: Per-QA Value Extraction (parallel)
- Stage 4.5: Extraction Validation
- Stage 5: Deduplicate & Populate DB
- Stage 6: Round-Trip Verification
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import (
    AttributeType,
    EntityType,
    RelationType,
    TypeRegistry,
)
from src.generator.pipeline_execution import (
    discover_attributes_and_relations_parallel,
    run_extraction_validation,
    run_extraction_validation_async,
    run_verification,
)
from src.generator.pipeline_flow import (
    run_discovery_and_schema_stages,
    run_extraction_and_registry_stages,
)
from src.generator.pipeline_settings import AlignedDBPipelineSettings
from src.generator.pipeline_writer import PipelineResultsWriter

if TYPE_CHECKING:
    from src.generator.attribute_discoverer import AttributeDiscoverer
    from src.generator.attribute_normalizer import AttributeNormalizer
    from src.generator.dynamic_schema_generator import DynamicSchemaGenerator
    from src.generator.entity_type_discoverer import EntityTypeDiscoverer
    from src.generator.extraction_validator import ExtractionValidator
    from src.generator.per_qa_extractor import PerQAExtractor
    from src.generator.relation_discoverer import RelationDiscoverer
    from src.generator.round_trip_verifier import RoundTripVerifier

logger = logging.getLogger("AlignedDBPipeline")


class AlignedDBPipeline:
    """Six-stage pipeline for Aligned DB generation.

    This pipeline implements a clean separation of concerns:

    Stage 1: Entity Type Discovery (discover_all)
    Stage 2: Attribute & Relation Discovery (discover_all)
    Stage 3: Schema Generation (from TypeRegistry)
    Stage 4: Per-QA Value Extraction (parallel)
    Stage 4.5: Extraction Validation
    Stage 5: Deduplicate & Populate DB
    Stage 6: Round-Trip Verification

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
        pg_client: PostgreSQL client for database operations
    """

    def __init__(
        self,
        api_cfg: DictConfig,
        global_cfg: DictConfig,
        pg_client: Optional[Any] = None,
    ) -> None:
        """Initialize the AlignedDBPipeline.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
            pg_client: PostgreSQL client for database operations (optional)
        """
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg
        self.pg_client = pg_client
        logger.info("AlignedDBPipeline initialized (6-stage architecture)")

    # =========================================================================
    # Cached Properties - Component Instances
    # =========================================================================

    @cached_property
    def entity_type_discoverer(self) -> EntityTypeDiscoverer:
        """Get the entity type discoverer."""
        from src.generator.entity_type_discoverer import EntityTypeDiscoverer

        return EntityTypeDiscoverer(self.api_cfg, self.global_cfg)

    @cached_property
    def attribute_discoverer(self) -> AttributeDiscoverer:
        """Get the attribute discoverer."""
        from src.generator.attribute_discoverer import AttributeDiscoverer

        return AttributeDiscoverer(self.api_cfg, self.global_cfg)

    @cached_property
    def relation_discoverer(self) -> RelationDiscoverer:
        """Get the relation discoverer."""
        from src.generator.relation_discoverer import RelationDiscoverer

        return RelationDiscoverer(self.api_cfg, self.global_cfg)

    @cached_property
    def attribute_normalizer(self) -> AttributeNormalizer:
        """Get the attribute normalizer."""
        from src.generator.attribute_normalizer import AttributeNormalizer

        return AttributeNormalizer(self.api_cfg, self.global_cfg)

    @cached_property
    def schema_generator(self) -> DynamicSchemaGenerator:
        """Get the dynamic schema generator."""
        from src.generator.dynamic_schema_generator import DynamicSchemaGenerator

        return DynamicSchemaGenerator(self.api_cfg, self.global_cfg)

    @cached_property
    def per_qa_extractor(self) -> PerQAExtractor:
        """Get the per-QA extractor."""
        from src.generator.per_qa_extractor import PerQAExtractor

        return PerQAExtractor(self.api_cfg, self.global_cfg)

    @cached_property
    def extraction_validator(self) -> ExtractionValidator:
        """Get the extraction validator."""
        from src.generator.extraction_validator import ExtractionValidator

        return ExtractionValidator(self.api_cfg, self.global_cfg)

    @cached_property
    def round_trip_verifier(self) -> RoundTripVerifier:
        """Get the round-trip verifier."""
        from src.generator.round_trip_verifier import RoundTripVerifier

        return RoundTripVerifier(self.api_cfg, self.global_cfg)

    @cached_property
    def settings(self) -> AlignedDBPipelineSettings:
        """Get resolved aligned-DB pipeline settings."""
        return AlignedDBPipelineSettings.from_config(self.global_cfg)

    @cached_property
    def results_writer(self) -> PipelineResultsWriter:
        """Get a conditional writer for intermediate and final stage artifacts."""
        from src.utils.results_saver import IntermediateResultsSaver

        results_saver = None
        if self.settings.save_intermediate_results:
            results_saver = IntermediateResultsSaver.from_config(
                self.global_cfg,
                "aligned_db_pipeline",
            )
        return PipelineResultsWriter(
            enabled=self.settings.save_intermediate_results,
            results_saver=results_saver,
        )

    # =========================================================================
    # Main Pipeline Method
    # =========================================================================

    def run(
        self,
        qa_pairs: List[Tuple[str, str]],
        extraction_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        qa_sources: Optional[List[str]] = None,
    ) -> Tuple[SchemaRegistry, EntityRegistry, QAExtractionRegistry]:
        """Run the complete 6-stage pipeline.

        Args:
            qa_pairs: List of (question, answer) tuples
            qa_sources: Optional list of source labels ("retain" or "forget") per QA pair.
                       Used for nullification to identify which entities to remove.

        Returns:
            Tuple of (SchemaRegistry, EntityRegistry, QAExtractionRegistry)
        """
        logger.info(
            f"\n{'='*60}\n"
            f"Starting AlignedDBPipeline\n"
            f"  QA pairs: {len(qa_pairs)}\n"
            f"  Intermediate logging: {'enabled' if self.settings.save_intermediate_results else 'disabled'}\n"
            f"{'='*60}"
        )

        discovery_artifacts = run_discovery_and_schema_stages(
            qa_pairs=qa_pairs,
            entity_type_discoverer=self.entity_type_discoverer,
            attribute_normalizer=self.attribute_normalizer,
            schema_generator=self.schema_generator,
            entity_type_batch_size=self.settings.entity_type_batch_size,
            discovery_max_workers=self.settings.discovery_max_workers,
            discover_attributes_and_relations_parallel_fn=self._discover_attributes_and_relations_parallel,
            save_stage1_results_fn=self.results_writer.save_stage1_results,
            save_stage2_results_fn=self.results_writer.save_stage2_results,
            save_stage3_results_fn=self.results_writer.save_stage3_results,
        )
        extraction_artifacts = run_extraction_and_registry_stages(
            qa_pairs=qa_pairs,
            extraction_qa_pairs=extraction_qa_pairs,
            qa_sources=qa_sources,
            schema_registry=discovery_artifacts.schema_registry,
            type_registry=discovery_artifacts.type_registry,
            per_qa_extractor=self.per_qa_extractor,
            extraction_max_concurrency=self.settings.extraction_max_concurrency,
            validation_enabled=self.settings.validation_enabled,
            run_extraction_validation_fn=self._run_extraction_validation,
            save_stage4_results_fn=self.results_writer.save_stage4_results,
            save_stage4_5_results_fn=self.results_writer.save_stage4_5_results,
            save_stage5_results_fn=self.results_writer.save_stage5_results,
        )

        # Stage 6: Verification (if enabled)
        if self.settings.verification_enabled and self.pg_client:
            logger.info("\n[Stage 6] Round-Trip Verification")
            self._run_verification(
                qa_pairs,
                extraction_artifacts.qa_extractions,
                discovery_artifacts.schema_registry,
            )

        # Save final summary
        self.results_writer.save_final_summary(
            qa_pairs,
            discovery_artifacts.entity_types,
            discovery_artifacts.attributes,
            discovery_artifacts.relations,
            discovery_artifacts.type_registry,
            discovery_artifacts.schema_registry,
            extraction_artifacts.qa_extractions,
            extraction_artifacts.entity_registry,
        )

        logger.info(f"\n{'='*60}\n" f"AlignedDBPipeline Complete\n" f"{'='*60}")

        return (
            discovery_artifacts.schema_registry,
            extraction_artifacts.entity_registry,
            extraction_artifacts.qa_extractions,
        )

    # =========================================================================
    # Stage Methods
    # =========================================================================

    def _discover_attributes_and_relations_parallel(
        self,
        qa_pairs: List[Tuple[str, str]],
        entity_types: List[EntityType],
    ) -> Tuple[Dict[str, List[AttributeType]], List[RelationType]]:
        """Run attribute and relation discovery in parallel.

        Both discoverers are independent - they only need qa_pairs and entity_types.
        Running them in parallel can significantly reduce Stage 2 time.
        Each discoverer also processes its batches in parallel internally.

        Args:
            qa_pairs: List of (question, answer) tuples
            entity_types: List of EntityType objects from Stage 1

        Returns:
            Tuple of (attributes dict, relations list)
        """
        return discover_attributes_and_relations_parallel(
            qa_pairs=qa_pairs,
            entity_types=entity_types,
            attribute_discoverer=self.attribute_discoverer,
            relation_discoverer=self.relation_discoverer,
            attribute_batch_size=self.settings.attribute_batch_size,
            max_workers=self.settings.discovery_max_workers,
        )

    def _run_extraction_validation(
        self,
        qa_pairs: List[Tuple[str, str]],
        extractions: QAExtractionRegistry,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> QAExtractionRegistry:
        """Run fact-based validation loop until coverage threshold is met.

        This method implements Stage 4.5: Extraction Validation. It extracts
        facts from answers, checks coverage against extractions, and performs
        targeted re-extraction for any missing facts.

        Uses async parallel processing for LLM calls to improve performance.

        Args:
            qa_pairs: List of (question, answer) tuples
            extractions: QAExtractionRegistry with initial extractions
            schema_registry: SchemaRegistry for schema-backed relation filtering
            type_registry: TypeRegistry for re-extraction

        Returns:
            Updated QAExtractionRegistry with validated extractions
        """
        return run_extraction_validation(
            qa_pairs=qa_pairs,
            extractions=extractions,
            schema_registry=schema_registry,
            type_registry=type_registry,
            extraction_validator=self.extraction_validator,
            validation_max_iterations=self.settings.validation_max_iterations,
            validation_coverage_threshold=self.settings.validation_coverage_threshold,
            extraction_max_concurrency=self.settings.extraction_max_concurrency,
        )

    async def _run_extraction_validation_async(
        self,
        qa_pairs: List[Tuple[str, str]],
        extractions: QAExtractionRegistry,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> QAExtractionRegistry:
        """Async implementation of extraction validation with parallel LLM calls.

        Args:
            qa_pairs: List of (question, answer) tuples
            extractions: QAExtractionRegistry with initial extractions
            schema_registry: SchemaRegistry for schema-backed relation filtering
            type_registry: TypeRegistry for re-extraction

        Returns:
            Updated QAExtractionRegistry with validated extractions
        """
        return await run_extraction_validation_async(
            qa_pairs=qa_pairs,
            extractions=extractions,
            schema_registry=schema_registry,
            type_registry=type_registry,
            extraction_validator=self.extraction_validator,
            validation_max_iterations=self.settings.validation_max_iterations,
            validation_coverage_threshold=self.settings.validation_coverage_threshold,
            extraction_max_concurrency=self.settings.extraction_max_concurrency,
        )

    def _run_verification(
        self,
        qa_pairs: List[Tuple[str, str]],
        qa_extractions: QAExtractionRegistry,
        schema_registry: SchemaRegistry,
    ) -> None:
        """Run round-trip verification stage.

        Args:
            qa_pairs: Original QA pairs
            qa_extractions: Extractions for exact table mapping
            schema_registry: Database schema
        """
        run_verification(
            qa_pairs=qa_pairs,
            qa_extractions=qa_extractions,
            schema_registry=schema_registry,
            round_trip_verifier=self.round_trip_verifier,
            pg_client=self.pg_client,
            verification_max_iterations=self.settings.verification_max_iterations,
        )
