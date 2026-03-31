"""Result-writing adapter for aligned DB pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import (
    AttributeType,
    EntityType,
    RelationType,
    TypeRegistry,
)
from src.generator.pipeline_persistence import (
    save_final_summary,
    save_stage1_results,
    save_stage2_results,
    save_stage3_results,
    save_stage4_5_results,
    save_stage4_results,
    save_stage5_results,
)


@dataclass
class PipelineResultsWriter:
    """Conditional writer for intermediate/final pipeline artifacts."""

    enabled: bool
    results_saver: Optional[object]

    def save_stage1_results(self, entity_types: List[EntityType]) -> None:
        if not self.enabled or self.results_saver is None:
            return
        save_stage1_results(
            results_saver=self.results_saver,
            entity_types=entity_types,
        )

    def save_stage2_results(
        self,
        entity_types: List[EntityType],
        attributes: Dict[str, List[AttributeType]],
        relations: List[RelationType],
        type_registry: TypeRegistry,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        save_stage2_results(
            results_saver=self.results_saver,
            entity_types=entity_types,
            attributes=attributes,
            relations=relations,
            type_registry=type_registry,
        )

    def save_stage3_results(
        self,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        del type_registry
        save_stage3_results(
            results_saver=self.results_saver,
            schema_registry=schema_registry,
        )

    def save_stage4_results(
        self,
        qa_pairs: List[Tuple[str, str]],
        qa_extractions: QAExtractionRegistry,
        quality_gate_summary: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        save_stage4_results(
            results_saver=self.results_saver,
            qa_pairs=qa_pairs,
            qa_extractions=qa_extractions,
            quality_gate_summary=quality_gate_summary,
        )

    def save_stage4_5_results(
        self,
        qa_extractions: QAExtractionRegistry,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        save_stage4_5_results(
            results_saver=self.results_saver,
            qa_extractions=qa_extractions,
        )

    def save_stage5_results(
        self,
        entity_registry: EntityRegistry,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        save_stage5_results(
            results_saver=self.results_saver,
            entity_registry=entity_registry,
        )

    def save_final_summary(
        self,
        qa_pairs: List[Tuple[str, str]],
        entity_types: List[EntityType],
        attributes: Dict[str, List[AttributeType]],
        relations: List[RelationType],
        type_registry: TypeRegistry,
        schema_registry: SchemaRegistry,
        qa_extractions: QAExtractionRegistry,
        entity_registry: EntityRegistry,
    ) -> None:
        if not self.enabled or self.results_saver is None:
            return
        del type_registry
        save_final_summary(
            results_saver=self.results_saver,
            qa_pairs=qa_pairs,
            entity_types=entity_types,
            attributes=attributes,
            relations=relations,
            schema_registry=schema_registry,
            qa_extractions=qa_extractions,
            entity_registry=entity_registry,
        )
