"""Intermediate result persistence helpers for the aligned DB pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import (
    AttributeType,
    EntityType,
    RelationType,
    TypeRegistry,
)


def save_stage1_results(
    *,
    results_saver: Any,
    entity_types: List[EntityType],
) -> None:
    """Save Stage 1 entity type discovery results."""
    data = {
        "stage": "1_entity_type_discovery",
        "entity_types_count": len(entity_types),
        "entity_types": [
            {
                "name": entity_type.name,
                "description": entity_type.description,
                "examples": entity_type.examples[:5] if entity_type.examples else [],
            }
            for entity_type in entity_types
        ],
    }
    results_saver.save(
        sub_dir_name="stage1_entity_types",
        data=data,
        file_prefix="entity_types",
    )


def save_stage2_results(
    *,
    results_saver: Any,
    entity_types: List[EntityType],
    attributes: Dict[str, List[AttributeType]],
    relations: List[RelationType],
    type_registry: TypeRegistry,
) -> None:
    """Save Stage 2 attribute and relation discovery results."""
    formatted_attrs: Dict[str, List[Dict[str, Any]]] = {}
    for entity_type, attr_list in attributes.items():
        formatted_attrs[entity_type] = [
            {
                "name": attr.name,
                "data_type": attr.data_type,
                "description": attr.description,
                "is_required": attr.is_required,
                "is_unique": attr.is_unique,
                "is_natural_key": attr.is_natural_key,
            }
            for attr in attr_list
        ]

    formatted_rels = [
        {
            "name": relation.name,
            "source_entity": relation.source_entity,
            "target_entity": relation.target_entity,
            "description": relation.description,
            "attributes": [attr.name for attr in relation.attributes],
        }
        for relation in relations
    ]

    data = {
        "stage": "2_attribute_relation_discovery",
        "entity_types_count": len(entity_types),
        "total_attributes": sum(len(values) for values in attributes.values()),
        "relations_count": len(relations),
        "attributes_by_entity": formatted_attrs,
        "relations": formatted_rels,
        "type_registry_summary": str(type_registry),
        "role_inference_summary": type_registry.get_role_inference_summary(),
    }
    results_saver.save(
        sub_dir_name="stage2_attributes_relations",
        data=data,
        file_prefix="attributes_relations",
    )


def save_stage3_results(
    *,
    results_saver: Any,
    schema_registry: SchemaRegistry,
) -> None:
    """Save Stage 3 schema generation results."""
    tables_info: List[Dict[str, Any]] = []
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table:
            tables_info.append(
                {
                    "name": table.name,
                    "columns": [
                        {
                            "name": col.name,
                            "data_type": col.data_type,
                            "is_primary_key": col.is_primary_key,
                            "is_nullable": col.is_nullable,
                            "is_unique": col.is_unique,
                        }
                        for col in table.columns
                    ],
                    "primary_key_columns": table.primary_key_columns,
                    "foreign_keys": [
                        {
                            "column": fk.column_name,
                            "references": f"{fk.references_table}({fk.references_column})",
                        }
                        for fk in table.foreign_keys
                    ],
                }
            )

    data = {
        "stage": "3_schema_generation",
        "tables_count": len(schema_registry.get_table_names()),
        "tables": tables_info,
        "sql_statements": schema_registry.to_sql_list(),
    }
    results_saver.save(
        sub_dir_name="stage3_schema",
        data=data,
        file_prefix="schema",
    )


def save_stage4_results(
    *,
    results_saver: Any,
    qa_pairs: List[Tuple[str, str]],
    qa_extractions: QAExtractionRegistry,
    quality_gate_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Save Stage 4 extraction summaries and per-QA details."""
    extraction_summary = [
        {
            "qa_index": extraction.qa_index,
            "entity_count": extraction.entity_count,
            "relation_count": extraction.relation_count,
            "entities_by_type": {
                entity_type: len(entities)
                for entity_type, entities in extraction.entities.items()
            },
            "entity_metadata_count": sum(
                len(meta)
                for entity_meta in extraction.entity_attribute_metadata.values()
                for meta in entity_meta
            ),
            "relation_metadata_count": len(
                [meta for meta in extraction.relation_metadata if meta]
            ),
            "extraction_confidence": extraction.extraction_confidence,
        }
        for extraction in qa_extractions
    ]

    data = {
        "stage": "4_per_qa_extraction",
        "total_qa_pairs": len(qa_pairs),
        "total_extractions": qa_extractions.count,
        "total_entities": sum(item["entity_count"] for item in extraction_summary),
        "total_relations": sum(item["relation_count"] for item in extraction_summary),
        "extractions_summary": extraction_summary,
    }
    if quality_gate_summary is not None:
        data["quality_gate_summary"] = quality_gate_summary
    results_saver.save(
        sub_dir_name="stage4_extraction",
        data=data,
        file_prefix="extraction_summary",
    )

    for extraction in qa_extractions:
        results_saver.save(
            sub_dir_name="stage4_extraction/per_qa",
            data=extraction.to_dict(),
            suffix=f"qa_{extraction.qa_index:05d}",
            file_prefix="extraction",
        )


def save_stage4_5_results(
    *,
    results_saver: Any,
    qa_extractions: QAExtractionRegistry,
) -> None:
    """Save Stage 4.5 validation statistics."""
    valid_extractions: List[Dict[str, Any]] = []
    invalid_extractions: List[Dict[str, Any]] = []

    for extraction in qa_extractions:
        info = {
            "qa_index": extraction.qa_index,
            "validation_status": extraction.validation_status,
            "missing_facts": extraction.missing_facts,
            "entity_count": extraction.entity_count,
            "extraction_confidence": extraction.extraction_confidence,
        }
        if extraction.validation_status == "valid":
            valid_extractions.append(info)
        else:
            invalid_extractions.append(info)

    data = {
        "stage": "4.5_validation",
        "total_extractions": qa_extractions.count,
        "valid_count": len(valid_extractions),
        "invalid_count": len(invalid_extractions),
        "validation_rate": len(valid_extractions) / max(qa_extractions.count, 1),
        "valid_extractions": valid_extractions,
        "invalid_extractions": invalid_extractions,
    }
    results_saver.save(
        sub_dir_name="stage4_5_validation",
        data=data,
        file_prefix="validation_results",
    )


def save_stage5_results(
    *,
    results_saver: Any,
    entity_registry: EntityRegistry,
) -> None:
    """Save Stage 5 deduplicated entity and relationship state."""
    entity_distribution = {
        entity_type: len(entity_registry.get_entities(entity_type))
        for entity_type in entity_registry.get_entity_types()
    }
    relationship_distribution = {
        junction_table: len(entity_registry.get_relationships(junction_table))
        for junction_table in entity_registry.get_junction_tables()
    }

    data = {
        "stage": "5_deduplication",
        "entity_types_count": len(entity_registry.get_entity_types()),
        "total_entities": entity_registry.count_entities(),
        "total_relationships": entity_registry.count_relationships(),
        "entity_distribution": entity_distribution,
        "relationship_distribution": relationship_distribution,
        "entities": entity_registry.entities,
        "relationships": entity_registry.relationships,
    }
    results_saver.save(
        sub_dir_name="stage5_deduplication",
        data=data,
        file_prefix="entity_registry",
    )


def save_final_summary(
    *,
    results_saver: Any,
    qa_pairs: List[Tuple[str, str]],
    entity_types: List[EntityType],
    attributes: Dict[str, List[AttributeType]],
    relations: List[RelationType],
    schema_registry: SchemaRegistry,
    qa_extractions: QAExtractionRegistry,
    entity_registry: EntityRegistry,
) -> None:
    """Save final pipeline summary metrics."""
    valid_extractions = sum(
        1 for extraction in qa_extractions if extraction.validation_status == "valid"
    )

    data = {
        "pipeline": "AlignedDBPipeline",
        "input": {
            "qa_pairs_count": len(qa_pairs),
        },
        "stage1_entity_discovery": {
            "entity_types_discovered": len(entity_types),
            "entity_type_names": [entity_type.name for entity_type in entity_types],
        },
        "stage2_attribute_discovery": {
            "total_attributes": sum(len(values) for values in attributes.values()),
            "attributes_per_type": {
                entity_type: len(values)
                for entity_type, values in attributes.items()
            },
            "relations_discovered": len(relations),
            "relation_names": [relation.name for relation in relations],
        },
        "stage3_schema_generation": {
            "tables_count": len(schema_registry.get_table_names()),
            "table_names": schema_registry.get_table_names(),
        },
        "stage4_extraction": {
            "extractions_count": qa_extractions.count,
            "total_entities_extracted": sum(
                extraction.entity_count for extraction in qa_extractions
            ),
            "total_relations_extracted": sum(
                extraction.relation_count for extraction in qa_extractions
            ),
        },
        "stage4_5_validation": {
            "valid_extractions": valid_extractions,
            "invalid_extractions": qa_extractions.count - valid_extractions,
            "validation_rate": valid_extractions / max(qa_extractions.count, 1),
        },
        "stage5_deduplication": {
            "final_entity_types": len(entity_registry.get_entity_types()),
            "final_entities": entity_registry.count_entities(),
            "final_relationships": entity_registry.count_relationships(),
            "entity_distribution": {
                entity_type: len(entity_registry.get_entities(entity_type))
                for entity_type in entity_registry.get_entity_types()
            },
        },
    }
    results_saver.save(
        sub_dir_name="summary",
        data=data,
        file_prefix="pipeline_summary",
    )
