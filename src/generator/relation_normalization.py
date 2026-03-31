"""Helpers for normalizing relation types across discovery and extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import AttributeType, RelationType, TypeRegistry
from src.utils.string import sanitize_sql_identifier

if TYPE_CHECKING:
    from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry


def _normalize_name(value: Any) -> str:
    """Normalize a relation or entity type name."""
    if not isinstance(value, str):
        return ""
    return sanitize_sql_identifier(value, default="")


def _pair_key(entity_a: str, entity_b: str) -> frozenset[str]:
    """Build a stable pair key for a relation between two entity types."""
    return frozenset({_normalize_name(entity_a), _normalize_name(entity_b)})


def build_relation_pair_index(
    relation_types: Iterable[RelationType],
) -> Dict[frozenset[str], RelationType]:
    """Index relation types by the unordered pair of endpoint entity types."""
    index: Dict[frozenset[str], RelationType] = {}
    for relation_type in relation_types:
        index.setdefault(
            _pair_key(relation_type.source_entity, relation_type.target_entity),
            relation_type,
        )
    return index


def get_allowed_relation_names(type_registry: TypeRegistry) -> List[str]:
    """Return the sorted list of supported relation types."""
    return sorted(type_registry.relation_type_names)


def format_allowed_relation_names(type_registry: TypeRegistry) -> str:
    """Format supported relation names for prompt inclusion."""
    allowed_names = get_allowed_relation_names(type_registry)
    if not allowed_names:
        return "None"
    return ", ".join(allowed_names)


def build_schema_backed_relation_registry(
    schema_registry: SchemaRegistry,
    type_registry: TypeRegistry,
) -> TypeRegistry:
    """Filter relation types to those actually materialized in the schema.

    Stage 2 can over-predict pair-based relation types that never make it into
    the final schema. Extraction and validation should only emit junction-table
    relations that the final schema really supports.
    """
    supported_tables: Set[str] = set()
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None:
            continue
        if len(table.foreign_keys) == 2:
            supported_tables.add(table_name)

    filtered_relations: List[RelationType] = []
    seen_names: Set[str] = set()

    for relation_type in type_registry.relation_types:
        if relation_type.name in supported_tables:
            filtered_relations.append(relation_type)
            seen_names.add(relation_type.name)

    for table_name in sorted(supported_tables):
        if table_name in seen_names:
            continue
        table = schema_registry.get_table(table_name)
        if table is None or len(table.foreign_keys) != 2:
            continue
        filtered_relations.append(
            RelationType(
                name=table_name,
                source_entity=table.foreign_keys[0].references_table,
                target_entity=table.foreign_keys[1].references_table,
            )
        )

    return TypeRegistry(
        entity_types=list(type_registry.entity_types),
        attribute_types=dict(type_registry.attribute_types),
        relation_types=filtered_relations,
    )


def get_schema_backed_relation_names(schema_registry: SchemaRegistry) -> Set[str]:
    """Return relation-table names that are actually materialized in the schema."""
    supported_tables: Set[str] = set()
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None:
            continue
        if table.is_junction_table():
            supported_tables.add(table_name)
    return supported_tables


def filter_qa_extractions_to_schema_relations(
    schema_registry: SchemaRegistry,
    qa_extractions: "QAExtractionRegistry",
) -> tuple["QAExtractionRegistry", int]:
    """Drop extracted relations whose types are not present in the final schema.

    Stage-2 discovery and later gap-fill can drift relative to the finalized
    stage-3 schema. This keeps the extraction artifacts aligned to the actual
    supported junction tables before persistence or DB insertion.
    """
    allowed_relations = get_schema_backed_relation_names(schema_registry)
    removed_relations = 0

    for extraction in qa_extractions:
        original_count = len(extraction.relations)
        extraction.relations = [
            relation
            for relation in extraction.relations
            if relation.get("type") in allowed_relations
        ]
        removed_relations += original_count - len(extraction.relations)
        extraction.update_relevant_tables()

    return qa_extractions, removed_relations


def _normalize_entity_value(value: Any) -> str:
    """Normalize an entity display value for matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _natural_key_for_entity_type(
    entity_type: str,
    type_registry: TypeRegistry,
) -> str:
    """Return the natural-key field to use for a relation-derived entity stub."""
    natural_key = type_registry.get_natural_key_for(entity_type)
    if natural_key:
        return natural_key
    return "title" if entity_type == "work" else "name"


def _build_extraction_entity_index(
    extraction: "QAExtraction",
    type_registry: TypeRegistry,
) -> Dict[str, Dict[str, str]]:
    """Build a normalized lookup index of extracted entity display values."""
    index: Dict[str, Dict[str, str]] = {}

    for entity_type, entities in extraction.entities.items():
        natural_key = _natural_key_for_entity_type(entity_type, type_registry)
        values = index.setdefault(entity_type, {})
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            candidate_keys = [natural_key, "name", "title", f"{entity_type}_name"]
            for key in candidate_keys:
                raw_value = entity.get(key)
                normalized = _normalize_entity_value(raw_value)
                if normalized:
                    values.setdefault(normalized, str(raw_value).strip())
    return index


def backfill_relation_entities(
    extraction: "QAExtraction",
    type_registry: TypeRegistry,
    *,
    is_generic_work_title: Optional[Callable[[str], bool]] = None,
) -> tuple["QAExtraction", int, int]:
    """Backfill missing relation endpoints as minimal entities.

    This improves relation retention by ensuring valid junction-table relations
    have concrete endpoint entities to link against. It also swaps source/target
    when both clearly match the opposite endpoint types.
    """
    entity_index = _build_extraction_entity_index(extraction, type_registry)
    added_entities = 0
    swapped_relations = 0

    for relation in extraction.relations:
        relation_type = relation.get("type")
        if not isinstance(relation_type, str):
            continue

        relation_def = type_registry.get_relation_type(relation_type)
        if relation_def is None:
            continue

        source_value = str(relation.get("source", "")).strip()
        target_value = str(relation.get("target", "")).strip()
        if not source_value or not target_value:
            continue

        source_type = relation_def.source_entity
        target_type = relation_def.target_entity

        source_norm = _normalize_entity_value(source_value)
        target_norm = _normalize_entity_value(target_value)

        source_matches_source = source_norm in entity_index.get(source_type, {})
        target_matches_target = target_norm in entity_index.get(target_type, {})
        source_matches_target = source_norm in entity_index.get(target_type, {})
        target_matches_source = target_norm in entity_index.get(source_type, {})

        should_swap = (
            not source_matches_source
            and source_matches_target
            and (
                target_matches_source
                or not target_matches_target
            )
        )

        if should_swap:
            relation["source"], relation["target"] = target_value, source_value
            source_value, target_value = target_value, source_value
            source_norm, target_norm = target_norm, source_norm
            swapped_relations += 1

        for entity_type, raw_value, normalized_value in (
            (source_type, source_value, source_norm),
            (target_type, target_value, target_norm),
        ):
            if normalized_value in entity_index.get(entity_type, {}):
                continue

            if (
                entity_type == "work"
                and is_generic_work_title is not None
                and is_generic_work_title(raw_value)
            ):
                continue

            natural_key = _natural_key_for_entity_type(entity_type, type_registry)
            stub_entity = {natural_key: raw_value}
            extraction.add_entity(entity_type, stub_entity)
            entity_index.setdefault(entity_type, {})[normalized_value] = raw_value
            added_entities += 1

    if added_entities or swapped_relations:
        extraction.update_relevant_tables()

    return extraction, added_entities, swapped_relations


def normalize_extracted_relation(
    relation: Dict[str, Any],
    type_registry: TypeRegistry,
) -> Optional[Dict[str, Any]]:
    """Normalize an extracted relation record to a supported canonical type.

    Accepts exact relation names and reversible `<entity_a>_<entity_b>` aliases
    that correspond to a known relation-type entity pair. Unsupported semantic
    relation labels are dropped.
    """
    raw_type = _normalize_name(relation.get("type"))
    if not raw_type:
        return None

    canonical_relation = type_registry.get_relation_type(raw_type)
    swap_endpoints = False

    if canonical_relation is None:
        parts = raw_type.split("_")
        if len(parts) != 2:
            return None

        pair_index = build_relation_pair_index(type_registry.relation_types)
        canonical_relation = pair_index.get(_pair_key(parts[0], parts[1]))
        if canonical_relation is None:
            return None

        swap_endpoints = (
            parts[0] == _normalize_name(canonical_relation.target_entity)
            and parts[1] == _normalize_name(canonical_relation.source_entity)
        )

    source = relation.get("source")
    target = relation.get("target")
    if source is None or target is None:
        return None

    cleaned_source = str(source).strip()
    cleaned_target = str(target).strip()
    if not cleaned_source or not cleaned_target:
        return None

    cleaned_relation = dict(relation)
    cleaned_relation["type"] = canonical_relation.name
    cleaned_relation["source"] = cleaned_source
    cleaned_relation["target"] = cleaned_target

    if swap_endpoints:
        cleaned_relation["source"], cleaned_relation["target"] = (
            cleaned_relation["target"],
            cleaned_relation["source"],
        )

    return cleaned_relation


def normalize_discovered_relation(
    item: Dict[str, Any],
    *,
    allowed_entity_names: Set[str],
    preferred_relations: Sequence[RelationType],
) -> Optional[RelationType]:
    """Normalize a discovered relation to a pair-based junction-table relation.

    The discovery stage only supports many-to-many junction tables between
    known entity types. Semantic predicate labels like `author_of` are mapped
    to a canonical pair-based name when the endpoints are valid, otherwise
    discarded.
    """
    source_entity = _normalize_name(item.get("source_entity"))
    target_entity = _normalize_name(item.get("target_entity"))
    if not source_entity or not target_entity:
        return None
    if source_entity == target_entity:
        return None
    if source_entity not in allowed_entity_names or target_entity not in allowed_entity_names:
        return None

    preferred_index = build_relation_pair_index(preferred_relations)
    preferred_relation = preferred_index.get(_pair_key(source_entity, target_entity))
    if preferred_relation is not None:
        canonical_name = preferred_relation.name
        canonical_source = preferred_relation.source_entity
        canonical_target = preferred_relation.target_entity
    else:
        ordered_entities = sorted([source_entity, target_entity])
        canonical_name = f"{ordered_entities[0]}_{ordered_entities[1]}"
        canonical_source, canonical_target = ordered_entities

    additional_attrs: List[AttributeType] = []
    for attr_name in item.get("additional_attributes", []):
        if isinstance(attr_name, str) and attr_name.strip():
            additional_attrs.append(
                AttributeType(
                    name=attr_name.strip(),
                    data_type="TEXT",
                    description="Additional attribute on relationship",
                )
            )

    examples = item.get("examples", [])
    if not isinstance(examples, list):
        examples = []

    return RelationType(
        name=canonical_name,
        source_entity=canonical_source,
        target_entity=canonical_target,
        description=str(item.get("description", "")).strip(),
        attributes=additional_attrs,
        examples=examples,
    )
