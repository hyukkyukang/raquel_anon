"""Helpers for pruning saved QA extractions against the aligned DB baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("AlignedDB")


@dataclass(frozen=True)
class ExtractionCleanupStats:
    """Summary of QA-extraction pruning against the aligned DB."""

    total_entities_before: int
    total_entities_after: int
    removed_entities: int
    touched_extractions: int
    removed_by_type: Dict[str, int]
    total_relations_before: int
    total_relations_after: int
    removed_relations: int
    relinked_relations: int
    removed_relations_by_type: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a serializable dictionary."""
        return {
            "total_entities_before": self.total_entities_before,
            "total_entities_after": self.total_entities_after,
            "removed_entities": self.removed_entities,
            "touched_extractions": self.touched_extractions,
            "removed_by_type": self.removed_by_type,
            "total_relations_before": self.total_relations_before,
            "total_relations_after": self.total_relations_after,
            "removed_relations": self.removed_relations,
            "relinked_relations": self.relinked_relations,
            "removed_relations_by_type": self.removed_relations_by_type,
        }


@dataclass(frozen=True)
class RelationCleanupMetadata:
    """Relation-table metadata needed for pruning and canonicalization."""

    canonical_type: str
    source_table: str
    target_table: str


def _resolve_entity_key(
    entity_type: str,
    entity: Dict[str, Any],
    schema_registry: SchemaRegistry,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the natural-key column/value for an extracted entity."""
    table = schema_registry.get_table(entity_type)
    if table is None:
        return None, None

    conflict_key = table.get_conflict_key()
    valid_columns = {col.name for col in table.columns}
    conflict_column = table.get_column(conflict_key) if conflict_key else None

    alias_keys = [conflict_key, "name", "title", "full_name", f"{entity_type}_name"]
    if (
        conflict_key
        and conflict_column is not None
        and conflict_column.data_type.upper() in ("TEXT", "VARCHAR")
    ):
        for alias_key in alias_keys:
            if not alias_key:
                continue
            value = entity.get(alias_key)
            if value:
                return conflict_key, str(value)

    fallback_keys = ["name", "title", "full_name", f"{entity_type}_name"]
    for fallback_key in fallback_keys:
        if fallback_key not in valid_columns:
            continue
        value = entity.get(fallback_key)
        if value:
            return fallback_key, str(value)

    return None, None


def _normalize_lookup_value(value: Any) -> str:
    """Normalize a candidate lookup value for comparison."""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _build_reference_value_maps(
    *,
    pg_client: Any,
    schema_registry: SchemaRegistry,
    table_names: Set[str],
    existing_tables: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Load canonical aligned-DB lookup values for a set of entity tables."""
    value_maps: Dict[str, Dict[str, str]] = {}
    cursor = pg_client.conn.cursor()
    try:
        for table_name in sorted(table_names):
            if existing_tables is not None and table_name not in existing_tables:
                continue
            table = schema_registry.get_table(table_name)
            if table is None:
                continue
            lookup_column = table.get_conflict_key()
            if not lookup_column:
                continue

            query = (
                f'SELECT "{lookup_column}" FROM "{table_name}" '
                f'WHERE "{lookup_column}" IS NOT NULL'
            )
            cursor.execute(query)
            value_maps[table_name] = {}
            for (raw_value,) in cursor.fetchall():
                if raw_value is None:
                    continue
                normalized = _normalize_lookup_value(raw_value)
                value_maps[table_name].setdefault(normalized, str(raw_value))
    finally:
        cursor.close()

    return value_maps


def _load_existing_tables(pg_client: Any) -> Set[str]:
    """Load the set of currently materialized tables in the aligned DB."""
    cursor = pg_client.conn.cursor()
    try:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {str(row[0]) for row in cursor.fetchall() if row and row[0]}
    finally:
        cursor.close()


def _build_relation_cleanup_metadata(
    schema_registry: SchemaRegistry,
) -> Dict[str, RelationCleanupMetadata]:
    """Build relation cleanup metadata for supported junction tables."""
    metadata: Dict[str, RelationCleanupMetadata] = {}

    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None or len(table.foreign_keys) != 2:
            continue

        parts = table_name.split("_")
        if len(parts) == 2:
            source_table, target_table = parts
        else:
            source_table = table.foreign_keys[0].references_table
            target_table = table.foreign_keys[1].references_table

        metadata[table_name] = RelationCleanupMetadata(
            canonical_type=table_name,
            source_table=source_table,
            target_table=target_table,
        )

    return metadata


def _canonicalize_relation_type(
    relation_type: Any,
    relation_metadata: Dict[str, RelationCleanupMetadata],
) -> Optional[str]:
    """Map a relation type to a supported canonical junction-table name."""
    if not isinstance(relation_type, str):
        return None
    normalized_type = relation_type.strip()
    if normalized_type in relation_metadata:
        return normalized_type

    parts = normalized_type.split("_")
    if len(parts) == 2:
        reversed_type = f"{parts[1]}_{parts[0]}"
        if reversed_type in relation_metadata:
            return reversed_type

    return None


def _extract_relation_endpoints(
    relation: Dict[str, Any],
    relation_meta: RelationCleanupMetadata,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract raw source/target values from a relation record."""
    source_value = relation.get("source")
    target_value = relation.get("target")

    if not source_value:
        source_value = relation.get(f"{relation_meta.source_table}_name")
    if not target_value:
        target_value = relation.get(f"{relation_meta.target_table}_name")

    if source_value is not None:
        source_value = str(source_value)
    if target_value is not None:
        target_value = str(target_value)

    return source_value, target_value


def _resolve_relation_endpoints(
    *,
    source_value: Optional[str],
    target_value: Optional[str],
    relation_meta: RelationCleanupMetadata,
    reference_value_maps: Dict[str, Dict[str, str]],
) -> Optional[Tuple[str, str, bool]]:
    """Resolve relation endpoints to canonical aligned-DB names."""
    if not source_value or not target_value:
        return None

    source_map = reference_value_maps.get(relation_meta.source_table, {})
    target_map = reference_value_maps.get(relation_meta.target_table, {})

    direct_source = source_map.get(_normalize_lookup_value(source_value))
    direct_target = target_map.get(_normalize_lookup_value(target_value))
    if direct_source and direct_target:
        relinked = (
            direct_source != source_value
            or direct_target != target_value
        )
        return direct_source, direct_target, relinked

    swapped_source = source_map.get(_normalize_lookup_value(target_value))
    swapped_target = target_map.get(_normalize_lookup_value(source_value))
    if swapped_source and swapped_target:
        relinked = True
        return swapped_source, swapped_target, relinked

    return None


def prune_qa_extractions_to_aligned_db(
    *,
    pg_client: Any,
    schema_registry: SchemaRegistry,
    qa_extractions: QAExtractionRegistry,
) -> Tuple[QAExtractionRegistry, ExtractionCleanupStats]:
    """Prune extracted entities that do not exist in the aligned DB."""
    candidate_values: Dict[Tuple[str, str], Set[str]] = {}
    total_entities_before = 0
    existing_tables = _load_existing_tables(pg_client)

    for extraction in qa_extractions:
        total_entities_before += extraction.entity_count
        for entity_type, entities in extraction.entities.items():
            for entity in entities:
                key_column, key_value = _resolve_entity_key(
                    entity_type, entity, schema_registry
                )
                if key_column and key_value:
                    candidate_values.setdefault((entity_type, key_column), set()).add(
                        key_value
                    )

    existing_values: Dict[Tuple[str, str], Set[str]] = {}
    cursor = pg_client.conn.cursor()
    try:
        for (table_name, key_column), values in candidate_values.items():
            existing_values[(table_name, key_column)] = set()
            if table_name not in existing_tables:
                continue
            if not values:
                continue

            ordered_values = list(values)
            for index in range(0, len(ordered_values), 500):
                chunk = ordered_values[index : index + 500]
                placeholders = ", ".join(["%s"] * len(chunk))
                query = (
                    f'SELECT "{key_column}" FROM "{table_name}" '
                    f'WHERE "{key_column}" IN ({placeholders})'
                )
                cursor.execute(query, tuple(chunk))
                existing_values[(table_name, key_column)].update(
                    str(row[0]) for row in cursor.fetchall()
                )
    finally:
        cursor.close()

    cleaned_registry = QAExtractionRegistry.empty()
    removed_by_type: Dict[str, int] = {}
    touched_extractions = 0
    total_entities_after = 0
    total_relations_before = sum(extraction.relation_count for extraction in qa_extractions)
    total_relations_after = 0
    relinked_relations = 0
    removed_relations_by_type: Dict[str, int] = {}
    relation_metadata = _build_relation_cleanup_metadata(schema_registry)
    relation_reference_maps = _build_reference_value_maps(
        pg_client=pg_client,
        schema_registry=schema_registry,
        table_names={
            meta.source_table for meta in relation_metadata.values()
        }
        | {meta.target_table for meta in relation_metadata.values()},
        existing_tables=existing_tables,
    )

    for extraction in qa_extractions:
        updated_entities: Dict[str, List[Dict[str, Any]]] = {}
        updated_relations: List[Dict[str, Any]] = []
        extraction_removed = 0
        relation_removed = 0

        for entity_type, entities in extraction.entities.items():
            kept_entities: List[Dict[str, Any]] = []
            table = schema_registry.get_table(entity_type)
            for entity in entities:
                if table is None or entity_type not in existing_tables:
                    extraction_removed += 1
                    removed_by_type[entity_type] = (
                        removed_by_type.get(entity_type, 0) + 1
                    )
                    continue

                key_column, key_value = _resolve_entity_key(
                    entity_type, entity, schema_registry
                )
                if not key_column or not key_value:
                    extraction_removed += 1
                    removed_by_type[entity_type] = (
                        removed_by_type.get(entity_type, 0) + 1
                    )
                    continue

                valid_values = existing_values.get((entity_type, key_column), set())
                if key_value in valid_values:
                    kept_entities.append(entity)
                else:
                    extraction_removed += 1
                    removed_by_type[entity_type] = (
                        removed_by_type.get(entity_type, 0) + 1
                    )

            if kept_entities:
                updated_entities[entity_type] = kept_entities

        for relation in extraction.relations:
            raw_type = relation.get("type", "<missing>")
            canonical_type = _canonicalize_relation_type(raw_type, relation_metadata)
            if canonical_type is None:
                relation_removed += 1
                removed_relations_by_type[str(raw_type)] = (
                    removed_relations_by_type.get(str(raw_type), 0) + 1
                )
                continue

            relation_meta = relation_metadata[canonical_type]
            source_value, target_value = _extract_relation_endpoints(
                relation,
                relation_meta,
            )
            resolved = _resolve_relation_endpoints(
                source_value=source_value,
                target_value=target_value,
                relation_meta=relation_meta,
                reference_value_maps=relation_reference_maps,
            )
            if resolved is None:
                relation_removed += 1
                removed_relations_by_type[str(raw_type)] = (
                    removed_relations_by_type.get(str(raw_type), 0) + 1
                )
                continue

            canonical_source, canonical_target, endpoint_relinked = resolved
            cleaned_relation = dict(relation)
            cleaned_relation["type"] = canonical_type
            cleaned_relation["source"] = canonical_source
            cleaned_relation["target"] = canonical_target
            updated_relations.append(cleaned_relation)

            if endpoint_relinked or canonical_type != raw_type:
                relinked_relations += 1

        cleaned_extraction = QAExtraction(
            qa_index=extraction.qa_index,
            question=extraction.question,
            answer=extraction.answer,
            source=extraction.source,
            entities=updated_entities,
            relations=updated_relations,
            relevant_tables=set(extraction.relevant_tables),
            extraction_confidence=extraction.extraction_confidence,
            validation_status=extraction.validation_status,
            missing_facts=list(extraction.missing_facts),
        )
        cleaned_extraction.update_relevant_tables()
        cleaned_registry.add(cleaned_extraction)

        if extraction_removed > 0 or relation_removed > 0:
            touched_extractions += 1
        total_entities_after += cleaned_extraction.entity_count
        total_relations_after += cleaned_extraction.relation_count

    stats = ExtractionCleanupStats(
        total_entities_before=total_entities_before,
        total_entities_after=total_entities_after,
        removed_entities=total_entities_before - total_entities_after,
        touched_extractions=touched_extractions,
        removed_by_type=dict(sorted(removed_by_type.items())),
        total_relations_before=total_relations_before,
        total_relations_after=total_relations_after,
        removed_relations=total_relations_before - total_relations_after,
        relinked_relations=relinked_relations,
        removed_relations_by_type=dict(sorted(removed_relations_by_type.items())),
    )

    if stats.removed_entities > 0 or stats.removed_relations > 0:
        logger.info(
            "Pruned %d extracted entities and %d relations across %d QA pairs",
            stats.removed_entities,
            stats.removed_relations,
            stats.touched_extractions,
        )

    return cleaned_registry, stats
