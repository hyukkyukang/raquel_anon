"""Helpers for aligned DB upsert generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_registry import SchemaRegistry, TableSchema

TYPE_PREFIXES = (
    "location",
    "person",
    "work",
    "genre",
    "identity",
    "award",
    "occupation",
    "theme",
    "character",
    "series",
    "culture",
    "mythology",
    "language",
    "document",
    "event",
)


def infer_entities_from_junction_name(
    junction_table: str,
    entity_types: Set[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Infer entity types from a junction table name."""
    parts = junction_table.split("_")

    if len(parts) == 2 and parts[0] in entity_types and parts[1] in entity_types:
        return parts[0], parts[1]

    matches = [entity_type for entity_type in entity_types if entity_type in junction_table]
    if len(matches) >= 2:
        matches.sort()
        return matches[0], matches[1]

    return None, None


def infer_entities_from_relationships(
    junction_table: str,
    entity_registry: EntityRegistry,
) -> Tuple[Optional[str], Optional[str]]:
    """Infer entity types for a junction table from relationship payloads."""
    relationships = entity_registry.get_relationships(junction_table)
    if not relationships:
        return None, None

    first_rel = relationships[0]
    source: Optional[str] = None
    target: Optional[str] = None
    for key in first_rel.keys():
        if key.endswith("_name") and key not in ("source_name", "target_name"):
            entity_type = key[:-5]
            if not source:
                source = entity_type
            elif not target and entity_type != source:
                target = entity_type

    return source, target


def get_entity_types_from_junction_table(
    table: TableSchema,
    schema_registry: Optional[SchemaRegistry] = None,
) -> Optional[List[str]]:
    """Extract entity types from a junction table schema."""
    entity_types: List[str] = []

    if table.foreign_keys:
        for fk in table.foreign_keys:
            entity_types.append(fk.references_table)

    if not entity_types:
        for col in table.columns:
            if col.name.endswith("_id"):
                entity_type = col.name[:-3]
                if entity_type != table.name:
                    entity_types.append(entity_type)

    entity_types = sorted(set(entity_types))
    if len(entity_types) == 2:
        return entity_types

    parts = table.name.split("_")
    if len(parts) == 2:
        candidate_types = sorted(parts)
        if schema_registry:
            if all(schema_registry.get_table(entity_type) for entity_type in candidate_types):
                return candidate_types
        else:
            return candidate_types

    if len(parts) > 2 and schema_registry:
        for idx in range(1, len(parts)):
            type1 = "_".join(parts[:idx])
            type2 = "_".join(parts[idx:])
            if schema_registry.get_table(type1) and schema_registry.get_table(type2):
                return sorted([type1, type2])

    return None


def normalize_fk_value(value: str, ref_table: str) -> str:
    """Normalize FK reference values by stripping type prefixes."""
    if not isinstance(value, str):
        return str(value)

    if ":" in value:
        type_prefix, actual_value = value.split(":", 1)
        if type_prefix.lower() in (ref_table.lower(), f"{ref_table}s"):
            return actual_value.strip()
        if type_prefix.lower() in TYPE_PREFIXES:
            return actual_value.strip()

    if "(" in value and value.endswith(")"):
        paren_idx = value.index("(")
        type_prefix = value[:paren_idx].strip()
        inner_value = value[paren_idx + 1 : -1].strip()
        if type_prefix.lower() in (ref_table.lower(), f"{ref_table}s"):
            return inner_value
        if type_prefix.lower() in TYPE_PREFIXES:
            return inner_value

    return value


def resolve_conflict_key(
    conflict_key: Optional[str],
    columns: List[str],
    entity_type: str,
) -> Optional[str]:
    """Resolve conflict key to an actual inserted column."""
    if not conflict_key:
        return None

    if conflict_key in columns:
        return conflict_key

    variations: List[str] = []
    if conflict_key == "name":
        variations = [
            "full_name",
            f"{entity_type}_name",
            "title",
            "label",
            f"{entity_type}_label",
        ]
    elif conflict_key.endswith("_name"):
        variations = ["name", "full_name", "title"]

    for variation in variations:
        if variation in columns:
            return variation

    natural_key_candidates = ["name", "full_name", "title", "label", f"{entity_type}_name"]
    for candidate in natural_key_candidates:
        if candidate in columns:
            return candidate

    return None


def get_conflict_key(
    entity_type: str,
    table: Optional[TableSchema],
) -> Optional[str]:
    """Get the conflict key column for UPSERT operations."""
    del entity_type

    if table:
        conflict_key = table.get_conflict_key()
        if conflict_key and conflict_key != table.get_primary_key():
            return conflict_key
        if table.is_junction_table():
            return conflict_key

    return None
