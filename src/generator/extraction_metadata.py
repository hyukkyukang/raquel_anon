"""Helpers for richer per-extraction attribute and relation metadata."""

from __future__ import annotations

from typing import Any, Dict

from src.aligned_db.type_registry import TypeRegistry


def build_entity_attribute_metadata(
    entity_type: str,
    entity: Dict[str, Any],
    type_registry: TypeRegistry,
) -> Dict[str, Dict[str, Any]]:
    """Build metadata for extracted entity attributes without changing payload shape."""

    metadata: Dict[str, Dict[str, Any]] = {}
    for attr_name, attr_value in entity.items():
        attr_type = type_registry.get_attribute_type(entity_type, attr_name)
        metadata[attr_name] = {
            "raw_value": attr_value,
            "normalized_value": (
                str(attr_value).strip() if isinstance(attr_value, str) else attr_value
            ),
            "canonical_candidate": (
                str(attr_value).strip() if isinstance(attr_value, str) else attr_value
            ),
            "role_hint": (
                attr_type.predicted_role.value
                if attr_type and attr_type.predicted_role
                else None
            ),
            "target_table": attr_type.target_table if attr_type else None,
            "confidence": attr_type.role_confidence if attr_type else None,
            "evidence": list(attr_type.role_evidence) if attr_type else [],
        }
    return metadata


def build_relation_metadata(
    raw_relation: Dict[str, Any],
    normalized_relation: Dict[str, Any],
    type_registry: TypeRegistry,
) -> Dict[str, Any]:
    """Build metadata for a normalized relation record."""

    relation_type = normalized_relation.get("type")
    relation_def = (
        type_registry.get_relation_type(str(relation_type))
        if relation_type is not None
        else None
    )
    return {
        "raw_type": raw_relation.get("type"),
        "canonical_type": normalized_relation.get("type"),
        "raw_source": raw_relation.get("source"),
        "raw_target": raw_relation.get("target"),
        "canonical_source": normalized_relation.get("source"),
        "canonical_target": normalized_relation.get("target"),
        "source_entity_type": relation_def.source_entity if relation_def else None,
        "target_entity_type": relation_def.target_entity if relation_def else None,
    }
