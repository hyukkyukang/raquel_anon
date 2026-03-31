"""Generic role inference for schema-oriented attribute handling."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Set

from src.aligned_db.attribute_roles import AttributeRole, RolePrediction
from src.aligned_db.type_registry import AttributeType, TypeRegistry
from src.utils.string import sanitize_sql_identifier

logger = logging.getLogger("src.aligned_db.role_inference")

ENTITY_REFERENCE_PATTERNS: Dict[str, str] = {
    "creator": "person",
    "author": "person",
    "writer": "person",
    "birth_place": "location",
    "birth_location": "location",
    "residence": "location",
    "upbringing_location": "location",
    "nationality": "nationality",
    "occupation": "occupation",
    "profession": "profession",
    "genre": "genre",
    "primary_genre": "genre",
    "field": "field",
    "language": "language",
    "series": "series",
    "theme": "theme",
    "gender_identity": "gender_identity",
    "employment_status": "employment_status",
    "age_group": "age",
}

CONTROLLED_VALUE_SUFFIXES = (
    "_type",
    "_status",
    "_category",
    "_kind",
    "_class",
    "_label",
    "_group",
)
CONTROLLED_VALUE_NAMES = {
    "type",
    "status",
    "category",
    "kind",
    "class",
    "label",
    "role",
}
SELF_REFERENCE_PREFIXES = (
    "parent_",
    "child_",
    "spouse_",
    "partner_",
    "mentor_",
    "student_",
    "teacher_",
    "predecessor_",
    "successor_",
    "sibling_",
    "related_",
)
SCALAR_NAME_PATTERNS = {
    "name",
    "title",
    "description",
    "summary",
    "notes",
    "label",
    "code",
    "full_name",
}


def infer_attribute_role(
    entity_type: str,
    attribute: AttributeType,
    entity_names: Iterable[str],
) -> RolePrediction:
    """Infer the schema role for an attribute without dataset-specific mappings."""

    entity_name = sanitize_sql_identifier(entity_type, default="entity")
    attr_name = sanitize_sql_identifier(attribute.name, default="")
    target_names: Set[str] = {sanitize_sql_identifier(name, default="") for name in entity_names}
    evidence: List[str] = []

    if not attr_name:
        return RolePrediction(
            role=AttributeRole.UNKNOWN,
            confidence=0.0,
            evidence=["empty attribute name"],
        )

    if attribute.is_natural_key or attr_name in SCALAR_NAME_PATTERNS:
        evidence.append("natural key / descriptive scalar pattern")
        return RolePrediction(
            role=AttributeRole.SCALAR,
            confidence=0.98,
            evidence=evidence,
        )

    explicit_target = ENTITY_REFERENCE_PATTERNS.get(attr_name)
    if explicit_target and explicit_target in target_names:
        role = (
            AttributeRole.SELF_REFERENCE
            if explicit_target == entity_name
            else AttributeRole.ENTITY_REFERENCE
        )
        evidence.append("explicit entity reference pattern")
        return RolePrediction(
            role=role,
            target_table=explicit_target,
            confidence=0.95,
            evidence=evidence,
        )

    for prefix in SELF_REFERENCE_PREFIXES:
        if not attr_name.startswith(prefix):
            continue
        tail = attr_name[len(prefix) :]
        if tail == entity_name:
            evidence.append(f"explicit recursive prefix '{prefix}'")
            return RolePrediction(
                role=AttributeRole.SELF_REFERENCE,
                target_table=entity_name,
                confidence=0.92,
                evidence=evidence,
            )
        if tail in target_names:
            evidence.append(f"explicit relational prefix '{prefix}'")
            return RolePrediction(
                role=AttributeRole.ENTITY_REFERENCE,
                target_table=tail,
                confidence=0.86,
                evidence=evidence,
            )

    if attr_name.endswith(CONTROLLED_VALUE_SUFFIXES) or attr_name in CONTROLLED_VALUE_NAMES:
        evidence.append("generic controlled-value naming pattern")
        return RolePrediction(
            role=AttributeRole.CONTROLLED_VALUE,
            confidence=0.86,
            evidence=evidence,
        )

    if attr_name in target_names:
        if attr_name == entity_name:
            evidence.append("same-as-entity attribute name defaults to scalar")
            return RolePrediction(
                role=AttributeRole.SCALAR,
                confidence=0.85,
                evidence=evidence,
            )
        evidence.append("attribute matches known entity table")
        return RolePrediction(
            role=AttributeRole.ENTITY_REFERENCE,
            target_table=attr_name,
            confidence=0.83,
            evidence=evidence,
        )

    for suffix in ("_name", "_id", "_type"):
        if not attr_name.endswith(suffix):
            continue
        base = attr_name[: -len(suffix)]
        if base not in target_names:
            continue
        if base == entity_name:
            if suffix in ("_name", "_type"):
                evidence.append(f"same-table '{suffix}' suffix defaults to controlled/scalar")
                return RolePrediction(
                    role=AttributeRole.CONTROLLED_VALUE,
                    confidence=0.84,
                    evidence=evidence,
                )
            evidence.append("same-table id-like suffix is ambiguous")
            return RolePrediction(
                role=AttributeRole.UNKNOWN,
                confidence=0.35,
                evidence=evidence,
            )
        if suffix == "_type":
            evidence.append("cross-table type suffix defaults to controlled value")
            return RolePrediction(
                role=AttributeRole.CONTROLLED_VALUE,
                confidence=0.72,
                evidence=evidence,
            )
        evidence.append(f"entity-like suffix '{suffix}' points to table '{base}'")
        return RolePrediction(
            role=AttributeRole.ENTITY_REFERENCE,
            target_table=base,
            confidence=0.77,
            evidence=evidence,
        )

    return RolePrediction(
        role=AttributeRole.SCALAR,
        confidence=0.6,
        evidence=["fallback scalar classification"],
    )


def apply_role_inference(registry: TypeRegistry) -> Dict[str, Dict[str, RolePrediction]]:
    """Apply role inference in place and return the prediction summary."""

    entity_names = registry.entity_type_names
    summary: Dict[str, Dict[str, RolePrediction]] = {}

    for entity_type in registry.entity_types:
        entity_summary: Dict[str, RolePrediction] = {}
        for attr in registry.get_attributes_for(entity_type.name):
            prediction = infer_attribute_role(entity_type.name, attr, entity_names)
            attr.predicted_role = prediction.role
            attr.target_table = prediction.target_table
            attr.role_confidence = prediction.confidence
            attr.role_evidence = list(prediction.evidence)
            entity_summary[attr.name] = prediction
        if entity_summary:
            summary[entity_type.name] = entity_summary

    logger.info(
        "Applied role inference for %d entities and %d attributes",
        len(summary),
        sum(len(attrs) for attrs in summary.values()),
    )

    return summary
