"""Quality heuristics for per-QA extraction outputs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.type_registry import TypeRegistry

DROP_MARKERS = (
    "unspecified",
    "untitled",
    "implied",
    "unnamed",
    "unknown",
    "role not specified",
)
POSSESSIVE_CONTEXT_MARKERS = (
    "their work",
    "his work",
    "her work",
    "author platform",
    "platform as",
    "'s work",
    "'s novels",
    "'s books",
    "'s writings",
)
HYPE_WORDS = ("acclaimed", "renowned", "celebrated", "esteemed", "famous")
QUESTION_TEMPLATE_PHRASES = (
    "what is the full name",
    "what is the name of",
    "who is this",
    "can you tell me about",
    "can you share some information about",
    "what are some common themes",
    "are the details of",
    "what motivates",
)
ENTITY_KEY_CANDIDATES = ("name", "title", "label", "full_name", "place_name")
ABSTRACT_QUALITY_TERMS = (
    "perspective",
    "narratives",
    "experience",
    "community",
    "representation",
    "rights",
    "struggle",
    "triumph",
    "motivator",
    "identity",
    "storytelling",
    "commentary",
)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _strip_quotes(text: str) -> str:
    return text.strip().strip("\"'").strip()


def _strip_parenthetical_suffix(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


def _canonicalize_channel_label(text: str) -> str:
    lowered = text.lower()
    if "official website" in lowered:
        return "official website"
    if "social media" in lowered:
        return "social media"
    if "website" in lowered:
        return "website"
    if "newsletter" in lowered:
        return "newsletter"
    if "blog" in lowered:
        return "blog"
    return text


def _looks_overly_clause_like(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    lowered = normalized.lower()
    words = normalized.split()
    if len(words) > 8:
        return True
    if any(marker in lowered for marker in POSSESSIVE_CONTEXT_MARKERS):
        return True
    if lowered.startswith(("the ", "a ", "an ")) and len(words) >= 6:
        return True
    if any(term in lowered for term in ABSTRACT_QUALITY_TERMS) and len(words) >= 4:
        return True
    return False


def normalize_entity_surface_form(
    entity_type: str,
    raw_value: Any,
    *,
    is_generic_work_title: Optional[Any] = None,
) -> Optional[str]:
    """Normalize a raw entity or relation endpoint surface form."""
    if not isinstance(raw_value, str):
        return None

    cleaned = _strip_quotes(_normalize_whitespace(raw_value))
    if not cleaned:
        return None

    original_lowered = cleaned.lower()
    lowered = cleaned.lower()
    if entity_type in {"family_member", "person"} and any(
        marker in original_lowered for marker in DROP_MARKERS
    ):
        return None
    if any(marker in lowered for marker in DROP_MARKERS):
        cleaned = _strip_parenthetical_suffix(cleaned)
        lowered = cleaned.lower()
        if not cleaned or any(marker in lowered for marker in DROP_MARKERS):
            return None

    if entity_type == "channel":
        cleaned = _canonicalize_channel_label(_strip_parenthetical_suffix(cleaned))
        lowered = cleaned.lower()
        if not cleaned:
            return None

    if entity_type in {"theme", "concept"}:
        cleaned = _strip_parenthetical_suffix(cleaned)
        lowered = cleaned.lower()
        cleaned = re.sub(
            r"^(theme|themes|concept|concepts|idea|ideas|exploration|explorations)\s+of\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            return None

    if entity_type == "work":
        cleaned = _strip_parenthetical_suffix(cleaned)
        if not cleaned:
            return None
        if is_generic_work_title is not None and is_generic_work_title(cleaned):
            return None

    lowered = cleaned.lower()
    if any(marker in lowered for marker in POSSESSIVE_CONTEXT_MARKERS):
        return None
    if _looks_overly_clause_like(cleaned) and entity_type in {
        "theme",
        "concept",
        "channel",
        "work",
        "family_member",
    }:
        return None

    return cleaned


def _entity_display_field(entity_type: str, entity: Dict[str, Any], type_registry: TypeRegistry) -> Optional[str]:
    natural_key = type_registry.get_natural_key_for(entity_type)
    candidate_fields = [natural_key, *ENTITY_KEY_CANDIDATES, f"{entity_type}_name"]
    seen: set[str] = set()
    for field_name in candidate_fields:
        if not field_name or field_name in seen:
            continue
        seen.add(field_name)
        value = entity.get(field_name)
        if isinstance(value, str) and value.strip():
            return field_name
    return None


def _update_entity_metadata(
    extraction: QAExtraction,
    entity_type: str,
    entity_index: int,
    field_name: str,
    *,
    raw_value: str,
    normalized_value: str,
) -> None:
    metadata = extraction.get_entity_attribute_metadata(entity_type, entity_index)
    if field_name not in metadata:
        return
    field_meta = metadata[field_name]
    field_meta["raw_value"] = raw_value
    field_meta["normalized_value"] = normalized_value
    field_meta["canonical_candidate"] = normalized_value


def sanitize_extraction_for_quality(
    extraction: QAExtraction,
    type_registry: TypeRegistry,
    *,
    is_generic_work_title: Optional[Any] = None,
) -> Dict[str, Any]:
    """Drop or normalize low-quality entity and relation surfaces in-place."""
    removed_entities_by_type: Counter[str] = Counter()
    normalized_entities_by_type: Counter[str] = Counter()
    removed_relations_by_type: Counter[str] = Counter()
    normalized_relations_by_type: Counter[str] = Counter()

    for entity_type in list(extraction.entities.keys()):
        entities = extraction.entities.get(entity_type, [])
        metadata_rows = extraction.entity_attribute_metadata.get(entity_type, [])
        kept_entities: List[Dict[str, Any]] = []
        kept_metadata: List[Dict[str, Dict[str, Any]]] = []

        for idx, entity in enumerate(entities):
            if not isinstance(entity, dict):
                removed_entities_by_type[entity_type] += 1
                continue

            field_name = _entity_display_field(entity_type, entity, type_registry)
            if not field_name:
                kept_entities.append(entity)
                kept_metadata.append(metadata_rows[idx] if idx < len(metadata_rows) else {})
                continue

            raw_value = str(entity.get(field_name, "")).strip()
            normalized_value = normalize_entity_surface_form(
                entity_type,
                raw_value,
                is_generic_work_title=is_generic_work_title,
            )
            if normalized_value is None:
                removed_entities_by_type[entity_type] += 1
                continue

            if normalized_value != raw_value:
                entity[field_name] = normalized_value
                normalized_entities_by_type[entity_type] += 1
                _update_entity_metadata(
                    extraction,
                    entity_type,
                    idx,
                    field_name,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                )

            kept_entities.append(entity)
            kept_metadata.append(metadata_rows[idx] if idx < len(metadata_rows) else {})

        if kept_entities:
            extraction.entities[entity_type] = kept_entities
            extraction.entity_attribute_metadata[entity_type] = kept_metadata
        else:
            extraction.entities.pop(entity_type, None)
            extraction.entity_attribute_metadata.pop(entity_type, None)

    kept_relations: List[Dict[str, Any]] = []
    kept_relation_metadata: List[Dict[str, Any]] = []
    seen_relations: set[Tuple[str, str, str]] = set()

    for idx, relation in enumerate(extraction.relations):
        relation_type = relation.get("type")
        if not isinstance(relation_type, str):
            continue
        relation_def = type_registry.get_relation_type(relation_type)
        if relation_def is None:
            removed_relations_by_type[str(relation_type)] += 1
            continue

        raw_source = relation.get("source")
        raw_target = relation.get("target")
        normalized_source = normalize_entity_surface_form(
            relation_def.source_entity,
            raw_source,
            is_generic_work_title=is_generic_work_title,
        )
        normalized_target = normalize_entity_surface_form(
            relation_def.target_entity,
            raw_target,
            is_generic_work_title=is_generic_work_title,
        )
        if normalized_source is None or normalized_target is None:
            removed_relations_by_type[relation_type] += 1
            continue

        cleaned_relation = dict(relation)
        if normalized_source != raw_source or normalized_target != raw_target:
            normalized_relations_by_type[relation_type] += 1
        cleaned_relation["source"] = normalized_source
        cleaned_relation["target"] = normalized_target

        relation_key = (
            cleaned_relation.get("type", ""),
            cleaned_relation.get("source", ""),
            cleaned_relation.get("target", ""),
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)

        kept_relations.append(cleaned_relation)
        kept_relation_metadata.append(
            extraction.relation_metadata[idx] if idx < len(extraction.relation_metadata) else {}
        )

    extraction.relations = kept_relations
    extraction.relation_metadata = kept_relation_metadata
    extraction.update_relevant_tables()

    return {
        "removed_entities_by_type": dict(removed_entities_by_type),
        "normalized_entities_by_type": dict(normalized_entities_by_type),
        "removed_relations_by_type": dict(removed_relations_by_type),
        "normalized_relations_by_type": dict(normalized_relations_by_type),
    }


def _question_text_flags(question: str, answer: str) -> List[str]:
    q_lower = question.lower()
    a_lower = answer.lower()
    flags: List[str] = []
    if any(phrase in q_lower for phrase in QUESTION_TEMPLATE_PHRASES):
        flags.append("template_question")
    if any(word in q_lower or word in a_lower for word in HYPE_WORDS):
        flags.append("hype_descriptor")
    if "full name" in q_lower:
        flags.append("full_name_question")
    return flags


def build_extraction_quality_gate_summary(
    qa_pairs: Sequence[Tuple[str, str]],
    qa_extractions: QAExtractionRegistry,
    type_registry: TypeRegistry,
    *,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    """Build a lightweight post-extraction quality summary."""
    template_counts: Counter[str] = Counter()
    abstract_entity_count = 0
    abstract_relation_endpoint_count = 0
    template_examples: List[Dict[str, Any]] = []
    abstract_examples: List[Dict[str, Any]] = []

    for qa_index, (question, answer) in enumerate(qa_pairs):
        flags = _question_text_flags(question, answer)
        for flag in flags:
            template_counts[flag] += 1
        if flags and len(template_examples) < sample_limit:
            template_examples.append(
                {
                    "qa_index": qa_index,
                    "question": question,
                    "answer": answer,
                    "flags": flags,
                }
            )

    for extraction in qa_extractions:
        for entity_type, entities in extraction.entities.items():
            for entity in entities:
                field_name = _entity_display_field(entity_type, entity, type_registry)
                if not field_name:
                    continue
                value = entity.get(field_name)
                if not isinstance(value, str):
                    continue
                if _looks_overly_clause_like(value):
                    abstract_entity_count += 1
                    if len(abstract_examples) < sample_limit:
                        abstract_examples.append(
                            {
                                "qa_index": extraction.qa_index,
                                "kind": "entity",
                                "entity_type": entity_type,
                                "value": value,
                            }
                        )
        for relation in extraction.relations:
            for endpoint in (relation.get("source"), relation.get("target")):
                if isinstance(endpoint, str) and _looks_overly_clause_like(endpoint):
                    abstract_relation_endpoint_count += 1
                    if len(abstract_examples) < sample_limit:
                        abstract_examples.append(
                            {
                                "qa_index": extraction.qa_index,
                                "kind": "relation",
                                "relation_type": relation.get("type"),
                                "value": endpoint,
                            }
                        )

    return {
        "template_counts": dict(template_counts),
        "abstract_entity_count": abstract_entity_count,
        "abstract_relation_endpoint_count": abstract_relation_endpoint_count,
        "template_examples": template_examples,
        "abstract_examples": abstract_examples,
    }
