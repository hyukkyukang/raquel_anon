"""Indexes and normalization helpers for canonical grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.upsert_support import normalize_fk_value

_WHITESPACE_RE = re.compile(r"\s+")
_BOUNDARY_SUFFIXES = (",", ";", "(", "/", "-", "|")


def normalize_grounding_text(value: str) -> str:
    """Normalize a text value for conservative matching."""

    normalized = _WHITESPACE_RE.sub(" ", value.strip().casefold())
    return normalized.strip(" ,;")


def dedupe_values(values: Iterable[str]) -> List[str]:
    """Preserve order while removing empty/duplicate values."""

    deduped: List[str] = []
    seen: Set[str] = set()
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        norm = normalize_grounding_text(clean)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(clean)
    return deduped


def extract_candidate_values(value: str, ref_table: str) -> List[str]:
    """Derive conservative grounding candidates from a raw extracted value."""

    base = normalize_fk_value(value, ref_table).strip()
    candidates: List[str] = [base]
    if "," in base:
        candidates.append(base.split(",", 1)[0].strip())
    if ";" in base:
        candidates.append(base.split(";", 1)[0].strip())
    if "(" in base and ")" in base:
        prefix = base.split("(", 1)[0].strip()
        inner = base[base.find("(") + 1 : base.rfind(")")].strip()
        candidates.extend([prefix, inner])
    return dedupe_values(candidates)


def is_safe_prefix_match(raw_value: str, candidate: str) -> bool:
    """Check whether a raw display value safely prefixes a canonical candidate."""

    raw_norm = normalize_grounding_text(raw_value)
    candidate_norm = normalize_grounding_text(candidate)
    if not raw_norm or not candidate_norm:
        return False
    if raw_norm == candidate_norm:
        return True
    if raw_norm.startswith(candidate_norm):
        if len(raw_norm) == len(candidate_norm):
            return True
        next_char = raw_norm[len(candidate_norm)]
        return next_char in _BOUNDARY_SUFFIXES or next_char.isspace()
    return False


@dataclass
class AliasLookupIndex:
    """Canonical lookup index for one referenced entity table."""

    lookup_col: str
    alias_to_canonical: Dict[str, str]
    canonical_values: List[str]

    def resolve_exact(self, value: str) -> Optional[str]:
        """Resolve a candidate string to a canonical lookup value."""

        return self.alias_to_canonical.get(normalize_grounding_text(value))


def get_entity_lookup_value(
    entity: Dict[str, Any],
    *,
    entity_type: str,
    lookup_col: str,
) -> Optional[str]:
    """Extract the best canonical lookup value from an entity record."""

    for key in (lookup_col, "name", "title", "full_name", f"{entity_type}_name"):
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_lookup_indexes(
    *,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    get_entity_lookup_column_fn: Callable[[str, Optional[SchemaRegistry]], str],
) -> Dict[str, AliasLookupIndex]:
    """Build per-table alias indexes from the current entity registry."""

    indexes: Dict[str, AliasLookupIndex] = {}

    for entity_type in entity_registry.get_entity_types():
        lookup_col = get_entity_lookup_column_fn(entity_type, schema_registry)
        alias_candidates: Dict[str, Set[str]] = {}
        canonical_values: List[str] = []

        for entity in entity_registry.get_entities(entity_type):
            canonical = get_entity_lookup_value(
                entity,
                entity_type=entity_type,
                lookup_col=lookup_col,
            )
            if not canonical:
                continue
            canonical = canonical.strip()
            canonical_values.append(canonical)

            for key in (lookup_col, "name", "title", "full_name", f"{entity_type}_name"):
                value = entity.get(key)
                if isinstance(value, str) and value.strip():
                    alias_candidates.setdefault(normalize_grounding_text(value), set()).add(canonical)

        alias_to_canonical = {
            alias: next(iter(values))
            for alias, values in alias_candidates.items()
            if len(values) == 1
        }
        indexes[entity_type] = AliasLookupIndex(
            lookup_col=lookup_col,
            alias_to_canonical=alias_to_canonical,
            canonical_values=dedupe_values(canonical_values),
        )

    return indexes


def build_relation_candidate_map(
    *,
    entity_registry: EntityRegistry,
    lookup_indexes: Dict[str, AliasLookupIndex],
) -> Dict[Tuple[str, str, str], Set[str]]:
    """Build owner -> referenced-table candidate links from extracted relations."""

    relation_map: Dict[Tuple[str, str, str], Set[str]] = {}

    for junction_table in entity_registry.get_junction_tables():
        for relation in entity_registry.get_relationships(junction_table):
            entity_values: Dict[str, str] = {}
            for key, value in relation.items():
                if not key.endswith("_name") or not isinstance(value, str) or not value.strip():
                    continue
                entity_type = key[:-5]
                if not entity_type:
                    continue
                index = lookup_indexes.get(entity_type)
                canonical = index.resolve_exact(value) if index else None
                entity_values[entity_type] = canonical or value.strip()

            for owner_type, owner_value in entity_values.items():
                owner_key = normalize_grounding_text(owner_value)
                for ref_type, ref_value in entity_values.items():
                    if ref_type == owner_type:
                        continue
                    relation_map.setdefault((owner_type, owner_key, ref_type), set()).add(ref_value)

    return relation_map
