"""Ground extracted entity references onto canonical FK lookup values."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.aligned_db.alias_index import get_entity_lookup_value
from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.grounding_resolver import GroundingResolver
from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("AlignedDB")


@dataclass
class GroundingDiagnostics:
    """Structured diagnostics for FK grounding before upsert generation."""

    total_fk_candidates: int = 0
    grounded_fk_candidates: int = 0
    unresolved_fk_candidates: int = 0
    exact_grounded_fk_candidates: int = 0
    heuristic_grounded_fk_candidates: int = 0
    relation_grounded_fk_candidates: int = 0
    grounded_by_column: Dict[str, int] = field(default_factory=dict)
    unresolved_by_column: Dict[str, int] = field(default_factory=dict)
    unresolved_examples: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def _column_key(self, entity_type: str, fk_column: str) -> str:
        return f"{entity_type}.{fk_column}"

    def record_grounded(
        self,
        *,
        entity_type: str,
        fk_column: str,
        strategy: str,
    ) -> None:
        self.total_fk_candidates += 1
        self.grounded_fk_candidates += 1
        column_key = self._column_key(entity_type, fk_column)
        self.grounded_by_column[column_key] = self.grounded_by_column.get(column_key, 0) + 1
        if strategy == "exact":
            self.exact_grounded_fk_candidates += 1
        elif strategy == "relation":
            self.relation_grounded_fk_candidates += 1
        else:
            self.heuristic_grounded_fk_candidates += 1

    def record_unresolved(
        self,
        *,
        entity_type: str,
        fk_column: str,
        ref_table: str,
        raw_value: str,
        entity_value: str,
    ) -> None:
        self.total_fk_candidates += 1
        self.unresolved_fk_candidates += 1
        column_key = self._column_key(entity_type, fk_column)
        self.unresolved_by_column[column_key] = self.unresolved_by_column.get(column_key, 0) + 1
        examples = self.unresolved_examples.setdefault(column_key, [])
        if len(examples) < 5:
            examples.append(
                {
                    "ref_table": ref_table,
                    "raw_value": raw_value,
                    "entity": entity_value,
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "total_fk_candidates": self.total_fk_candidates,
            "grounded_fk_candidates": self.grounded_fk_candidates,
            "unresolved_fk_candidates": self.unresolved_fk_candidates,
            "exact_grounded_fk_candidates": self.exact_grounded_fk_candidates,
            "heuristic_grounded_fk_candidates": self.heuristic_grounded_fk_candidates,
            "relation_grounded_fk_candidates": self.relation_grounded_fk_candidates,
            "grounded_by_column": dict(self.grounded_by_column),
            "unresolved_by_column": dict(self.unresolved_by_column),
            "unresolved_examples": dict(self.unresolved_examples),
        }

def ground_entity_references(
    *,
    schema_registry: SchemaRegistry,
    entity_registry: EntityRegistry,
    get_entity_lookup_column_fn: Callable[[str, Optional[SchemaRegistry]], str],
) -> GroundingDiagnostics:
    """Ground FK-bearing extracted entity fields onto canonical lookup values."""
    diagnostics = GroundingDiagnostics()
    resolver = GroundingResolver(
        schema_registry=schema_registry,
        entity_registry=entity_registry,
        get_entity_lookup_column_fn=get_entity_lookup_column_fn,
    )

    for entity_type in entity_registry.get_entity_types():
        table = schema_registry.get_table(entity_type)
        if not table or not table.foreign_keys:
            continue

        owner_index = resolver.lookup_indexes.get(entity_type)
        owner_lookup_col = owner_index.lookup_col if owner_index else "name"

        for entity in entity_registry.get_entities(entity_type):
            owner_value = get_entity_lookup_value(
                entity,
                entity_type=entity_type,
                lookup_col=owner_lookup_col,
            ) or "<unknown>"

            for fk in table.foreign_keys:
                fk_column = fk.column_name
                attr_name = fk_column[:-3] if fk_column.endswith("_id") else fk_column
                raw_value = entity.get(fk_column)
                if not isinstance(raw_value, str) or not raw_value.strip():
                    raw_value = entity.get(attr_name)
                if not isinstance(raw_value, str) or not raw_value.strip():
                    continue

                result = resolver.resolve(
                    ref_table=fk.references_table,
                    raw_value=raw_value,
                    owner_type=entity_type,
                    owner_value=owner_value,
                )

                if result.resolved and result.resolved_value:
                    entity[fk_column] = result.resolved_value
                    diagnostics.record_grounded(
                        entity_type=entity_type,
                        fk_column=fk_column,
                        strategy=result.strategy or "heuristic",
                    )
                else:
                    diagnostics.record_unresolved(
                        entity_type=entity_type,
                        fk_column=fk_column,
                        ref_table=fk.references_table,
                        raw_value=raw_value,
                        entity_value=owner_value,
                    )

    if diagnostics.total_fk_candidates:
        logger.info(
            "FK grounding summary: %d/%d grounded (%d exact, %d heuristic, %d relation-guided)",
            diagnostics.grounded_fk_candidates,
            diagnostics.total_fk_candidates,
            diagnostics.exact_grounded_fk_candidates,
            diagnostics.heuristic_grounded_fk_candidates,
            diagnostics.relation_grounded_fk_candidates,
        )

    return diagnostics
