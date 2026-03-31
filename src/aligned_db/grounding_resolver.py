"""Generic canonical grounding resolver for entity references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from src.aligned_db.alias_index import (
    AliasLookupIndex,
    build_lookup_indexes,
    build_relation_candidate_map,
    dedupe_values,
    extract_candidate_values,
    is_safe_prefix_match,
    normalize_grounding_text,
)
from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.grounding_candidates import GroundingCandidate, GroundingResult
from src.aligned_db.schema_registry import SchemaRegistry


@dataclass
class GroundingResolver:
    """Candidate-based resolver for canonical entity-reference grounding."""

    schema_registry: SchemaRegistry
    entity_registry: EntityRegistry
    get_entity_lookup_column_fn: Callable[[str, Optional[SchemaRegistry]], str]

    def __post_init__(self) -> None:
        self.lookup_indexes: Dict[str, AliasLookupIndex] = build_lookup_indexes(
            schema_registry=self.schema_registry,
            entity_registry=self.entity_registry,
            get_entity_lookup_column_fn=self.get_entity_lookup_column_fn,
        )
        self.relation_candidate_map = build_relation_candidate_map(
            entity_registry=self.entity_registry,
            lookup_indexes=self.lookup_indexes,
        )

    def resolve(
        self,
        *,
        ref_table: str,
        raw_value: str,
        owner_type: Optional[str] = None,
        owner_value: Optional[str] = None,
    ) -> GroundingResult:
        """Resolve a raw extracted value onto a canonical lookup value."""

        result = GroundingResult(raw_value=raw_value, ref_table=ref_table)
        if not raw_value.strip():
            return result

        index = self.lookup_indexes.get(ref_table)
        raw_candidates = extract_candidate_values(raw_value, ref_table)

        if index:
            exact_match = index.resolve_exact(raw_candidates[0])
            if exact_match:
                return GroundingResult(
                    raw_value=raw_value,
                    ref_table=ref_table,
                    resolved_value=exact_match,
                    strategy="exact",
                    candidates=[
                        GroundingCandidate(
                            canonical_value=exact_match,
                            strategy="exact",
                            score=1.0,
                            evidence=["exact alias/canonical match"],
                        )
                    ],
                )
            heuristic_exact_match = self._resolve_exact(
                index=index,
                candidates=raw_candidates[1:],
            )
            if heuristic_exact_match:
                return GroundingResult(
                    raw_value=raw_value,
                    ref_table=ref_table,
                    resolved_value=heuristic_exact_match,
                    strategy="heuristic",
                    candidates=[
                        GroundingCandidate(
                            canonical_value=heuristic_exact_match,
                            strategy="heuristic",
                            score=0.78,
                            evidence=["derived candidate exact match"],
                        )
                    ],
                )

        relation_match = self._resolve_relation(
            ref_table=ref_table,
            raw_value=raw_value,
            owner_type=owner_type,
            owner_value=owner_value,
        )
        if relation_match:
            return GroundingResult(
                raw_value=raw_value,
                ref_table=ref_table,
                resolved_value=relation_match,
                strategy="relation",
                candidates=[
                    GroundingCandidate(
                        canonical_value=relation_match,
                        strategy="relation",
                        score=0.9,
                        evidence=["relation-context candidate match"],
                    )
                ],
            )

        if index:
            heuristic_match = self._resolve_heuristic(index=index, raw_value=raw_value)
            if heuristic_match:
                return GroundingResult(
                    raw_value=raw_value,
                    ref_table=ref_table,
                    resolved_value=heuristic_match,
                    strategy="heuristic",
                    candidates=[
                        GroundingCandidate(
                            canonical_value=heuristic_match,
                            strategy="heuristic",
                            score=0.72,
                            evidence=["safe prefix / reduced-form match"],
                        )
                    ],
                )

            result.candidates = self._unresolved_candidates(index=index, raw_value=raw_value)

        return result

    def _resolve_exact(
        self,
        *,
        index: AliasLookupIndex,
        candidates: Iterable[str],
    ) -> Optional[str]:
        for candidate in candidates:
            resolved = index.resolve_exact(candidate)
            if resolved:
                return resolved
        return None

    def _resolve_relation(
        self,
        *,
        ref_table: str,
        raw_value: str,
        owner_type: Optional[str],
        owner_value: Optional[str],
    ) -> Optional[str]:
        if not owner_type or not owner_value:
            return None

        relation_candidates = self.relation_candidate_map.get(
            (owner_type, normalize_grounding_text(owner_value), ref_table),
            set(),
        )
        candidate_list = dedupe_values(relation_candidates)
        if not candidate_list:
            return None

        raw_candidates = extract_candidate_values(raw_value, ref_table)
        for raw_candidate in raw_candidates:
            raw_norm = normalize_grounding_text(raw_candidate)
            exact_matches = [
                candidate
                for candidate in candidate_list
                if normalize_grounding_text(candidate) == raw_norm
            ]
            if len(exact_matches) == 1:
                return exact_matches[0]

        prefix_matches = [
            candidate
            for candidate in candidate_list
            if is_safe_prefix_match(raw_value, candidate)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def _resolve_heuristic(
        self,
        *,
        index: AliasLookupIndex,
        raw_value: str,
    ) -> Optional[str]:
        matches = [
            candidate
            for candidate in index.canonical_values
            if is_safe_prefix_match(raw_value, candidate)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _unresolved_candidates(
        self,
        *,
        index: AliasLookupIndex,
        raw_value: str,
    ) -> list[GroundingCandidate]:
        raw_norm = normalize_grounding_text(raw_value)
        candidates: list[GroundingCandidate] = []
        for canonical in index.canonical_values:
            canonical_norm = normalize_grounding_text(canonical)
            score = 0.0
            evidence: list[str] = []
            if canonical_norm.startswith(raw_norm) or raw_norm.startswith(canonical_norm):
                score = 0.45
                evidence.append("prefix overlap")
            elif canonical_norm and raw_norm and canonical_norm.split(" ")[0] == raw_norm.split(" ")[0]:
                score = 0.3
                evidence.append("shared head token")
            if score > 0:
                candidates.append(
                    GroundingCandidate(
                        canonical_value=canonical,
                        strategy="candidate",
                        score=score,
                        evidence=evidence,
                    )
                )
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:5]
