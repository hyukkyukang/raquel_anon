"""Reusable candidate/result structures for canonical grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GroundingCandidate:
    """A candidate canonical value for a raw extracted reference."""

    canonical_value: str
    strategy: str
    score: float
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the candidate to a JSON-serializable dictionary."""

        return {
            "canonical_value": self.canonical_value,
            "strategy": self.strategy,
            "score": self.score,
            "evidence": list(self.evidence),
        }


@dataclass
class GroundingResult:
    """Resolution outcome for a raw extracted reference."""

    raw_value: str
    ref_table: str
    resolved_value: Optional[str] = None
    strategy: Optional[str] = None
    candidates: List[GroundingCandidate] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """Whether the grounding result found a canonical value."""

        return self.resolved_value is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return {
            "raw_value": self.raw_value,
            "ref_table": self.ref_table,
            "resolved_value": self.resolved_value,
            "strategy": self.strategy,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
