"""Role metadata for extracted attributes and grounding decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AttributeRole(str, Enum):
    """Role an extracted attribute should play in the schema."""

    SCALAR = "scalar"
    ENTITY_REFERENCE = "entity_reference"
    SELF_REFERENCE = "self_reference"
    CONTROLLED_VALUE = "controlled_value"
    RELATION_ATTRIBUTE = "relation_attribute"
    UNKNOWN = "unknown"


def parse_attribute_role(value: Optional[str | "AttributeRole"]) -> Optional["AttributeRole"]:
    """Parse a serialized role value into an enum instance."""

    if value is None or value == "":
        return None
    if isinstance(value, AttributeRole):
        return value
    try:
        return AttributeRole(str(value))
    except ValueError:
        return None


@dataclass
class RolePrediction:
    """Predicted role metadata for an attribute."""

    role: AttributeRole
    target_table: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return {
            "role": self.role.value,
            "target_table": self.target_table,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolePrediction":
        """Create a prediction from serialized data."""

        role = parse_attribute_role(data.get("role")) or AttributeRole.UNKNOWN
        return cls(
            role=role,
            target_table=data.get("target_table"),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            evidence=list(data.get("evidence", [])),
        )
