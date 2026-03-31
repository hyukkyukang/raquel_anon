"""Attribute normalization prompt for clustering and standardizing attributes."""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import dspy

from src.prompt.base import Prompt

RAW_ATTRIBUTES_KEYWORD = "{raw_attributes}"


class AttributeNormalizationSignature(dspy.Signature):
    """DSPy signature for attribute normalization."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Raw_Attributes: str = dspy.InputField(
        description="Raw attributes to normalize, grouped by entity type"
    )
    Normalized_Attributes: str = dspy.OutputField(
        description="JSON object containing normalized attribute mappings"
    )


@dataclass
class AttributeNormalizationPrompt(Prompt):
    """Prompt for normalizing discovered attributes.

    This prompt clusters similar attributes and creates canonical names,
    e.g., mapping "birthplace", "hometown", "born_in" → "birth_place".

    Attributes:
        raw_attributes: Dictionary mapping entity_type -> list of attribute dicts
    """

    raw_attributes: Dict[str, List[Dict[str, Any]]]

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Raw_Attributes(self) -> str:
        """Format raw attributes for the prompt."""
        formatted: List[str] = []
        for entity_type, attrs in self.raw_attributes.items():
            formatted.append(f"\n## {entity_type}")
            for attr in attrs:
                name = attr.get("name", "unknown")
                dtype = attr.get("data_type", "TEXT")
                desc = attr.get("description", "")
                formatted.append(f"  - {name} ({dtype}): {desc}")
        return "\n".join(formatted)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        return self.user_instruction.replace(
            RAW_ATTRIBUTES_KEYWORD, self.Raw_Attributes
        )

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return AttributeNormalizationSignature

