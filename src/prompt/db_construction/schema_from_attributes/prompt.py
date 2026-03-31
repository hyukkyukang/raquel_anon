"""Schema generation prompt for creating tables from entity types and attributes."""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import dspy

from src.prompt.base import Prompt

ENTITY_TYPES_KEYWORD = "{entity_types}"
NORMALIZED_ATTRIBUTES_KEYWORD = "{normalized_attributes}"


class SchemaFromAttributesSignature(dspy.Signature):
    """DSPy signature for schema generation from attributes."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Entity_Types: str = dspy.InputField(
        description="List of entity types to create tables for"
    )
    Normalized_Attributes: str = dspy.InputField(
        description="Normalized attributes for each entity type"
    )
    Schema_SQL: str = dspy.OutputField(
        description="PostgreSQL CREATE TABLE statements"
    )


@dataclass
class SchemaFromAttributesPrompt(Prompt):
    """Prompt for generating schema from entity types and attributes.

    This prompt creates normalized PostgreSQL CREATE TABLE statements
    based on discovered entity types and their canonical attributes.

    Attributes:
        entity_types: List of entity type dictionaries with name and description
        normalized_attributes: Dict mapping entity_type -> list of normalized attrs
    """

    entity_types: List[Dict[str, str]]
    normalized_attributes: Dict[str, List[Dict[str, Any]]]

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Entity_Types(self) -> str:
        """Format entity types for the prompt."""
        formatted: List[str] = []
        for et in self.entity_types:
            name = et.get("name", "unknown")
            desc = et.get("description", "")
            formatted.append(f"- {name}: {desc}")
        return "\n".join(formatted)

    @property
    def Normalized_Attributes(self) -> str:
        """Format normalized attributes for the prompt."""
        formatted: List[str] = []
        for entity_type, attrs in self.normalized_attributes.items():
            formatted.append(f"\n## {entity_type}")
            for attr in attrs:
                canonical = attr.get("canonical_name", "unknown")
                dtype = attr.get("data_type", "TEXT")
                desc = attr.get("description", "")
                formatted.append(f"  - {canonical} ({dtype}): {desc}")
        return "\n".join(formatted)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(
            ENTITY_TYPES_KEYWORD, self.Entity_Types
        )
        user_prompt = user_prompt.replace(
            NORMALIZED_ATTRIBUTES_KEYWORD, self.Normalized_Attributes
        )
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return SchemaFromAttributesSignature

