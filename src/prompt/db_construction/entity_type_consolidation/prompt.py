"""Entity type consolidation prompt for merging entity types from multiple batches."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Type

import dspy

from src.prompt.base import Prompt

EXISTING_ENTITY_TYPES_KEYWORD = "{existing_entity_types}"
NEW_ENTITY_TYPES_KEYWORD = "{new_entity_types}"


class EntityTypeConsolidationSignature(dspy.Signature):
    """DSPy signature for entity type consolidation."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Existing_Entity_Types: str = dspy.InputField(
        description="Existing canonical entity types from previous batches"
    )
    New_Entity_Types: str = dspy.InputField(
        description="New entity types discovered from recent batches"
    )
    Consolidated_Entity_Types: str = dspy.OutputField(
        description="JSON object containing consolidated entity types"
    )


@dataclass
class EntityTypeConsolidationPrompt(Prompt):
    """Prompt for consolidating entity types from multiple batches.

    This prompt merges entity types discovered from different batches,
    deduplicating and normalizing synonymous types.

    Attributes:
        existing_entity_types: List of canonical entity types from previous batches
        new_entity_types: List of new entity types to merge
    """

    existing_entity_types: List[Dict[str, str]]
    new_entity_types: List[Dict[str, str]]

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Existing_Entity_Types(self) -> str:
        """Format existing entity types for the prompt."""
        if not self.existing_entity_types:
            return "(none - this is the first batch)"
        return json.dumps(self.existing_entity_types, indent=2)

    @property
    def New_Entity_Types(self) -> str:
        """Format new entity types for the prompt."""
        return json.dumps(self.new_entity_types, indent=2)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        prompt = self.user_instruction.replace(
            EXISTING_ENTITY_TYPES_KEYWORD, self.Existing_Entity_Types
        )
        prompt = prompt.replace(NEW_ENTITY_TYPES_KEYWORD, self.New_Entity_Types)
        return prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return EntityTypeConsolidationSignature

