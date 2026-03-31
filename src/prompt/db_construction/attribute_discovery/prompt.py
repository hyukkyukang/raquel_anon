"""Attribute discovery prompt for identifying entity attributes from QA pairs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import dspy

from src.prompt.base import Prompt

ENTITY_TYPES_KEYWORD = "{entity_types}"
QA_PAIRS_BATCH_KEYWORD = "{qa_pairs_batch}"


class AttributeDiscoverySignature(dspy.Signature):
    """DSPy signature for attribute discovery."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Entity_Types: str = dspy.InputField(
        description="List of entity types to find attributes for"
    )
    QA_Pairs_Batch: str = dspy.InputField(
        description="Batch of QA pairs to analyze for attributes"
    )
    Discovered_Attributes: str = dspy.OutputField(
        description="JSON object containing discovered attributes per entity type"
    )


@dataclass
class AttributeDiscoveryPrompt(Prompt):
    """Prompt for discovering attributes for each entity type.

    This prompt analyzes QA pairs to identify what attributes/properties
    are mentioned for each entity type.

    Attributes:
        entity_types: List of entity type dictionaries with name and description
        qa_pairs_batch: List of (question, answer) tuples to analyze
    """

    entity_types: List[Dict[str, str]]
    qa_pairs_batch: List[Tuple[str, str]]

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
    def QA_Pairs_Batch(self) -> str:
        """Format QA pairs batch for the prompt."""
        formatted: List[str] = []
        for i, (q, a) in enumerate(self.qa_pairs_batch, 1):
            formatted.append(f"[{i}] Q: {q}\n    A: {a}")
        return "\n\n".join(formatted)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(
            ENTITY_TYPES_KEYWORD, self.Entity_Types
        )
        user_prompt = user_prompt.replace(
            QA_PAIRS_BATCH_KEYWORD, self.QA_Pairs_Batch
        )
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return AttributeDiscoverySignature

