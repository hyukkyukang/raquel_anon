"""Per-QA extraction prompt for extracting values from a single QA pair."""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import dspy

from src.prompt.base import Prompt

SCHEMA_KEYWORD = "{schema}"
ENTITY_TYPES_KEYWORD = "{entity_types}"
RELATION_TYPES_KEYWORD = "{relation_types}"
QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"


class PerQAExtractionSignature(dspy.Signature):
    """DSPy signature for per-QA entity extraction."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Schema: str = dspy.InputField(
        description="Database schema as CREATE TABLE statements"
    )
    EntityTypes: str = dspy.InputField(
        description="List of entity types and their attributes"
    )
    RelationTypes: str = dspy.InputField(
        description="List of relationship types (junction tables)"
    )
    Question: str = dspy.InputField(description="The question from the QA pair")
    Answer: str = dspy.InputField(description="The answer from the QA pair")
    Extracted_Data: str = dspy.OutputField(
        description="JSON object containing extracted entities and relations"
    )


@dataclass
class PerQAExtractionPrompt(Prompt):
    """Prompt for extracting entities from a single QA pair.

    This prompt extracts structured entities and relations from a single
    QA pair given the database schema and type definitions.

    Attributes:
        schema: List of CREATE TABLE SQL statements
        entity_types: Dict mapping entity type to list of attribute names
        relation_types: List of relation type definitions
        question: The question text
        answer: The answer text
    """

    schema: List[str]
    entity_types: Dict[str, List[str]]
    relation_types: List[Dict[str, Any]]
    question: str
    answer: str

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Schema(self) -> str:
        """Format schema for the prompt."""
        return "\n\n".join(self.schema)

    @property
    def EntityTypes(self) -> str:
        """Format entity types for the prompt."""
        lines: List[str] = []
        for entity_type, attrs in self.entity_types.items():
            lines.append(f"- {entity_type}: {', '.join(attrs)}")
        return "\n".join(lines) if lines else "(none)"

    @property
    def RelationTypes(self) -> str:
        """Format relation types for the prompt."""
        lines: List[str] = []
        for rel in self.relation_types:
            name = rel.get("name", "unknown")
            source = rel.get("source_entity", "?")
            target = rel.get("target_entity", "?")
            lines.append(f"- {name}: {source} <-> {target}")
        return "\n".join(lines) if lines else "(none)"

    @property
    def Question(self) -> str:
        """Get the question text."""
        return self.question

    @property
    def Answer(self) -> str:
        """Get the answer text."""
        return self.answer

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction
        user_prompt = user_prompt.replace(SCHEMA_KEYWORD, self.Schema)
        user_prompt = user_prompt.replace(ENTITY_TYPES_KEYWORD, self.EntityTypes)
        user_prompt = user_prompt.replace(RELATION_TYPES_KEYWORD, self.RelationTypes)
        user_prompt = user_prompt.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return PerQAExtractionSignature
