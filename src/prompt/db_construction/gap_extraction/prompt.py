"""Gap extraction prompt for targeted re-extraction of missing facts."""

from dataclasses import dataclass
from typing import List, Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"
MISSING_FACTS_KEYWORD = "{missing_facts}"
ENTITY_TYPES_KEYWORD = "{entity_types}"
RELATION_TYPES_KEYWORD = "{relation_types}"


class GapExtractionSignature(dspy.Signature):
    """DSPy signature for gap extraction."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Question text")
    Answer: str = dspy.InputField(description="Answer text")
    Missing_Facts: str = dspy.InputField(description="Facts not captured in extraction")
    Entity_Types: str = dspy.InputField(description="Available entity types")
    Relation_Types: str = dspy.InputField(description="Available relation types")
    Extraction: str = dspy.OutputField(description="JSON with entities and relations")


@dataclass
class GapExtractionPrompt(Prompt):
    """Prompt for targeted extraction of missing facts.

    This prompt extracts specific entities and relations to fill gaps
    identified in the initial extraction.

    Attributes:
        question: Question text
        answer: Answer text
        missing_facts: List of missing fact descriptions
        entity_types: List of available entity type names
    """

    question: str
    answer: str
    missing_facts: List[str]
    entity_types: List[str]
    relation_types: str

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Question(self) -> str:
        """Get the question."""
        return self.question

    @property
    def Answer(self) -> str:
        """Get the answer."""
        return self.answer

    @property
    def Missing_Facts(self) -> str:
        """Get missing facts as formatted string."""
        return "\n".join(f"- {fact}" for fact in self.missing_facts)

    @property
    def Entity_Types(self) -> str:
        """Get entity types as formatted string."""
        return ", ".join(self.entity_types)

    @property
    def Relation_Types(self) -> str:
        """Get relation types as formatted string."""
        return self.relation_types

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        user_prompt = user_prompt.replace(MISSING_FACTS_KEYWORD, self.Missing_Facts)
        user_prompt = user_prompt.replace(ENTITY_TYPES_KEYWORD, self.Entity_Types)
        user_prompt = user_prompt.replace(RELATION_TYPES_KEYWORD, self.Relation_Types)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return GapExtractionSignature
