"""Round-trip comparison prompt for verifying data integrity."""

from dataclasses import dataclass
from typing import List, Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"
RECONSTRUCTED_TEXT_KEYWORD = "{reconstructed_text}"
SCHEMA_KEYWORD = "{schema}"


class RoundTripComparisonSignature(dspy.Signature):
    """DSPy signature for round-trip comparison."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Original question")
    Answer: str = dspy.InputField(description="Original answer")
    Reconstructed_Text: str = dspy.InputField(
        description="Text reconstructed from database query results"
    )
    Schema: str = dspy.InputField(description="Current database schema")
    Comparison_Result: str = dspy.OutputField(
        description="JSON with comparison results and suggested fixes"
    )


@dataclass
class RoundTripComparisonPrompt(Prompt):
    """Prompt for comparing original QA against reconstructed text.

    This prompt verifies data integrity by comparing the original QA pair
    with text reconstructed from database contents.

    Attributes:
        question: Original question text
        answer: Original answer text
        reconstructed_text: Text generated from database query results
        schema: List of CREATE TABLE SQL statements
        qa_index: Index of the QA pair being verified
    """

    question: str
    answer: str
    reconstructed_text: str
    schema: List[str]
    qa_index: int = 0

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
    def Reconstructed_Text(self) -> str:
        """Get the reconstructed text."""
        return self.reconstructed_text

    @property
    def Schema(self) -> str:
        """Format schema for the prompt."""
        return "\n\n".join(self.schema)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(
            QUESTION_KEYWORD, self.Question
        )
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        user_prompt = user_prompt.replace(
            RECONSTRUCTED_TEXT_KEYWORD, self.Reconstructed_Text
        )
        user_prompt = user_prompt.replace(SCHEMA_KEYWORD, self.Schema)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return RoundTripComparisonSignature

