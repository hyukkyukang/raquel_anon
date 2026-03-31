"""Round-trip answer prompt for synthesizing answers from database records."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{{ question }}"
RECORDS_KEYWORD = "{{ records }}"


class RoundTripAnswerSignature(dspy.Signature):
    """Synthesize an answer from raw database records, or report missing data."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(desc="The user's question")
    Records: str = dspy.InputField(
        desc="Raw text representation of database records found"
    )
    Synthesized_Answer: str = dspy.OutputField(
        desc="Synthesized answer or 'MISSING_DATA'"
    )


@dataclass
class RoundTripAnswerPrompt(Prompt):
    """Prompt for synthesizing answers from database records.

    This prompt instructs the LLM to generate an answer using ONLY the
    provided database records, or return MISSING_DATA if insufficient.

    Attributes:
        question: The user's question
        records: Raw text representation of database records found
    """

    question: str
    records: str

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Question(self) -> str:
        """Get the question."""
        return self.question

    @property
    def Records(self) -> str:
        """Get the records."""
        return self.records

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(RECORDS_KEYWORD, self.Records)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return RoundTripAnswerSignature
