"""Round-trip lookup prompt for generating database lookup plans."""

from dataclasses import dataclass
from typing import List, Type

import dspy

from src.prompt.base import Prompt

SCHEMA_KEYWORD = "{{ schema }}"
QUESTION_KEYWORD = "{{ question }}"
ANSWER_KEYWORD = "{{ answer }}"


class RoundTripLookupSignature(dspy.Signature):
    """Generate a structured lookup plan to verify a QA pair against a database schema."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    DB_Schema: str = dspy.InputField(desc="The SQL schema of the database")
    Question: str = dspy.InputField(desc="The question being verified")
    Answer: str = dspy.InputField(desc="The expected answer")
    Lookup_Plan: str = dspy.OutputField(
        desc="JSON object containing the list of table lookups"
    )


@dataclass
class RoundTripLookupPrompt(Prompt):
    """Prompt for generating lookup plans for round-trip verification.

    This prompt instructs the LLM to analyze the QA pair and schema to
    determine which database records need to be queried to verify the answer.

    Attributes:
        schema: The SQL schema of the database (can be string or list)
        question: The question being verified
        answer: The expected answer
    """

    schema: str
    question: str
    answer: str

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def DB_Schema(self) -> str:
        """Get the database schema."""
        if isinstance(self.schema, list):
            return "\n\n".join(self.schema)
        return self.schema

    @property
    def Question(self) -> str:
        """Get the question."""
        return self.question

    @property
    def Answer(self) -> str:
        """Get the answer."""
        return self.answer

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(SCHEMA_KEYWORD, self.DB_Schema)
        user_prompt = user_prompt.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return RoundTripLookupSignature
