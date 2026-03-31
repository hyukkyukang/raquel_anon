"""Round-trip judgment prompt for verifying data sufficiency."""

from dataclasses import dataclass
from typing import List, Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"
RECORDS_KEYWORD = "{records}"
SCHEMA_KEYWORD = "{schema}"


class RoundTripJudgmentSignature(dspy.Signature):
    """DSPy signature for round-trip judgment."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Original question")
    Answer: str = dspy.InputField(description="Original answer")
    Records: str = dspy.InputField(
        description="Database records retrieved for verification"
    )
    Schema: str = dspy.InputField(description="Current database schema")
    Judgment_Result: str = dspy.OutputField(
        description="JSON with sufficiency judgment, missing facts, and suggested fixes"
    )


@dataclass
class RoundTripJudgmentPrompt(Prompt):
    """Prompt for judging whether database records verify a QA pair.

    This prompt directly judges data sufficiency by comparing the original
    QA pair with database records, without intermediate answer synthesis.

    Attributes:
        question: Original question text
        answer: Original answer text
        records: Text representation of retrieved database records
        schema: List of CREATE TABLE SQL statements
        qa_index: Index of the QA pair being verified
    """

    question: str
    answer: str
    records: str
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
    def Records(self) -> str:
        """Get the records text."""
        return self.records

    @property
    def Schema(self) -> str:
        """Format schema for the prompt."""
        if isinstance(self.schema, list):
            return "\n\n".join(self.schema)
        return self.schema

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        user_prompt = user_prompt.replace(RECORDS_KEYWORD, self.Records)
        user_prompt = user_prompt.replace(SCHEMA_KEYWORD, self.Schema)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return RoundTripJudgmentSignature
