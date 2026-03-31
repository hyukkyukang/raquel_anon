"""QA consistency check prompt for detecting native QA inconsistencies."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"


class QAConsistencyCheckSignature(dspy.Signature):
    """DSPy signature for QA consistency check."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Question to analyze")
    Answer: str = dspy.InputField(description="Answer to analyze")
    Consistency_Result: str = dspy.OutputField(
        description="JSON with consistency analysis result"
    )


@dataclass
class QAConsistencyCheckPrompt(Prompt):
    """Prompt for checking QA pair semantic consistency.

    This prompt analyzes whether an answer actually addresses what
    the question asks about, detecting native data quality issues.

    Attributes:
        question: Question text to analyze
        answer: Answer text to analyze
    """

    question: str
    answer: str

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

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return QAConsistencyCheckSignature
