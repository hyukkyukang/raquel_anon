"""Answer fact extraction prompt for extracting structured facts from QA pairs."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"


class AnswerFactExtractionSignature(dspy.Signature):
    """DSPy signature for answer fact extraction."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Question text")
    Answer: str = dspy.InputField(description="Answer text")
    Facts: str = dspy.OutputField(description="JSON with extracted facts")


@dataclass
class AnswerFactExtractionPrompt(Prompt):
    """Prompt for extracting structured facts from QA pairs.

    This prompt extracts discrete facts as subject-predicate-object triples
    that can be verified against database records.

    Attributes:
        question: Question text
        answer: Answer text
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
        return AnswerFactExtractionSignature
