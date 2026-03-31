"""Prompt for QA text naturalization."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

STYLE_KEYWORD = "{style_instruction}"
QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"


class QATextNaturalizationSignature(dspy.Signature):
    """DSPy signature for QA naturalization."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Style_Instruction: str = dspy.InputField(description="Requested style guidance")
    Question: str = dspy.InputField(description="Canonical question text")
    Answer: str = dspy.InputField(description="Canonical answer text")
    Naturalized_QA: str = dspy.OutputField(
        description="JSON object with rewritten question and answer"
    )


@dataclass
class QATextNaturalizationPrompt(Prompt):
    """Prompt for rewriting one QA pair into more natural language."""

    style_instruction: str
    question: str
    answer: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Style_Instruction(self) -> str:
        return self.style_instruction

    @property
    def Question(self) -> str:
        return self.question

    @property
    def Answer(self) -> str:
        return self.answer

    def get_user_prompt(self) -> str:
        user_prompt = self.user_instruction
        user_prompt = user_prompt.replace(STYLE_KEYWORD, self.Style_Instruction)
        user_prompt = user_prompt.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        return user_prompt

    def __str__(self) -> str:
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return QATextNaturalizationSignature
