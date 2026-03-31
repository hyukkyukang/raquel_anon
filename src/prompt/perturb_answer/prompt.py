from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

KEYWORD_TO_REPLACE = "{text}"


class PerturbAnswerSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Text: str = dspy.InputField(description="A question and its correct answer")
    PerturbedAnswers: str = dspy.OutputField(
        description="A list of incorrect but plausible answers"
    )


@dataclass
class PerturbAnswerPrompt(Prompt):
    question: str
    answer: str
    num_perturbations: int = 5

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Text(self) -> str:
        return f"Question: {self.question}\nCorrect Answer: {self.answer}\nGenerate {self.num_perturbations} incorrect but plausible answers."

    def get_user_prompt(self) -> str:
        assert (
            KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        return self.user_instruction.replace(KEYWORD_TO_REPLACE, self.Text)

    def __str__(self) -> str:
        user_prompt = self.get_user_prompt()
        return f"{self.system_instruction}\n{user_prompt}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return PerturbAnswerSignature
