from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

KEYWORD_TO_REPLACE = "{text_to_paraphrase}"


class ParaphrasingSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Text: str = dspy.InputField(description="Text to paraphrase")
    Paraphrased_Text: str = dspy.OutputField(
        description="Paraphrased version of the text"
    )


@dataclass
class ParaphrasingPrompt(Prompt):
    text: str
    text_type: str  # e.g., 'question' or 'answer'

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Text(self) -> str:
        return f"{self.text_type.capitalize()}: {self.text}"

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
        return ParaphrasingSignature
