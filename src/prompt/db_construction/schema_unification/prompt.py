from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt

KEYWORD_TO_REPLACE = "{schema}"


class SchemaUnificationSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Input_Schema: str = dspy.InputField(
        description="Input schema of an Aligned Database in SQL CREATE TABLE format"
    )
    Output_Schema: str = dspy.OutputField(
        description="Output schema of an Aligned Database in SQL CREATE TABLE format"
    )


@dataclass
class SchemaUnificationPrompt(Prompt):
    schema: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Input_Schema(self) -> str:
        return self.schema

    def get_user_prompt(self) -> str:
        assert (
            KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        return self.user_instruction.replace(KEYWORD_TO_REPLACE, self.schema)

    def __str__(self) -> str:
        user_prompt = self.get_user_prompt()
        return f"{self.system_instruction}\n{user_prompt}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return SchemaUnificationSignature
