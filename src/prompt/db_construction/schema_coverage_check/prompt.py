from dataclasses import dataclass
from typing import List, Tuple, Type

import dspy

from src.prompt.base import Prompt

CURRENT_SCHEMA_KEYWORD = "{current_schema}"
NEW_QA_PAIR_KEYWORD = "{new_qa_pair}"


class SchemaCoverageCheckSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Current_Schema: str = dspy.InputField(description="Current PostgreSQL schema")
    New_QA_Pair: str = dspy.InputField(
        description="New question-answer pair to analyze"
    )
    Coverage_Result: str = dspy.OutputField(
        description="YES if schema modification needed, NO if current schema is sufficient"
    )


@dataclass
class SchemaCoverageCheckPrompt(Prompt):
    current_schema: List[str]
    new_qa_pair: Tuple[str, str]

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Current_Schema(self) -> str:
        if not self.current_schema:
            return "No schema yet"
        return "\n\n".join(self.current_schema)

    @property
    def New_QA_Pair(self) -> str:
        return f"Q: {self.new_qa_pair[0]}\nA: {self.new_qa_pair[1]}"

    def get_user_prompt(self) -> str:
        current_schema_text = self.Current_Schema
        formatted_qa_pair = self.New_QA_Pair

        user_prompt = self.user_instruction.replace(
            CURRENT_SCHEMA_KEYWORD, current_schema_text
        )
        user_prompt = user_prompt.replace(NEW_QA_PAIR_KEYWORD, formatted_qa_pair)

        return user_prompt

    def __str__(self) -> str:
        user_prompt = self.get_user_prompt()
        return f"{self.system_instruction}\n{user_prompt}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return SchemaCoverageCheckSignature
