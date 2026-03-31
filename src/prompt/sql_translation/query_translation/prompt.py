from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt


class QueryTranslationSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    SQL_query: str = dspy.InputField(description="A SQL query to translate")
    Text_query: str = dspy.OutputField(
        description="A natural language sentence that is semantically equivalent to the SQL query"
    )


@dataclass
class QueryTranslationPrompt(Prompt):
    sql_query: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def SQL_query(self) -> str:
        return self.sql_query

    def get_user_prompt(self) -> str:
        return f"{self.user_instruction}\n{self.SQL_query}"

    def __str__(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return QueryTranslationSignature
