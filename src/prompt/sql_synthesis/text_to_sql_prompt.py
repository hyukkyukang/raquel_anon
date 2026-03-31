from dataclasses import dataclass
from typing import List, Type

import dspy

from src.prompt.base import Prompt


class TextToSQLSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Schema: str = dspy.InputField(
        description="A schema of an Aligned DB in SQL CREATE TABLE format"
    )
    Insert_queries: str = dspy.InputField(description="A list of insert queries")
    Question: str = dspy.InputField(
        description="A natural language question to translate to SQL"
    )
    SQL_query: str = dspy.OutputField(
        description="A SQL query that answers the question"
    )


@dataclass
class TextToSQLPrompt(Prompt):
    schema: str
    insert_queries: List[str]
    question: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Schema(self) -> str:
        return self.schema

    @property
    def Insert_queries(self) -> str:
        return "\n".join(self.insert_queries)

    @property
    def Question(self) -> str:
        return self.question

    def get_user_prompt(self) -> str:
        prompt = f"{self.user_instruction}\n\n"
        prompt += f"Schema:\n{self.Schema}\n\n"
        prompt += f"Insert Queries:\n{self.Insert_queries}\n\n"
        prompt += f"Question:\n{self.Question}\n"
        return prompt.strip()

    def __str__(self) -> str:
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return TextToSQLSignature
