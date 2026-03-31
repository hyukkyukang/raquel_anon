from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt


class ResultToTextSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(
        description="The natural language question corresponding to the SQL query"
    )
    Result: str = dspy.InputField(description="A SQL execution result to convert")
    Text_sentence: str = dspy.OutputField(
        description="A natural language sentence that describes the SQL result in the context of the question"
    )


@dataclass
class ResultToTextPrompt(Prompt):
    question: str
    result: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Question(self) -> str:
        return self.question

    @property
    def Result(self) -> str:
        return self.result

    def get_user_prompt(self) -> str:
        return (
            f"Question: {self.Question}\nResult: {self.Result}\n{self.user_instruction}"
        )

    def __str__(self) -> str:
        return f"{self.system_instruction}\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return ResultToTextSignature
