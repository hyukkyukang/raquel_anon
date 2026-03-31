"""Prompt definition for semantic QA equivalence judgments."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt


class SemanticEquivalenceSignature(dspy.Signature):
    Instruction: str = dspy.InputField(
        description="System-level guidance for how to evaluate the QA pairs."
    )
    QA_Pairs: str = dspy.InputField(
        description="JSON payload containing reference/prediction pairs to judge."
    )
    Evaluation: str = dspy.OutputField(
        description="JSON response with match decisions per QA pair."
    )


@dataclass
class SemanticEquivalencePrompt(Prompt):
    """Lightweight prompt wrapper for LLM-based semantic equivalence judgments."""

    instruction: str
    payload: str

    @property
    def Instruction(self) -> str:
        return self.instruction

    @property
    def QA_Pairs(self) -> str:
        return self.payload

    @property
    def system_instruction(self) -> str:  # type: ignore[override]
        return self.instruction

    @property
    def user_instruction(self) -> str:  # type: ignore[override]
        return "Review the provided QA pairs JSON and respond exactly with the requested JSON schema."

    def get_user_prompt(self) -> str:
        return self.payload

    def __str__(self) -> str:
        return f"{self.instruction}\n{self.payload}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return SemanticEquivalenceSignature
