from dataclasses import dataclass
from typing import List, Optional, Type

import dspy

from src.prompt.base import Prompt


class TemplateSpecGenerationSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed template design brief")
    Schema: str = dspy.InputField(description="Schema with join hints")
    ValueHints: str = dspy.InputField(description="Representative values and ranges")
    RecentFailures: str = dspy.InputField(description="Common errors to avoid")
    TemplateJSON: str = dspy.OutputField(description="Strict JSON template specification")


@dataclass
class TemplateSpecGenerationPrompt(Prompt):
    """Prompt that asks the LLM to emit a TemplateSpec JSON blob."""

    type_name: str
    type_description: str
    schema_text: str
    value_hints: str
    recent_failures: Optional[List[str]] = None
    additional_guidance: Optional[str] = None

    @property
    def Instruction(self) -> str:
        extra = f"\nAdditional guidance:\n{self.additional_guidance.strip()}\n" if self.additional_guidance else ""
        return (
            f"{self.system_instruction}\n\n"
            f"Query type: {self.type_name}\n"
            f"Intent: {self.type_description}\n"
            f"{extra}"
        ).strip()

    @property
    def Schema(self) -> str:
        return self.schema_text.strip()

    @property
    def ValueHints(self) -> str:
        return self.value_hints.strip() if self.value_hints else "(No value hints provided)"

    @property
    def RecentFailures(self) -> str:
        if not self.recent_failures:
            return "None."
        bullet_list = "\n".join(f"- {failure}" for failure in self.recent_failures)
        return f"Recent validation failures to avoid:\n{bullet_list}"

    def get_user_prompt(self) -> str:
        prompt_sections = [
            self.user_instruction.strip(),
            f"Type details:\n- Name: {self.type_name}\n- Description: {self.type_description}",
            f"Schema:\n{self.Schema}",
            f"Value hints:\n{self.ValueHints}",
            self.RecentFailures,
            "Respond with a single JSON object that matches the required schema."
            " Do not wrap the JSON in markdown.",
        ]
        return "\n\n".join(section for section in prompt_sections if section.strip())

    def __str__(self) -> str:
        return f"{self.Instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return TemplateSpecGenerationSignature
