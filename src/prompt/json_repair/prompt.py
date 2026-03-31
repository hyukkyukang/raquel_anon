"""JSON repair prompt for fixing malformed JSON strings."""

from dataclasses import dataclass
from typing import Type

import dspy

from src.prompt.base import Prompt


class JSONRepairSignature(dspy.Signature):
    """Signature for JSON repair task."""

    Instruction: str = dspy.InputField(description="Detailed instruction for repair")
    Malformed_JSON: str = dspy.InputField(description="The malformed JSON string")
    Error_Message: str = dspy.InputField(description="The JSON parse error message")
    Repaired_JSON: str = dspy.OutputField(description="The corrected JSON string")


@dataclass
class JSONRepairPrompt(Prompt):
    """Prompt for repairing malformed JSON strings.

    Attributes:
        json_string: The malformed JSON string to repair
        error_message: The JSONDecodeError message describing the syntax error
    """

    json_string: str
    error_message: str

    @property
    def Instruction(self) -> str:
        """Get the full instruction combining system and user prompts."""
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Malformed_JSON(self) -> str:
        """Get the malformed JSON string."""
        return self.json_string

    @property
    def Error_Message(self) -> str:
        """Get the error message."""
        return self.error_message

    def get_user_prompt(self) -> str:
        """Generate the user prompt with the JSON and error message substituted.

        Returns:
            The formatted user prompt string
        """
        prompt = self.user_instruction.replace("{json_string}", self.json_string)
        prompt = prompt.replace("{error_message}", self.error_message)
        return prompt

    def __str__(self) -> str:
        """Get the full prompt as a string."""
        return f"{self.system_instruction}\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Get the DSPy signature for this prompt.

        Returns:
            The JSONRepairSignature class
        """
        return JSONRepairSignature
