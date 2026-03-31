"""Entity type discovery prompt for identifying entity categories from QA pairs."""

from dataclasses import dataclass
from typing import List, Tuple, Type

import dspy

from src.prompt.base import Prompt

QA_PAIRS_BATCH_KEYWORD = "{qa_pairs_batch}"


class EntityTypeDiscoverySignature(dspy.Signature):
    """DSPy signature for entity type discovery."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    QA_Pairs_Batch: str = dspy.InputField(
        description="Batch of QA pairs to analyze for entity types"
    )
    Entity_Types: str = dspy.OutputField(
        description="JSON object containing discovered entity types"
    )


@dataclass
class EntityTypeDiscoveryPrompt(Prompt):
    """Prompt for discovering entity types from QA pairs.

    This prompt analyzes QA pairs to identify what categories of entities
    are being discussed (e.g., person, work, award, organization).

    Attributes:
        qa_pairs_batch: List of (question, answer) tuples to analyze
    """

    qa_pairs_batch: List[Tuple[str, str]]

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def QA_Pairs_Batch(self) -> str:
        """Format QA pairs batch for the prompt."""
        formatted: List[str] = []
        for i, (q, a) in enumerate(self.qa_pairs_batch, 1):
            formatted.append(f"[{i}] Q: {q}\n    A: {a}")
        return "\n\n".join(formatted)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        return self.user_instruction.replace(
            QA_PAIRS_BATCH_KEYWORD, self.QA_Pairs_Batch
        )

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return EntityTypeDiscoverySignature

