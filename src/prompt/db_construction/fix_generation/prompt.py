"""Fix generation prompt for generating database fixes from missing facts."""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import dspy

from src.prompt.base import Prompt

QUESTION_KEYWORD = "{question}"
ANSWER_KEYWORD = "{answer}"
MISSING_FACTS_KEYWORD = "{missing_facts}"
SCHEMA_KEYWORD = "{schema}"


class FixGenerationSignature(dspy.Signature):
    """DSPy signature for fix generation."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    Question: str = dspy.InputField(description="Original question")
    Answer: str = dspy.InputField(description="Original answer")
    Missing_Facts: str = dspy.InputField(description="Facts missing from database")
    Schema: str = dspy.InputField(description="Available tables and columns")
    Fixes: str = dspy.OutputField(description="JSON array of fix objects")


@dataclass
class FixGenerationPrompt(Prompt):
    """Prompt for generating database fixes from missing facts.

    This prompt takes missing facts identified during verification and
    generates specific INSERT/UPDATE statements to add the missing data.

    Attributes:
        question: Original question text
        answer: Original answer text
        missing_facts: List of facts that are missing from the database
        schema: List of CREATE TABLE SQL statements
        table_columns: Dict mapping table names to their column lists
    """

    question: str
    answer: str
    missing_facts: List[Dict[str, Any]]
    schema: List[str]
    table_columns: Dict[str, List[str]]

    @property
    def Instruction(self) -> str:
        """Get the combined instruction with substituted values."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @property
    def Question(self) -> str:
        """Get the question."""
        return self.question

    @property
    def Answer(self) -> str:
        """Get the answer."""
        return self.answer

    @property
    def Missing_Facts(self) -> str:
        """Format missing facts for the prompt."""
        if not self.missing_facts:
            return "No missing facts identified."

        lines: List[str] = []
        for i, fact in enumerate(self.missing_facts, 1):
            fact_text = fact.get("fact", str(fact))
            severity = fact.get("severity", "unknown")
            lines.append(f"{i}. [{severity}] {fact_text}")
        return "\n".join(lines)

    @property
    def Schema(self) -> str:
        """Format schema with table/column information for the prompt."""
        lines: List[str] = []
        for table_name, columns in self.table_columns.items():
            lines.append(f"Table: {table_name}")
            lines.append(f"  Columns: {', '.join(columns)}")
            lines.append("")
        return "\n".join(lines)

    def get_user_prompt(self) -> str:
        """Generate the user prompt with substituted values."""
        user_prompt = self.user_instruction.replace(QUESTION_KEYWORD, self.Question)
        user_prompt = user_prompt.replace(ANSWER_KEYWORD, self.Answer)
        user_prompt = user_prompt.replace(MISSING_FACTS_KEYWORD, self.Missing_Facts)
        user_prompt = user_prompt.replace(SCHEMA_KEYWORD, self.Schema)
        return user_prompt

    def __str__(self) -> str:
        """Return string representation of the prompt."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return the DSPy signature for this prompt."""
        return FixGenerationSignature

