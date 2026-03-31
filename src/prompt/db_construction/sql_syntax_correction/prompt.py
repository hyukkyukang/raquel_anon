from dataclasses import dataclass, field
from typing import List, Optional, Type

import dspy

from src.prompt.base import Prompt

SQL_KEYWORD_TO_REPLACE = "{sql_to_be_corrected}"
ERROR_KEYWORD_TO_REPLACE = "{error_message}"
SCHEMA_KEYWORD_TO_REPLACE = "{schema_context}"


class SQLSyntaxCorrectionSignature(dspy.Signature):
    """Signature for SQL syntax correction task."""

    Instruction: str = dspy.InputField(description="Detailed instruction")
    SQL: str = dspy.InputField(description="A SQL statement to be corrected")
    Error_Message: str = dspy.InputField(
        description="The database error message (if available)"
    )
    Schema_Context: str = dspy.InputField(
        description="Related schema context (if available)"
    )
    Corrected_SQL: str = dspy.OutputField(description="A corrected SQL statement")


@dataclass
class SQLSyntaxCorrectionPrompt(Prompt):
    """Prompt for correcting SQL syntax errors.

    Attributes:
        sql: The SQL statement that failed
        error_message: Optional error message from the database for better context
        schema_context: Optional list of related CREATE TABLE statements for FK context
    """

    sql: str
    error_message: Optional[str] = field(default=None)
    schema_context: Optional[List[str]] = field(default=None)

    @property
    def Instruction(self) -> str:
        """Get the full instruction combining system and user prompts."""
        return f"{self.system_instruction}\n{self.get_user_prompt()}"

    @property
    def SQL(self) -> str:
        """Get the SQL statement to correct."""
        return self.sql

    @property
    def Error_Message(self) -> str:
        """Get the error message, or empty string if not provided."""
        return self.error_message or ""

    @property
    def Schema_Context(self) -> str:
        """Get the schema context, or empty string if not provided."""
        if self.schema_context:
            return "\n\n".join(self.schema_context)
        return ""

    def get_user_prompt(self) -> str:
        """Generate the user prompt with SQL, error message, and schema substituted.

        Returns:
            The formatted user prompt string
        """
        prompt = self.user_instruction.replace(SQL_KEYWORD_TO_REPLACE, self.sql)

        # Include error message if provided
        if self.error_message:
            prompt = prompt.replace(ERROR_KEYWORD_TO_REPLACE, self.error_message)
        else:
            # Remove the error message placeholder line if no error provided
            prompt = prompt.replace(
                f"\nDatabase error: {ERROR_KEYWORD_TO_REPLACE}\n", "\n"
            )
            prompt = prompt.replace(f"Database error: {ERROR_KEYWORD_TO_REPLACE}", "")

        # Include schema context if provided
        if self.schema_context:
            schema_str = "\n\n".join(self.schema_context)
            prompt = prompt.replace(SCHEMA_KEYWORD_TO_REPLACE, schema_str)
        else:
            # Remove the schema context section if not provided
            prompt = prompt.replace(
                f"\nRelated schema (for reference):\n{SCHEMA_KEYWORD_TO_REPLACE}\n", "\n"
            )
            prompt = prompt.replace(
                f"Related schema (for reference):\n{SCHEMA_KEYWORD_TO_REPLACE}", ""
            )

        return prompt

    def __str__(self) -> str:
        """Get the full prompt as a string."""
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Get the DSPy signature for this prompt."""
        return SQLSyntaxCorrectionSignature
