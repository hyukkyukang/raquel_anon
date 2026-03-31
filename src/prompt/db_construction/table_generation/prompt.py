from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional

import dspy

from src.prompt.base import Prompt

TABLE_NAME_KEYWORD = "{table_name}"
QA_CONTEXT_KEYWORD = "{qa_context}"
EXISTING_TABLE_KEYWORD = "{existing_table}"
RELATED_TABLES_KEYWORD = "{related_tables}"
REQUIREMENTS_KEYWORD = "{requirements}"


class TableGenerationSignature(dspy.Signature):
    """Signature for table generation prompt."""
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Table_Name: str = dspy.InputField(description="Name of table to generate")
    QA_Context: str = dspy.InputField(description="Question-answer context")
    Existing_Table: str = dspy.InputField(description="Existing table schema (if any)")
    Related_Tables: str = dspy.InputField(description="Related tables in schema")
    Requirements: str = dspy.InputField(description="Specific requirements")
    Table_Schema: str = dspy.OutputField(
        description="CREATE TABLE statement for the requested table"
    )


@dataclass
class TableGenerationPrompt(Prompt):
    """Prompt for generating individual table schemas."""
    table_name: str
    qa_pair: Tuple[str, str]
    existing_table_sql: Optional[str] = None
    related_tables: List[str] = None
    requirements: str = ""

    def __post_init__(self):
        if self.related_tables is None:
            self.related_tables = []

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Table_Name(self) -> str:
        return self.table_name

    @property
    def QA_Context(self) -> str:
        return f"Q: {self.qa_pair[0]}\nA: {self.qa_pair[1]}"

    @property
    def Existing_Table(self) -> str:
        if self.existing_table_sql:
            return self.existing_table_sql
        return "None (creating new table)"

    @property
    def Related_Tables(self) -> str:
        if not self.related_tables:
            return "None"
        return "\n\n".join(self.related_tables)

    @property
    def Requirements(self) -> str:
        if not self.requirements:
            return "Generate appropriate schema based on QA context"
        return self.requirements

    def get_user_prompt(self) -> str:
        user_prompt = self.user_instruction.replace(TABLE_NAME_KEYWORD, self.Table_Name)
        user_prompt = user_prompt.replace(QA_CONTEXT_KEYWORD, self.QA_Context)
        user_prompt = user_prompt.replace(EXISTING_TABLE_KEYWORD, self.Existing_Table)
        user_prompt = user_prompt.replace(RELATED_TABLES_KEYWORD, self.Related_Tables)
        user_prompt = user_prompt.replace(REQUIREMENTS_KEYWORD, self.Requirements)

        return user_prompt

    def __str__(self) -> str:
        user_prompt = self.get_user_prompt()
        return f"{self.system_instruction}\n{user_prompt}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return TableGenerationSignature