from dataclasses import dataclass
from typing import List, Tuple, Type

import dspy

from src.prompt.base import Prompt

SCHEMA_KEYWORD_TO_REPLACE = "{list_of_create_table_statements}"
INSERT_STATEMENTS_KEYWORD_TO_REPLACE = "{list_of_insert_statements}"
QA_PAIRS_KEYWORD_TO_REPLACE = "{list_of_qa_pairs}"
UPSERT_STATEMENT_TO_MODIFY_KEYWORD_TO_REPLACE = "{upsert_statement_to_modify}"
ERROR_MESSAGE_KEYWORD_TO_REPLACE = "{error_message}"


class UpsertNullModificationSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Schema: str = dspy.InputField(
        description="Schema of an Aligned Database in SQL CREATE TABLE format"
    )
    Insert_Statements: str = dspy.InputField(
        description="Insert statements (tuples) that contain the data to be nullified"
    )
    QA_Pair: str = dspy.InputField(description="A question-answer pair")
    Upsert_Statement_To_Modify: str = dspy.InputField(
        description="A PostgreSQL-compatible UPSERT statement to modify"
    )
    Error_Message: str = dspy.InputField(
        description="An PostgreSQL error message when executing the upsert statement to modify"
    )
    Modified_Upsert_Statement: str = dspy.OutputField(
        description="A PostgreSQL-compatible UPSERT statement that nullifies the content"
    )


@dataclass
class UpsertNullModificationPrompt(Prompt):
    schema: List[str]
    insert_statements: List[str]
    qa_pair: Tuple[str, str]
    upsert_statement: str
    error_msg: str

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}"

    @property
    def Schema(self) -> str:
        return "\n\n".join(self.schema)

    @property
    def Insert_Statements(self) -> str:
        return "\n\n".join(self.insert_statements)

    @property
    def QA_Pair(self) -> str:
        return f"Q: {self.qa_pair[0]}\nA: {self.qa_pair[1]}"

    @property
    def Upsert_Statement_To_Modify(self) -> str:
        return self.upsert_statement

    @property
    def Error_Message(self) -> str:
        return self.error_msg

    def get_user_prompt(self) -> str:
        assert (
            SCHEMA_KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{SCHEMA_KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        assert (
            INSERT_STATEMENTS_KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{INSERT_STATEMENTS_KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        assert (
            QA_PAIRS_KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{QA_PAIRS_KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        assert (
            UPSERT_STATEMENT_TO_MODIFY_KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{UPSERT_STATEMENT_TO_MODIFY_KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        assert (
            ERROR_MESSAGE_KEYWORD_TO_REPLACE in self.user_instruction
        ), f"{ERROR_MESSAGE_KEYWORD_TO_REPLACE} not found in prompt: {self.user_instruction}"
        return (
            self.user_instruction.replace(SCHEMA_KEYWORD_TO_REPLACE, self.Schema)
            .replace(INSERT_STATEMENTS_KEYWORD_TO_REPLACE, self.Insert_Statements)
            .replace(QA_PAIRS_KEYWORD_TO_REPLACE, self.QA_Pair)
            .replace(
                UPSERT_STATEMENT_TO_MODIFY_KEYWORD_TO_REPLACE,
                self.Upsert_Statement_To_Modify,
            )
            .replace(ERROR_MESSAGE_KEYWORD_TO_REPLACE, self.Error_Message)
        )

    def __str__(self) -> str:
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return UpsertNullModificationSignature
