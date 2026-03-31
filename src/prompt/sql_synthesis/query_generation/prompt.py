from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import dspy

from src.prompt.base import Prompt
from src.prompt.sql_synthesis.query_generation.examples import (
    format_examples_for_prompt,
)
from src.utils.table_data import (
    format_sample_data_smart,
    format_sample_values_for_prompt,
    format_schema_with_columns,
)


class QueryGenerationSignature(dspy.Signature):
    Instruction: str = dspy.InputField(description="Detailed instruction")
    Schema: str = dspy.InputField(
        description="A schema of a Aligned DB in SQL CREATE TABLE format"
    )
    Insert_queries: str = dspy.InputField(
        description="Sample table data or insert queries"
    )
    History: str = dspy.InputField(
        description="A list of previously generated SQL queries to avoid repeating"
    )
    Extra_instruction: str = dspy.InputField(
        description="Additional instruction for the LLM, e.g., to avoid repeated queries"
    )
    SQL_queries: str = dspy.OutputField(
        description="A list of SQL queries to retrieve the answer to the question"
    )


@dataclass
class QueryGenerationPrompt(Prompt):
    schema: str
    table_data: Dict[str, List[Dict[str, Any]]]
    history: List[str]
    extra_instruction: str = ""

    # Query type for few-shot examples
    type_name: str = ""

    # Error message from previous attempt (for error-aware retry)
    last_error: Optional[str] = None

    # For backward compatibility - will be removed later
    insert_queries: List[str] = None

    # Column statistics for smart data formatting (non-null counts per column)
    column_stats: Optional[Dict[str, Dict[str, int]]] = None

    # Sample values from database for generating valid queries
    sample_values: Optional[Dict[str, Dict[str, Any]]] = None

    # Feature flags
    use_explicit_columns: bool = True
    use_few_shot_examples: bool = True
    use_error_aware_retry: bool = True
    use_smart_data_formatting: bool = True
    use_sample_values: bool = True

    # Schema formatting options
    max_columns_per_table: int = 15

    @property
    def Instruction(self) -> str:
        return f"{self.system_instruction}\n{self.user_instruction}\n{self.extra_instruction}"

    @property
    def Schema(self) -> str:
        # Add explicit column listing if enabled (with column limiting)
        if self.use_explicit_columns:
            return format_schema_with_columns(
                self.schema,
                pg_client=None,  # No DB connection in prompt context
                max_columns_per_table=self.max_columns_per_table,
            )
        return self.schema

    @property
    def Examples(self) -> str:
        """Get few-shot examples for the current query type."""
        if not self.use_few_shot_examples or not self.type_name:
            return ""
        return format_examples_for_prompt(self.type_name)

    @property
    def Insert_queries(self) -> str:
        # Support backward compatibility first
        if self.insert_queries is not None:
            return "\n".join(self.insert_queries)

        # Format table data as readable data samples
        if not self.table_data:
            return "(No table data available.)"

        # Use smart formatting if enabled and column stats available
        if self.use_smart_data_formatting:
            return format_sample_data_smart(
                self.table_data,
                column_stats=self.column_stats,
                max_rows=5,
                max_columns_display=12,
            )

        # Legacy formatting (fallback)
        formatted_data = []
        for table_name, rows in self.table_data.items():
            if not rows:
                formatted_data.append(f"-- Table {table_name}: (empty)")
                continue

            formatted_data.append(f"-- Table {table_name} sample data:")
            # Show first few rows as examples
            sample_rows = rows[:3]  # Show max 3 rows as samples
            for i, row in enumerate(sample_rows, 1):
                formatted_data.append(f"-- Row {i}: {row}")
            if len(rows) > 3:
                formatted_data.append(f"-- ... and {len(rows) - 3} more rows")
            formatted_data.append("")  # Empty line between tables

        return "\n".join(formatted_data)

    @property
    def History(self) -> str:
        if not self.history:
            return "(No previously generated queries.)"
        return "\n".join(
            f"-- Previously generated query {i+1}:\n{q}"
            for i, q in enumerate(self.history)
        )

    @property
    def Extra_instruction(self) -> str:
        return self.extra_instruction

    @property
    def SampleValues(self) -> str:
        """Get formatted sample values for query generation."""
        if not self.use_sample_values or not self.sample_values:
            return ""
        return format_sample_values_for_prompt(self.sample_values)

    def get_user_prompt(self) -> str:
        prompt = f"{self.user_instruction}\n\n"
        prompt += f"Schema:\n{self.Schema}\n\n"

        # Use appropriate label based on data type
        data_label = (
            "Insert Queries" if self.insert_queries is not None else "Table Data"
        )
        prompt += f"{data_label}:\n{self.Insert_queries}\n\n"

        # Add sample values to help generate valid queries
        sample_vals = self.SampleValues
        if sample_vals:
            prompt += f"{sample_vals}\n\n"

        # Add few-shot examples if available
        examples = self.Examples
        if examples:
            prompt += f"{examples}\n\n"

        if self.history:
            prompt += (
                f"Previously Generated Queries (avoid repeating):\n{self.History}\n\n"
            )

        # Add error feedback from previous attempt if available
        if self.use_error_aware_retry and self.last_error:
            prompt += "=== PREVIOUS ATTEMPT FAILED ===\n"
            prompt += f"Error: {self.last_error}\n"
            prompt += "Please avoid this error in your next query.\n"
            prompt += "=" * 30 + "\n\n"

        if self.extra_instruction:
            prompt += f"Additional Instruction:\n{self.Extra_instruction}\n\n"
        return prompt.strip()

    def __str__(self) -> str:
        return f"{self.system_instruction}\n\n{self.get_user_prompt()}"

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        return QueryGenerationSignature
