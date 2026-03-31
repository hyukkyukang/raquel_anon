"""Modular SQL query generation and transformation using LLMs.

This module handles SQL generation, transformation, and correction using LLM-based
approaches, with fallback logic and post-processing for all SQL-related operations.
"""

import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from src.constants import SQL_STATEMENT_SEPARATOR
from src.llm import TooMuchThinkingError
from src.llm.parse import post_process_sql
from src.prompt import Prompt
from src.prompt.registry import (
    QUERY_GENERATION_PROMPT_REGISTRY,
    SQL_SYNTAX_CORRECTION_PROMPT_REGISTRY,
    TEXT_TO_SQL_PROMPT_REGISTRY,
    UPDATE_NULL_MODIFICATION_PROMPT_REGISTRY,
    UPSERT_NULL_MODIFICATION_PROMPT_REGISTRY,
    UPSERT_NULLIFY_GENERATION_PROMPT_REGISTRY,
)
from src.utils.llm import LLMGeneratorMixin

logger = logging.getLogger("src.generator.sql")


class SQLGenerator(LLMGeneratorMixin):
    """SQL query generation and transformation using LLMs.

    Handles fallback logic and post-processing for all SQL-related LLM calls.
    Uses LLMGeneratorMixin for standardized LLM management.

    Attributes:
        api_cfg: Configuration for LLM API callers
        global_cfg: Global configuration
    """

    SPLIT_CHAR: str = SQL_STATEMENT_SEPARATOR

    def __init__(
        self,
        api_cfg: DictConfig,
        global_cfg: DictConfig,
    ):
        """
        Args:
            api_cfg: The config for the LLM API callers (base and fallback).
            global_cfg: Global configuration.
        """
        super().__init__(api_cfg, global_cfg)

    def _call_with_fallback(self, prompt: Prompt) -> List[str]:
        """Call the LLM with fallback on TooMuchThinkingError."""

        def _call_llm(llm_caller, is_fallback=False):
            prefix = "sql_generation_fallback" if is_fallback else "sql_generation"
            return llm_caller(
                prompt,
                post_process_fn=post_process_sql,
                prefix=prefix,
            )

        try:
            return _call_llm(self.llm_api_caller)
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking: {e}")
            logger.warning("Using fallback model...")
            return _call_llm(self.fallback_llm_api_caller, is_fallback=True)

    def call_raw(self, prompt_text: str) -> str:
        """Call the LLM with a raw text prompt (no Prompt object needed).

        This is useful for simple prompts like template slot filling.

        Args:
            prompt_text: The full prompt text to send to the LLM

        Returns:
            Raw response from the LLM
        """
        try:
            # Use the underlying API caller's custom API method
            return self.llm_api_caller._call_custom_api(
                system_instruction="You are a helpful assistant.",
                user_prompt=prompt_text,
                temperature=0.7,  # Some creativity for diverse outputs
            )
        except Exception as e:
            logger.error(f"Error in raw LLM call: {e}")
            return ""

    @cached_property
    def text_to_sql_prompt(self):
        prompt_name = self.global_cfg.prompt.text_to_sql
        return TEXT_TO_SQL_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def query_generation_prompt(self):
        # This is a registry with multiple types (where, groupby, etc.)
        return QUERY_GENERATION_PROMPT_REGISTRY

    @cached_property
    def upsert_nullify_generation_prompt(self):
        prompt_name = self.global_cfg.prompt.upsert_nullify_generation
        return UPSERT_NULLIFY_GENERATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def upsert_null_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.upsert_null_modification
        return UPSERT_NULL_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def update_null_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.update_null_modification
        return UPDATE_NULL_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def sql_syntax_correction_prompt(self):
        prompt_name = self.global_cfg.prompt.sql_syntax_correction
        return SQL_SYNTAX_CORRECTION_PROMPT_REGISTRY[prompt_name]

    def text_to_sql(self, schema: str, insert_queries: List[str], question: str) -> str:
        """Generate a SQL query from natural language question and schema."""
        prompt = self.text_to_sql_prompt(
            schema=schema, insert_queries=insert_queries, question=question
        )
        sql_statements: List[str] = self._call_with_fallback(prompt)
        return sql_statements[0] if sql_statements else ""

    def synthesize_query(
        self,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
        history: List[str],
        extra_instruction: str,
        type_name: str,
        last_error: Optional[str] = None,
        max_columns_per_table: int = 15,
        column_stats: Optional[Dict[str, Dict[str, int]]] = None,
        sample_values: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Generate queries of a specific type (where, groupby, etc.).

        Args:
            schema: Database schema string
            table_data: Dictionary of table data for context
            history: Previously generated queries to avoid duplicates
            extra_instruction: Type-specific instruction
            type_name: Query type (e.g., 'join', 'groupby')
            last_error: Optional error message from previous attempt for error-aware retry
            max_columns_per_table: Maximum columns to show per table in schema
            column_stats: Optional column statistics for smart data formatting
            sample_values: Optional sample values for generating valid queries

        Returns:
            List of generated SQL queries
        """
        prompt_cls = self.query_generation_prompt[type_name]
        prompt = prompt_cls(
            schema=schema,
            table_data=table_data,
            history=history,
            extra_instruction=extra_instruction,
            type_name=type_name,
            last_error=last_error,
            max_columns_per_table=max_columns_per_table,
            column_stats=column_stats,
            sample_values=sample_values,
        )
        return self._call_with_fallback(prompt)

    def nullify_update(
        self,
        schema: List[str],
        insert_statements: List[str],
        qa_pair: Tuple[str, str],
    ) -> List[str]:
        """Generate nullify update SQL statements."""
        prompt = self.upsert_nullify_generation_prompt(
            schema=schema, insert_statements=insert_statements, qa_pair=qa_pair
        )
        return self._call_with_fallback(prompt)

    def modify_nullify_update(
        self,
        schema: List[str],
        insert_statements: List[str],
        qa_pair: Tuple[str, str],
        update_statement: str,
        error_msg: str,
    ) -> List[str]:
        """Modify a nullify update statement based on an error message."""
        prompt = self.update_null_modification_prompt(
            schema=schema,
            insert_statements=insert_statements,
            qa_pair=qa_pair,
            update_statement=update_statement,
            error_msg=error_msg,
        )
        return self._call_with_fallback(prompt)

    def sql_syntax_correction(
        self,
        sql: str,
        error_message: Optional[str] = None,
        schema_context: Optional[List[str]] = None,
    ) -> List[str]:
        """Correct SQL syntax using LLM.

        Args:
            sql: The SQL statement to correct
            error_message: Optional database error message for better context
            schema_context: Optional list of related CREATE TABLE statements
                for FK reference context

        Returns:
            List of corrected SQL statements
        """
        prompt = self.sql_syntax_correction_prompt(
            sql=sql, error_message=error_message, schema_context=schema_context
        )
        return self._call_with_fallback(prompt)

    def generate(self, prompt: Prompt) -> List[str]:
        """Generic SQL generation from a Prompt."""
        return self._call_with_fallback(prompt)
