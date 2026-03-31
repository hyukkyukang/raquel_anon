"""Legacy schema generator kept only for backwards compatibility.

Prefer :class:`src.generator.dynamic_schema_generator.DynamicSchemaGenerator`.
"""

import logging
import warnings
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from src.aligned_db.coverage_checker import SchemaCoverageChecker
from src.aligned_db.table_schema import SchemaRegistry
from src.llm import LLMAPICaller, TooMuchThinkingError
from src.llm.parse import post_process_sql
from src.prompt import Prompt
from src.prompt.registry import (
    SCHEMA_COLUMN_DEDUPLICATION_PROMPT_REGISTRY,
    SCHEMA_COVERAGE_CHECK_PROMPT_REGISTRY,
    SCHEMA_FOREIGN_KEY_MODIFICATION_PROMPT_REGISTRY,
    SCHEMA_NORMALIZATION_PROMPT_REGISTRY,
    SCHEMA_ORDER_MODIFICATION_PROMPT_REGISTRY,
    SCHEMA_PRIMARY_KEY_MODIFICATION_PROMPT_REGISTRY,
    SCHEMA_UNIFICATION_PROMPT_REGISTRY,
    SCHEMA_UNIQUE_KEY_MODIFICATION_PROMPT_REGISTRY,
    TABLE_ANALYSIS_PROMPT_REGISTRY,
    TABLE_GENERATION_PROMPT_REGISTRY,
)

logger = logging.getLogger("SchemaGenerator")

warnings.warn(
    "src.generator.schema.SchemaGenerator is legacy. "
    "Prefer src.generator.dynamic_schema_generator.DynamicSchemaGenerator.",
    DeprecationWarning,
    stacklevel=2,
)


class SchemaGenerator:
    """
    Modular schema generation and transformation using LLMs.
    Handles fallback logic and post-processing for all schema-related LLM calls.
    Instantiates its own LLMAPICaller(s) from config. Always uses a fallback model.
    """

    SPLIT_CHAR = "\n\n"

    def __init__(
        self,
        api_cfg: DictConfig,
        global_cfg: DictConfig,
    ):
        """
        Args:
            cfg: The config for the main LLM API caller (e.g., api_cfg, smarter_api_cfg, etc.).
            global_cfg: Global configuration.
        """
        self.global_cfg = global_cfg
        self.api_cfg = api_cfg
        self.llm_api_caller = LLMAPICaller(
            global_cfg=global_cfg,
            **api_cfg.base,
        )
        self.fallback_llm_api_caller = LLMAPICaller(
            global_cfg=global_cfg,
            **api_cfg.smart,
        )
        self.coverage_checker = SchemaCoverageChecker(api_cfg, global_cfg)

    @cached_property
    def schema_unification_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_unification
        return SCHEMA_UNIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_primary_key_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_modification_primary_key
        return SCHEMA_PRIMARY_KEY_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_foreign_key_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_modification_foreign_key
        return SCHEMA_FOREIGN_KEY_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_order_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_modification_order
        return SCHEMA_ORDER_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_unique_key_modification_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_modification_unique_key
        return SCHEMA_UNIQUE_KEY_MODIFICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_normalization_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_normalization
        return SCHEMA_NORMALIZATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_column_deduplication_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_column_deduplication
        return SCHEMA_COLUMN_DEDUPLICATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def schema_coverage_check_prompt(self):
        prompt_name = self.global_cfg.prompt.schema_coverage_check
        return SCHEMA_COVERAGE_CHECK_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def table_analysis_prompt(self):
        prompt_name = self.global_cfg.prompt.get("table_analysis", "default")
        return TABLE_ANALYSIS_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def table_generation_prompt(self):
        prompt_name = self.global_cfg.prompt.get("table_generation", "default")
        return TABLE_GENERATION_PROMPT_REGISTRY[prompt_name]

    def _call_with_fallback(self, prompt: Prompt) -> List[str]:
        """Call the LLM with fallback on TooMuchThinkingError."""
        try:
            return self.llm_api_caller(
                prompt, post_process_fn=post_process_sql, prefix="schema_generation"
            )
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking: {e}")
            logger.warning("Using fallback model...")
            return self.fallback_llm_api_caller(
                prompt,
                post_process_fn=post_process_sql,
                prefix="schema_generation_fallback",
            )

    def unify_schemas(self, schema1: List[str], schema2: List[str]) -> List[str]:
        """Unify two schemas into a single schema."""
        joined_schema = self.SPLIT_CHAR.join(schema1 + schema2)
        prompt = self.schema_unification_prompt(schema=joined_schema)
        return self._call_with_fallback(prompt)

    def modify_primary_keys(self, schema: List[str]) -> List[str]:
        """Modify primary key constraints in the schema."""
        prompt = self.schema_primary_key_modification_prompt(
            schema=self.SPLIT_CHAR.join(schema)
        )
        return self._call_with_fallback(prompt)

    def modify_foreign_keys(self, schema: List[str]) -> List[str]:
        """Modify foreign key constraints in the schema."""
        prompt = self.schema_foreign_key_modification_prompt(
            schema=self.SPLIT_CHAR.join(schema)
        )
        return self._call_with_fallback(prompt)

    def modify_order(self, schema: List[str]) -> List[str]:
        """Modify the order of tables in the schema."""
        prompt = self.schema_order_modification_prompt(
            schema=self.SPLIT_CHAR.join(schema)
        )
        return self._call_with_fallback(prompt)

    def modify_unique_keys(self, schema: List[str]) -> List[str]:
        """Modify unique key constraints in the schema."""
        prompt = self.schema_unique_key_modification_prompt(
            schema=self.SPLIT_CHAR.join(schema)
        )
        return self._call_with_fallback(prompt)

    def normalize_schema(self, schema: List[str]) -> List[str]:
        """Normalize the schema."""
        prompt = self.schema_normalization_prompt(schema=self.SPLIT_CHAR.join(schema))
        return self._call_with_fallback(prompt)

    def deduplicate_columns(self, schema: List[str]) -> List[str]:
        """Merge semantically equivalent columns in the schema."""
        prompt = self.schema_column_deduplication_prompt(
            schema=self.SPLIT_CHAR.join(schema)
        )
        return self._call_with_fallback(prompt)

    # =========================================================================
    # Table-based schema operations
    # =========================================================================

    def analyze_table_requirements(
        self, current_schema: List[str], qa_pair: Tuple[str, str]
    ) -> Dict[str, Union[List[str], str]]:
        """
        Analyze which tables need to be modified or created for a new QA pair.

        Args:
            current_schema: List of existing SQL CREATE TABLE statements
            qa_pair: Tuple of (question, answer)

        Returns:
            Dict with 'modified_tables' and 'new_tables' lists, and 'reasoning' string
        """
        logger.info("Analyzing table requirements for QA pair")

        # First check coverage
        needs_modification = self.coverage_checker.check_coverage(
            current_schema, qa_pair
        )

        if not needs_modification:
            logger.info("Schema already covers QA pair, no table changes needed")
            return {
                "modified_tables": [],
                "new_tables": [],
                "reasoning": "Schema already covers the QA pair",
            }

        # Call LLM for analysis
        prompt = self.table_analysis_prompt(
            current_schema=current_schema, new_qa_pair=qa_pair
        )

        try:
            result = self._call_analysis_with_fallback(prompt)
            return result
        except Exception as e:
            logger.error(f"Failed to analyze table requirements: {e}")
            return {
                "modified_tables": [],
                "new_tables": [],
                "reasoning": f"Analysis failed: {e}",
            }

    def generate_table_schema(
        self,
        table_name: str,
        qa_pair: Tuple[str, str],
        existing_table_sql: Optional[str] = None,
        related_tables: Optional[List[str]] = None,
        requirements: str = "",
    ) -> str:
        """
        Generate schema for a specific table.

        Args:
            table_name: Name of table to generate
            qa_pair: Context from QA pair
            existing_table_sql: Existing table SQL if modifying
            related_tables: Other tables in the schema for context
            requirements: Additional requirements

        Returns:
            CREATE TABLE SQL statement
        """
        logger.info(f"Generating schema for table: {table_name}")

        if related_tables is None:
            related_tables = []

        prompt = self.table_generation_prompt(
            table_name=table_name,
            qa_pair=qa_pair,
            existing_table_sql=existing_table_sql,
            related_tables=related_tables,
            requirements=requirements,
        )

        result = self._call_with_fallback(prompt)
        if result:
            return result[0]  # Take the first (and should be only) statement
        else:
            logger.error(f"Failed to generate table schema for {table_name}")
            return f"-- Failed to generate schema for {table_name}"

    def modify_schema_table_based(
        self, current_schema: List[str], qa_pair: Tuple[str, str]
    ) -> Tuple[List[str], bool]:
        """
        Modify schema using table-based approach instead of full regeneration.

        Args:
            current_schema: List of existing SQL CREATE TABLE statements
            qa_pair: Tuple of (question, answer)

        Returns:
            Tuple of (updated_schema, was_modification_needed)
        """
        logger.info("Starting table-based schema modification")

        # Step 1: Analyze which tables need changes
        analysis = self.analyze_table_requirements(current_schema, qa_pair)

        if not analysis["modified_tables"] and not analysis["new_tables"]:
            logger.info("No table modifications needed")
            return current_schema, False

        # Step 2: Create schema registry from current schema
        registry = SchemaRegistry.from_sql_list(current_schema)

        # Step 3: Generate schemas for new tables
        for table_name in analysis["new_tables"]:
            logger.info(f"Generating new table: {table_name}")
            table_sql = self.generate_table_schema(
                table_name=table_name,
                qa_pair=qa_pair,
                related_tables=registry.to_sql_list(),
                requirements=f"Create new table for: {analysis.get('reasoning', '')}",
            )

            # Parse and add to registry
            temp_registry = SchemaRegistry.from_sql_list([table_sql])
            for table in temp_registry.tables.values():
                registry.add_table(table)

        # Step 4: Modify existing tables
        for table_name in analysis["modified_tables"]:
            logger.info(f"Modifying existing table: {table_name}")
            existing_table = registry.get_table(table_name)
            existing_sql = existing_table.to_sql() if existing_table else None

            modified_sql = self.generate_table_schema(
                table_name=table_name,
                qa_pair=qa_pair,
                existing_table_sql=existing_sql,
                related_tables=registry.to_sql_list(),
                requirements=f"Modify existing table for: {analysis.get('reasoning', '')}",
            )

            # Parse and update in registry
            temp_registry = SchemaRegistry.from_sql_list([modified_sql])
            if temp_registry.tables:
                registry.add_table(list(temp_registry.tables.values())[0])

        # Step 5: Return updated schema
        updated_schema = registry.to_sql_list()
        logger.info(f"Table-based modification complete: {len(updated_schema)} tables")
        return updated_schema, True

    def _call_analysis_with_fallback(
        self, prompt: Prompt
    ) -> Dict[str, Union[List[str], str]]:
        """Call the LLM for table analysis with fallback on errors."""
        try:
            # Use custom post-processing to get full DSPy response
            raw_response = self.llm_api_caller(
                prompt,
                post_process_fn=self._extract_full_dspy_response,
                prefix="table_analysis",
            )
            return raw_response
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking in analysis: {e}")
            logger.warning("Using fallback model for analysis...")
            raw_response = self.fallback_llm_api_caller(
                prompt,
                post_process_fn=self._extract_full_dspy_response,
                prefix="table_analysis",
            )
            return raw_response

    def _extract_full_dspy_response(
        self, response: Union[str, List[str]]
    ) -> Dict[str, Union[List[str], str]]:
        """Custom post-processor to extract full DSPy response with multiple output fields."""
        try:
            if isinstance(response, list) and len(response) == 3:
                # DSPy returned multiple fields as expected
                modified_tables_str, new_tables_str, reasoning = response
                modified_tables = self._parse_table_list(modified_tables_str)
                new_tables = self._parse_table_list(new_tables_str)
                return {
                    "modified_tables": modified_tables,
                    "new_tables": new_tables,
                    "reasoning": reasoning,
                }
            elif isinstance(response, str):
                # DSPy returned only one field, treat as modified tables
                modified_tables = self._parse_table_list(response)
                return {
                    "modified_tables": modified_tables,
                    "new_tables": [],
                    "reasoning": "Single field DSPy response",
                }
            else:
                logger.error(f"Unexpected response format: {type(response)} - {response}")
                return {
                    "modified_tables": [],
                    "new_tables": [],
                    "reasoning": f"Unexpected response format: {type(response)}",
                }
        except Exception as e:
            logger.error(f"Error processing DSPy response: {e}")
            logger.error(f"Response was: {response}")
            return {
                "modified_tables": [],
                "new_tables": [],
                "reasoning": f"Processing error: {e}",
            }

    def _parse_dspy_analysis_response(
        self, response
    ) -> Dict[str, Union[List[str], str]]:
        """Parse DSPy response with structured fields."""
        try:
            # DSPy should return a dict with the output fields
            if isinstance(response, dict):
                # Extract the structured fields
                modified_tables_str = response.get("Modified_Tables", "none")
                new_tables_str = response.get("New_Tables", "none")
                reasoning = response.get("Reasoning", "No reasoning provided")

                # Parse comma-separated lists
                modified_tables = self._parse_table_list(modified_tables_str)
                new_tables = self._parse_table_list(new_tables_str)

                return {
                    "modified_tables": modified_tables,
                    "new_tables": new_tables,
                    "reasoning": reasoning,
                }
            else:
                logger.error(f"Expected dict response from DSPy, got: {type(response)}")
                logger.error(f"Response content: {response}")
                return {
                    "modified_tables": [],
                    "new_tables": [],
                    "reasoning": f"Invalid response type: {type(response)}",
                }

        except Exception as e:
            logger.error(f"Error parsing DSPy analysis response: {e}")
            logger.error(f"Response was: {response}")
            return {
                "modified_tables": [],
                "new_tables": [],
                "reasoning": f"Parsing error: {e}",
            }

    def _parse_table_list(self, table_str: str) -> List[str]:
        """Parse comma-separated table names, handling 'none' case."""
        if not table_str or table_str.lower().strip() in ["none", "null", "empty", ""]:
            return []

        # Split by comma and clean up whitespace
        tables = [table.strip() for table in table_str.split(",")]
        # Filter out empty strings and 'none' values
        tables = [table for table in tables if table and table.lower() != "none"]

        return tables
