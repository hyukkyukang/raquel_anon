"""SQL query synthesis from database schema.

This module generates SQL queries that can be used to retrieve data
from the aligned database, using LLM-based query synthesis.

Supports parallel query generation by type for improved performance.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hkkang_utils.pg as pg_utils
from omegaconf import DictConfig
from tqdm import tqdm

from src.generator.query_templates import (
    TEMPLATE_TYPES,
    QueryTemplate,
    get_slot_filling_prompt,
    get_template,
    is_template_type,
    parse_slot_values,
    validate_slot_values,
)
from src.generator.sql import SQLGenerator
from src.generator.template_instantiator import TemplateInstantiator
from src.generator.template_spec import TemplateSpec
from src.generator.template_spec_generator import TemplateSpecGenerator
from src.generator.template_witness import TemplateWitnessSampler
from src.llm.exception import SQLParsingError
from src.utils.async_utils import AsyncRateLimiter
from src.utils.table_data import (
    estimate_column_statistics_from_rows,
    extract_columns_from_schema,
    extract_table_names_from_schema_str,
    extract_sample_values_from_rows,
    fetch_table_data,
    format_schema_with_columns,
    format_join_hints_for_prompt,
    format_sample_values_for_prompt,
    get_all_column_statistics,
    get_column_sample_values,
    get_valid_join_pairs,
)

logger = logging.getLogger("src.generator.synthesizer")


@dataclass
class TypeGenerationResult:
    """Result of generating queries for a single type."""

    type_name: str
    queries: List[str]
    success_count: int
    total_attempts: int
    metadata: List[Dict[str, Any]] = field(default_factory=list)


# Query type definitions: (type_name, instruction)
# Single-feature types focus on one SQL feature
# Composed types combine multiple features for complex queries
QUERY_TYPES: List[Tuple[str, str]] = [
    # Single-feature types
    (
        "where",
        "Generate queries with WHERE clauses using comparisons, IN, LIKE, or date conditions.",
    ),
    (
        "groupby",
        "Generate queries with GROUP BY and aggregations (COUNT, SUM, AVG, MIN, MAX).",
    ),
    (
        "subquery",
        "Generate queries with subqueries in SELECT, FROM, or WHERE clauses.",
    ),
    (
        "join",
        "Generate queries joining 2 tables using INNER, LEFT, or RIGHT JOIN.",
    ),
    (
        "orderby",
        "Generate queries with ORDER BY (ASC/DESC) and LIMIT for top-N results.",
    ),
    (
        "having",
        "Generate queries with GROUP BY and HAVING to filter aggregated results.",
    ),
    (
        "distinct",
        "Generate queries using DISTINCT or COUNT(DISTINCT column).",
    ),
    (
        "like",
        "Generate queries using LIKE or ILIKE for pattern matching on text fields.",
    ),
    (
        "null_check",
        "Generate queries checking for NULL or NOT NULL values.",
    ),
    (
        "between",
        "Generate queries using BETWEEN for numeric or date ranges.",
    ),
    (
        "case_when",
        "Generate queries with CASE WHEN for conditional column values.",
    ),
    # Composed types (combine multiple features)
    (
        "join_groupby",
        "Generate queries that JOIN tables and use GROUP BY with aggregations.",
    ),
    (
        "join_where_orderby",
        "Generate queries that JOIN tables with WHERE filters and ORDER BY.",
    ),
    (
        "subquery_aggregation",
        "Generate queries with subqueries that include aggregation functions.",
    ),
    (
        "union_orderby",
        "Generate queries using UNION to combine results, with final ORDER BY.",
    ),
    (
        "groupby_having_orderby",
        "Generate queries with GROUP BY, HAVING filter, and ORDER BY.",
    ),
    (
        "multi_join",
        "Generate queries joining 3+ tables with meaningful conditions.",
    ),
    (
        "exists_subquery",
        "Generate queries using EXISTS or NOT EXISTS with correlated subqueries.",
    ),
    (
        "in_subquery",
        "Generate queries using IN or NOT IN with subquery results.",
    ),
    (
        "comparison_subquery",
        "Generate queries comparing values against subquery results (>, <, =, etc.).",
    ),
]


class QuerySynthesizer:
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.num_per_type: int = self.cfg.num_per_type
        self.instruction_suffix: str = self.cfg.instruction_suffix
        self.max_retries: int = self.cfg.max_retries
        self.generation_mode: str = self.cfg.get("generation_mode", "legacy")
        self.templates_per_type: int = self.cfg.get("templates_per_type", 4)
        self.max_instantiations_per_template: int = self.cfg.get(
            "max_instantiations_per_template", 200
        )
        self.max_witness_candidates: int = self.cfg.get("max_witness_candidates", 500)
        self.max_instantiation_attempts: int = self.cfg.get(
            "max_instantiation_attempts", 1000
        )
        self.template_spec_recent_failure_buffer: int = self.cfg.get(
            "template_spec_recent_failure_buffer", 5
        )
        self.template_spec_guidance: str = self.cfg.get("template_spec_guidance", "")
        self.template_instantiation_seed: Optional[int] = self.cfg.get(
            "template_instantiation_seed"
        )
        self._template_cache_needs_clear: bool = bool(
            self.cfg.get("template_spec_cache_overwrite", False)
        )
        self.sql_generator = SQLGenerator(global_cfg.llm, global_cfg)
        data_root = Path(self.global_cfg.project_path) / self.global_cfg.paths.data_dir
        self.template_debug_dir = data_root / "aligned_db" / "template_debug"
        self.template_debug_dir.mkdir(parents=True, exist_ok=True)
        self._latest_metadata: Optional[List[Dict[str, Any]]] = None

    @cached_property
    def max_concurrency(self) -> int:
        """Maximum number of concurrent type generations (parallel types)."""
        return self.cfg.get("max_concurrency", 5)

    @cached_property
    def requests_per_second(self) -> float:
        """Maximum LLM API requests per second for rate limiting."""
        return self.cfg.get("requests_per_second", 10.0)

    @cached_property
    def use_templates_for_complex(self) -> bool:
        """Whether to use templates for complex query types."""
        return self.cfg.get("use_templates_for_complex", True)

    @cached_property
    def template_types(self) -> List[str]:
        """List of query types that should use template-based generation."""
        return self.cfg.get("template_types", list(TEMPLATE_TYPES))

    @cached_property
    def queries_per_call(self) -> int:
        """Number of queries to request per LLM call (batch generation)."""
        return self.cfg.get("queries_per_call", 5)

    @cached_property
    def max_columns_per_table(self) -> int:
        """Maximum columns to show per table in schema for LLM."""
        return self.cfg.get("max_columns_per_table", 15)

    @cached_property
    def table_sample_row_limit(self) -> int:
        """Maximum rows to sample per table for prompt context."""
        return self.cfg.get("table_sample_row_limit", 25)

    @cached_property
    def use_database_profile_queries(self) -> bool:
        """Whether to run extra database-wide profiling queries for prompt context."""
        return self.cfg.get("use_database_profile_queries", False)

    @cached_property
    def max_history_queries_in_prompt(self) -> int:
        """Maximum previous queries to replay verbatim in the prompt."""
        return self.cfg.get("max_history_queries_in_prompt", 8)

    @cached_property
    def max_history_characters(self) -> int:
        """Maximum character budget for replayed history in each prompt."""
        return self.cfg.get("max_history_characters", 4000)

    @cached_property
    def api_max_retries(self) -> int:
        """Maximum retries for transient API errors."""
        return self.cfg.get("api_max_retries", 3)

    @cached_property
    def api_retry_delay(self) -> float:
        """Delay in seconds between API retries."""
        return self.cfg.get("api_retry_delay", 2.0)

    @cached_property
    def pg_client(self) -> pg_utils.PostgresConnector:
        """Get or create the PostgreSQL database connection."""
        return pg_utils.PostgresConnector(
            db_id=self.global_cfg.database.db_id,
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.port,
        )

    @cached_property
    def template_spec_generator(self) -> TemplateSpecGenerator:
        cache_dir = self.cfg.get("template_spec_cache_dir")
        return TemplateSpecGenerator(
            api_cfg=self.global_cfg.llm,
            global_cfg=self.global_cfg,
            cache_dir=cache_dir,
        )

    @cached_property
    def template_witness_sampler(self) -> TemplateWitnessSampler:
        return TemplateWitnessSampler(self.pg_client, debug_dir=self.template_debug_dir)

    @cached_property
    def template_instantiator(self) -> TemplateInstantiator:
        return TemplateInstantiator(seed=self.template_instantiation_seed)

    def _get_prompt_table_data(self, schema: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get sampled table data for prompt context.

        Args:
            schema: Database schema to extract table names from

        Returns:
            Dictionary mapping table names to their row data
        """
        table_names = extract_table_names_from_schema_str(schema)
        return fetch_table_data(
            self.pg_client,
            table_names,
            log_fetches=True,
            max_rows_per_table=self.table_sample_row_limit,
        )

    @staticmethod
    def _normalize_query_text(sql: str) -> str:
        """Normalize SQL text for bounded-history summaries."""
        return " ".join(sql.strip().split())

    def _build_prompt_history(self, history: List[str]) -> List[str]:
        """Keep only a bounded recent window of previously generated queries."""
        if not history:
            return []

        trimmed = list(history[-self.max_history_queries_in_prompt :])
        while (
            sum(len(query) for query in trimmed) > self.max_history_characters
            and len(trimmed) > 1
        ):
            trimmed.pop(0)
        return trimmed

    def _summarize_query_shape(self, sql: str) -> str:
        """Build a compact shape summary for a generated SQL query."""
        normalized = self._normalize_query_text(sql.lower())
        features: List[str] = []
        for keyword, label in [
            (" join ", "join"),
            (" where ", "where"),
            (" group by ", "groupby"),
            (" having ", "having"),
            (" order by ", "orderby"),
            (" distinct ", "distinct"),
            (" union ", "union"),
            (" exists ", "exists"),
        ]:
            if keyword in f" {normalized} ":
                features.append(label)

        tables = re.findall(r'\b(?:from|join)\s+"?([a-zA-Z_][\w]*)', normalized)
        table_summary = ",".join(list(dict.fromkeys(tables))[:3])
        feature_summary = "/".join(features) if features else "select"
        return (
            f"{feature_summary} on {table_summary}"
            if table_summary
            else feature_summary
        )

    def _build_history_summary(self, history: List[str]) -> str:
        """Summarize older history without replaying every prior query verbatim."""
        if len(history) <= self.max_history_queries_in_prompt:
            return ""

        recent_count = min(len(history), self.max_history_queries_in_prompt)
        older_history = history[: -recent_count]
        shapes: List[str] = []
        seen: set[str] = set()
        for sql in reversed(older_history):
            signature = self._summarize_query_shape(sql)
            if signature in seen:
                continue
            seen.add(signature)
            shapes.append(signature)
            if len(shapes) >= 6:
                break

        if not shapes:
            return ""
        ordered_shapes = list(reversed(shapes))
        return (
            "Avoid repeating these older query shapes: "
            + "; ".join(ordered_shapes)
            + "."
        )

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if an error is transient and should be retried.

        Args:
            error: The exception to check

        Returns:
            True if the error is transient (502, 503, 429, timeout, etc.)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for common transient error patterns
        transient_patterns = [
            "502",
            "503",
            "504",
            "429",  # Rate limit
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "rate limit",
            "too many requests",
            "timeout",
            "connection",
            "temporary",
            "overloaded",
        ]

        return any(
            pattern in error_str or pattern in error_type
            for pattern in transient_patterns
        )

    def _call_llm_with_retry(
        self,
        type_name: str,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
        column_stats: Optional[Dict[str, Dict[str, int]]],
        sample_values: Optional[Dict[str, Dict[str, Any]]],
        history: List[str],
        extra_instruction: str,
        last_error: Optional[str],
    ) -> List[str]:
        """Call LLM to generate queries with retry logic for transient errors.

        Args:
            type_name: Query type name
            schema: Database schema
            table_data: Table data for context
            column_stats: Column statistics for smart data formatting
            sample_values: Sample values from database
            history: Previously generated queries
            extra_instruction: Additional instruction
            last_error: Error from previous attempt

        Returns:
            List of generated SQL statements

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.api_max_retries):
            try:
                return self.sql_generator.synthesize_query(
                    schema=schema,
                    table_data=table_data,
                    column_stats=column_stats,
                    sample_values=sample_values,
                    history=history,
                    extra_instruction=extra_instruction,
                    type_name=type_name,
                    last_error=last_error,
                    max_columns_per_table=self.max_columns_per_table,
                )
            except Exception as e:
                last_exception = e
                if self._is_transient_error(e):
                    if attempt < self.api_max_retries - 1:
                        delay = self.api_retry_delay * (
                            attempt + 1
                        )  # Exponential backoff
                        logger.warning(
                            f"[{type_name}] Transient API error (attempt {attempt + 1}/{self.api_max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{type_name}] Transient API error persisted after {self.api_max_retries} retries: {e}"
                        )
                        raise
                else:
                    # Non-transient error, don't retry
                    raise

        # Should not reach here, but just in case
        raise last_exception

    def _generate_query_from_template(
        self,
        type_name: str,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[Tuple[str, QueryTemplate]]:
        """Generate a query using template-based approach.

        Args:
            type_name: The query type (must be a template type)
            schema: Database schema string
            table_data: Dictionary of table data

        Returns:
            Tuple of (SQL query, template) or None if generation failed
        """
        template = get_template(type_name)
        if not template:
            return None

        # Get schema info with columns
        schema_info = format_schema_with_columns(schema)

        # Format sample data
        sample_data_parts = []
        for table_name, rows in table_data.items():
            if rows:
                sample_data_parts.append(f"-- {table_name}: {rows[:2]}")
        sample_data = "\n".join(sample_data_parts) if sample_data_parts else "(no data)"

        # Get available tables and columns for validation
        available_tables = extract_table_names_from_schema_str(schema)
        available_columns = extract_columns_from_schema(schema)
        column_names = {
            table: [col[0] for col in cols] for table, cols in available_columns.items()
        }

        # Generate slot-filling prompt
        prompt = get_slot_filling_prompt(template, schema_info, sample_data)

        try:
            # Call LLM to fill slots
            response = self.sql_generator.call_raw(prompt)

            # Parse slot values
            slot_values = parse_slot_values(response)
            if not slot_values:
                logger.warning(f"[{type_name}] Failed to parse slot values from LLM")
                return None

            # Validate slot values
            is_valid, error = validate_slot_values(
                slot_values, template, available_tables, column_names
            )
            if not is_valid:
                logger.warning(f"[{type_name}] Invalid slot values: {error}")
                return None

            # Fill template
            sql = template.fill(slot_values)
            return (sql, template)

        except Exception as e:
            logger.error(f"[{type_name}] Template generation failed: {e}")
            return None

    def _fix_sql_syntax_error(self, sql: str, error_message: str) -> str:
        """Use LLM to fix SQL syntax errors.

        Args:
            sql: The SQL statement with syntax errors
            error_message: The error message from SQLParsingError

        Returns:
            Fixed SQL statement
        """
        try:
            logger.info(f"Attempting to fix SQL syntax error: {error_message}")
            corrected_sql_list = self.sql_generator.sql_syntax_correction(
                sql, error_message=error_message
            )
            if corrected_sql_list:
                corrected_sql = corrected_sql_list[0]  # Take the first corrected SQL
                logger.info(f"SQL correction successful: {corrected_sql[:100]}...")
                return corrected_sql
            else:
                logger.warning("SQL correction returned empty result")
                return sql
        except Exception as e:
            logger.error(f"Error during SQL syntax correction: {e}")
            return sql

    def _handle_sql_parsing_error(
        self, original_sql: str, error_message: str, max_corrections: int = 3
    ) -> List[str]:
        """Handle SQL parsing errors by attempting to fix them with LLM.

        Args:
            original_sql: The original SQL that failed to parse
            error_message: The error message from SQLParsingError
            max_corrections: Maximum number of correction attempts (default: 3)

        Returns:
            List of corrected SQL statements, empty if all correction attempts failed
        """
        current_sql = original_sql

        for correction_attempt in range(max_corrections):
            logger.info(
                f"SQL correction attempt {correction_attempt + 1}/{max_corrections}"
            )
            corrected_sql = current_sql  # Initialize to prevent unbound variable

            try:
                # Use LLM to fix the SQL
                corrected_sql = self._fix_sql_syntax_error(current_sql, error_message)

                if corrected_sql == current_sql:
                    logger.warning("LLM returned the same SQL, no correction made")
                    continue

                # Try to parse the corrected SQL to see if it's valid now
                from src.llm.parse import post_process_sql

                corrected_statements = post_process_sql(corrected_sql)

                if corrected_statements:
                    logger.info(
                        f"SQL correction successful after {correction_attempt + 1} attempts"
                    )
                    return corrected_statements
                else:
                    logger.warning(
                        f"Corrected SQL is empty, attempt {correction_attempt + 1} failed"
                    )
                    current_sql = (
                        corrected_sql  # Use the corrected SQL for next attempt
                    )

            except SQLParsingError as e:
                logger.warning(
                    f"Correction attempt {correction_attempt + 1} still has parsing error: {e.message}"
                )
                # Update current_sql with the attempted correction
                current_sql = corrected_sql
                error_message = str(e)  # Update error message for next attempt
            except Exception as e:
                logger.error(
                    f"Unexpected error during correction attempt {correction_attempt + 1}: {e}"
                )
                break

        logger.error(f"Failed to correct SQL after {max_corrections} attempts")
        return []

    def _load_from_cache(self, cache_file_path: str) -> List[str]:
        """Load generated queries from cache file.

        Args:
            cache_file_path: Path to the cache file

        Returns:
            List of cached queries, or empty list if cache doesn't exist
        """
        try:
            if not os.path.exists(cache_file_path):
                logger.debug(f"Cache file does not exist: {cache_file_path}")
                return []

            with open(cache_file_path, "r") as f:
                content = f.read().strip()

            if not content:
                logger.debug(f"Cache file is empty: {cache_file_path}")
                return []

            # Split by separator to get individual queries
            queries: List[str] = content.split(self.global_cfg.paths.separator + "\n")
            # Filter out empty strings from trailing separators
            queries = [q for q in queries if q.strip()]

            logger.info(f"Loaded {len(queries)} queries from cache: {cache_file_path}")
            return queries

        except Exception as e:
            logger.error(f"Error loading cache file {cache_file_path}: {e}")
            return []

    def _metadata_cache_path(self) -> Optional[str]:
        """Return the metadata cache path if configured."""
        if not hasattr(self.global_cfg.paths, "sql_queries_metadata"):
            return None
        return os.path.join(
            self.global_cfg.project_path,
            self.global_cfg.paths.data_dir,
            self.global_cfg.paths.sql_queries_metadata,
        )

    def _load_metadata_cache(
        self, metadata_path: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Load metadata list from cache if available."""
        if not metadata_path or not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            logger.warning("Failed to load metadata cache: %s", exc)
        return None

    @staticmethod
    def _make_template_id(text: str) -> str:
        """Create a stable identifier for a template string."""
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"sha1:{digest}"

    def _build_query_metadata(
        self,
        *,
        sql: str,
        query_type: str,
        template_source: str,
        template_name: Optional[str] = None,
        template_sql: Optional[str] = None,
        placeholder_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build a metadata record for a generated query."""
        metadata: Dict[str, Any] = {
            "sql": sql,
            "query_type": query_type,
            "template_source": template_source,
            "generation_mode": self.generation_mode,
        }
        if template_sql:
            metadata["template_sql"] = template_sql
            metadata["template_id"] = self._make_template_id(template_sql)
        if template_name:
            metadata["template_name"] = template_name
        if placeholder_values:
            metadata["placeholder_values"] = placeholder_values
        return metadata

    def _generate_queries_for_type(
        self,
        type_name: str,
        extra_instruction: str,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
        column_stats: Optional[Dict[str, Dict[str, int]]] = None,
        sample_values: Optional[Dict[str, Dict[str, Any]]] = None,
        schema_prompt: Optional[str] = None,
        value_hints: Optional[str] = None,
        pbar: Optional[tqdm] = None,
        pbar_lock: Optional[threading.Lock] = None,
    ) -> TypeGenerationResult:
        """Generate queries for a single query type using batch generation.

        Uses batch generation to reduce LLM calls - requests multiple queries
        per call and keeps all valid ones.

        Args:
            type_name: Name of the query type (e.g., 'where', 'join')
            extra_instruction: Type-specific instruction for the LLM
            schema: Database schema string
            table_data: Dictionary of table data for context
            column_stats: Optional column statistics for smart data formatting
            sample_values: Optional sample values for query generation
            pbar: Optional progress bar to update
            pbar_lock: Optional lock for thread-safe progress bar updates

        Returns:
            TypeGenerationResult with generated queries and statistics
        """
        if self.generation_mode == "template_instantiation":
            return self._generate_queries_with_templates(
                type_name=type_name,
                type_description=extra_instruction,
                schema=schema,
                schema_prompt=schema_prompt or "",
                value_hints_text=value_hints or "",
                sample_values=sample_values or {},
                pbar=pbar,
                pbar_lock=pbar_lock,
            )

        type_queries: List[str] = []
        query_metadata: List[Dict[str, Any]] = []
        total_attempts = 0
        consecutive_failures = 0

        # Check if this type should use template-based generation
        use_template = (
            self.use_templates_for_complex
            and type_name in self.template_types
            and is_template_type(type_name)
        )

        # Keep generating until we have enough queries or hit retry limit
        while len(type_queries) < self.num_per_type:
            queries_needed = self.num_per_type - len(type_queries)
            batch_size = min(self.queries_per_call, queries_needed)
            last_error: Optional[str] = None
            batch_valid_count = 0

            total_attempts += 1

            # Try template-based generation first for complex types
            if use_template and len(type_queries) == 0:
                logger.debug(f"[{type_name}] Trying template-based generation...")
                template_result = self._generate_query_from_template(
                    type_name, schema, table_data
                )
                if template_result:
                    template_sql, template = template_result
                else:
                    template_sql, template = None, None

                if template_sql and template_sql not in type_queries:
                    success, error_msg = self._execute_and_validate_query(template_sql)
                    if success:
                        type_queries.append(template_sql)
                        query_metadata.append(
                            self._build_query_metadata(
                                sql=template_sql,
                                query_type=type_name,
                                template_source="slot_filling",
                                template_name=template.name if template else None,
                                template_sql=template.template if template else None,
                            )
                        )
                        batch_valid_count += 1
                        logger.debug(f"[{type_name}] Template generation succeeded")
                        # Update progress
                        if pbar:
                            if pbar_lock:
                                with pbar_lock:
                                    pbar.update(1)
                            else:
                                pbar.update(1)
                    else:
                        last_error = error_msg

            # Generate batch of queries via LLM (with retry for transient errors)
            try:
                sql_statements: List[str] = self._call_llm_with_retry(
                    type_name=type_name,
                    schema=schema,
                    table_data=table_data,
                    column_stats=column_stats,
                    sample_values=sample_values,
                    history=self._build_prompt_history(type_queries),
                    extra_instruction=f"Generate exactly {batch_size} distinct queries. "
                    + extra_instruction
                    + " "
                    + self._build_history_summary(type_queries)
                    + " "
                    + self.instruction_suffix,
                    last_error=last_error,  # Pass previous error to LLM
                )
            except SQLParsingError as e:
                logger.warning(f"[{type_name}] SQL parsing error: {e.message}")
                sql_statements = self._handle_sql_parsing_error(e.sql, str(e))
                if not sql_statements:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_retries:
                        logger.warning(
                            f"[{type_name}] Too many consecutive failures, stopping early"
                        )
                        break
                    continue
            except Exception as e:
                # Non-recoverable error (after retries for transient errors)
                logger.error(f"[{type_name}] Generation failed with error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= self.max_retries:
                    logger.warning(
                        f"[{type_name}] Too many consecutive failures, stopping early"
                    )
                    break
                continue

            # Validate each query in the batch and keep all valid ones
            for sql in sql_statements:
                if len(type_queries) >= self.num_per_type:
                    break  # We have enough

                if sql in type_queries:
                    continue  # Skip duplicates

                success, error_msg = self._execute_and_validate_query(sql)
                if success:
                    type_queries.append(sql)
                    query_metadata.append(
                        self._build_query_metadata(
                            sql=sql,
                            query_type=type_name,
                            template_source="freeform",
                        )
                    )
                    batch_valid_count += 1
                    # Update progress for each valid query
                    if pbar:
                        if pbar_lock:
                            with pbar_lock:
                                pbar.update(1)
                        else:
                            pbar.update(1)
                else:
                    last_error = error_msg  # Track for next batch

            # Track consecutive failures
            if batch_valid_count == 0:
                consecutive_failures += 1
                logger.debug(
                    f"[{type_name}] Batch produced no valid queries (attempt {consecutive_failures}/{self.max_retries})"
                )
                if consecutive_failures >= self.max_retries:
                    logger.warning(
                        f"[{type_name}] Too many consecutive failures, stopping early"
                    )
                    break
            else:
                consecutive_failures = 0  # Reset on success
                logger.debug(
                    f"[{type_name}] Batch produced {batch_valid_count} valid queries, total: {len(type_queries)}/{self.num_per_type}"
                )

        success_count = len(type_queries)
        logger.info(
            f"[{type_name}] Completed: {success_count}/{self.num_per_type} queries in {total_attempts} LLM calls"
        )

        return TypeGenerationResult(
            type_name=type_name,
            queries=type_queries,
            metadata=query_metadata,
            success_count=success_count,
            total_attempts=total_attempts,
        )

    def _generate_queries_with_templates(
        self,
        *,
        type_name: str,
        type_description: str,
        schema: str,
        schema_prompt: str,
        value_hints_text: str,
        sample_values: Dict[str, Dict[str, Any]],
        pbar: Optional[tqdm],
        pbar_lock: Optional[threading.Lock],
    ) -> TypeGenerationResult:
        queries: List[str] = []
        query_metadata: List[Dict[str, Any]] = []
        normalized: set = set()
        total_attempts = 0
        recent_failures: List[str] = []
        duplicates_skipped = 0
        witness_resamples = 0

        if not value_hints_text and sample_values:
            value_hints_text = format_sample_values_for_prompt(sample_values)

        def failure_tail() -> List[str]:
            if not recent_failures:
                return []
            return recent_failures[-self.template_spec_recent_failure_buffer :]

        overwrite_cache = False
        if self._template_cache_needs_clear:
            overwrite_cache = True
            self._template_cache_needs_clear = False

        specs = self.template_spec_generator.ensure_specs(
            type_name=type_name,
            type_description=type_description,
            schema_text=schema_prompt,
            value_hints=value_hints_text,
            target_count=self.templates_per_type,
            overwrite=overwrite_cache,
            recent_failures=failure_tail(),
            additional_guidance=self.template_spec_guidance or None,
        )

        while (
            len(queries) < self.num_per_type
            and total_attempts < self.max_instantiation_attempts
        ):
            spec_progress = False
            specs_with_no_witnesses = 0
            for spec in specs:
                witness_pool = self.template_witness_sampler.sample_all_groups(
                    spec, max_candidates=self.max_witness_candidates
                )
                if not witness_pool or any(
                    len(rows) == 0 for rows in witness_pool.values()
                ):
                    specs_with_no_witnesses += 1
                    continue

                instantiations_for_spec = 0
                while (
                    instantiations_for_spec < self.max_instantiations_per_template
                    and len(queries) < self.num_per_type
                    and total_attempts < self.max_instantiation_attempts
                ):
                    instantiations_for_spec += 1
                    total_attempts += 1
                    instantiation = self.template_instantiator.instantiate(
                        spec, witness_pool
                    )
                    if instantiation is None:
                        witness_pool = self.template_witness_sampler.sample_all_groups(
                            spec, max_candidates=self.max_witness_candidates
                        )
                        witness_resamples += 1
                        continue

                    sql = instantiation.sql.strip()
                    normalized_sql = " ".join(sql.split())
                    if normalized_sql in normalized:
                        duplicates_skipped += 1
                        continue

                    success, error_msg = self._execute_and_validate_query(sql)
                    if success:
                        queries.append(sql)
                        normalized.add(normalized_sql)
                        query_metadata.append(
                            self._build_query_metadata(
                                sql=sql,
                                query_type=type_name,
                                template_source="template_instantiation",
                                template_name=spec.description or spec.type_name,
                                template_sql=spec.sql_template,
                                placeholder_values=instantiation.placeholder_values,
                            )
                        )
                        spec_progress = True
                        self._update_progress_bar(pbar, pbar_lock, 1)
                        if len(queries) >= self.num_per_type:
                            break
                    else:
                        self._record_failed_query(type_name, sql, error_msg)
                        if error_msg:
                            recent_failures.append(error_msg)
                            limit = self.template_spec_recent_failure_buffer
                            if limit > 0 and len(recent_failures) > limit:
                                del recent_failures[:-limit]
                        witness_pool = self.template_witness_sampler.sample_all_groups(
                            spec, max_candidates=self.max_witness_candidates
                        )
                        witness_resamples += 1

                if (
                    len(queries) >= self.num_per_type
                    or total_attempts >= self.max_instantiation_attempts
                ):
                    break

            # If all specs failed witness sampling, break early - database is too sparse
            if specs_with_no_witnesses == len(specs) and len(specs) > 0:
                logger.warning(
                    "[%s] All %d templates failed witness sampling - database too sparse",
                    type_name,
                    len(specs),
                )
                break

            if (
                len(queries) >= self.num_per_type
                or total_attempts >= self.max_instantiation_attempts
            ):
                break

        logger.info(
            "[%s] Template instantiation produced %d/%d queries in %d attempts",
            type_name,
            len(queries),
            self.num_per_type,
            total_attempts,
        )
        logger.info(
            "[%s] Template metrics: instantiations=%d, duplicates_skipped=%d, "
            "witness_resamples=%d, execution_failures=%d",
            type_name,
            total_attempts,
            duplicates_skipped,
            witness_resamples,
            len(recent_failures),
        )

        return TypeGenerationResult(
            type_name=type_name,
            queries=queries,
            metadata=query_metadata,
            success_count=len(queries),
            total_attempts=total_attempts,
        )

    async def _generate_queries_for_type_async(
        self,
        type_name: str,
        extra_instruction: str,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
        column_stats: Optional[Dict[str, Dict[str, int]]],
        sample_values: Optional[Dict[str, Dict[str, Any]]],
        schema_prompt: Optional[str],
        value_hints: Optional[str],
        semaphore: asyncio.Semaphore,
        rate_limiter: AsyncRateLimiter,
        executor: ThreadPoolExecutor,
        pbar: Optional[tqdm] = None,
        pbar_lock: Optional[threading.Lock] = None,
    ) -> TypeGenerationResult:
        """Async wrapper for generating queries for a single type.

        Args:
            type_name: Name of the query type
            extra_instruction: Type-specific instruction
            schema: Database schema string
            table_data: Dictionary of table data
            column_stats: Column statistics for smart data formatting
            sample_values: Sample values from database for query generation
            semaphore: Semaphore for concurrency control
            rate_limiter: Rate limiter for API calls
            executor: Thread pool for blocking calls
            pbar: Optional progress bar
            pbar_lock: Optional lock for thread-safe progress bar updates

        Returns:
            TypeGenerationResult with generated queries
        """
        async with semaphore:
            await rate_limiter.acquire()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                executor,
                self._generate_queries_for_type,
                type_name,
                extra_instruction,
                schema,
                table_data,
                column_stats,
                sample_values,
                schema_prompt,
                value_hints,
                pbar,
                pbar_lock,
            )
            return result

    async def _generate_all_types_parallel(
        self,
        schema: str,
        table_data: Dict[str, List[Dict[str, Any]]],
        column_stats: Optional[Dict[str, Dict[str, int]]] = None,
        sample_values: Optional[Dict[str, Dict[str, Any]]] = None,
        schema_prompt: Optional[str] = None,
        value_hints: Optional[str] = None,
    ) -> List[TypeGenerationResult]:
        """Generate queries for all types in parallel.

        Args:
            schema: Database schema string
            table_data: Dictionary of table data
            column_stats: Optional column statistics for smart data formatting
            sample_values: Optional sample values for query generation

        Returns:
            List of TypeGenerationResult for all types
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        rate_limiter = AsyncRateLimiter(rate=self.requests_per_second)
        executor = ThreadPoolExecutor(max_workers=self.max_concurrency)
        pbar_lock = threading.Lock()  # Thread-safe progress bar updates

        target_queries = len(QUERY_TYPES) * self.num_per_type
        logger.info(
            f"Starting parallel query generation: {len(QUERY_TYPES)} types × "
            f"{self.num_per_type} queries = {target_queries} target"
        )
        logger.info(
            f"Parallelization: max_concurrency={self.max_concurrency}, "
            f"rate_limit={self.requests_per_second} req/s"
        )
        logger.info("Note: Actual count may be lower if query types fail validation")

        # Create progress bar - tracks successful queries (may be less than target)
        pbar = tqdm(total=target_queries, desc="Generating SQL queries (target)")

        try:
            tasks = [
                self._generate_queries_for_type_async(
                    type_name=type_name,
                    extra_instruction=extra_instruction,
                    schema=schema,
                    table_data=table_data,
                    column_stats=column_stats,
                    sample_values=sample_values,
                    schema_prompt=schema_prompt,
                    value_hints=value_hints,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    executor=executor,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                )
                for type_name, extra_instruction in QUERY_TYPES
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            valid_results: List[TypeGenerationResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    type_name = QUERY_TYPES[i][0]
                    logger.error(
                        f"[{type_name}] Generation failed with error: {result}"
                    )
                else:
                    valid_results.append(result)

            # Update progress bar to show actual completion
            actual_count = sum(r.success_count for r in valid_results)
            pbar.n = actual_count  # Set to actual count
            pbar.refresh()

            return valid_results

        finally:
            pbar.close()
            executor.shutdown(wait=False)

    def _deduplicate_queries(
        self, results: List[TypeGenerationResult]
    ) -> Tuple[List[str], List[Dict[str, Any]], int]:
        """Merge and deduplicate queries from all types.

        Args:
            results: List of TypeGenerationResult from all types

        Returns:
            Tuple of (deduplicated queries list, metadata list, duplicates removed)
        """
        seen: set = set()
        deduplicated: List[str] = []
        deduplicated_meta: List[Dict[str, Any]] = []
        duplicates_removed = 0

        for result in results:
            meta_list = result.metadata or []
            for idx, query in enumerate(result.queries):
                # Normalize whitespace for comparison
                normalized = " ".join(query.split())
                if normalized not in seen:
                    seen.add(normalized)
                    deduplicated.append(query)
                    if idx < len(meta_list):
                        deduplicated_meta.append(meta_list[idx])
                    else:
                        deduplicated_meta.append(
                            self._build_query_metadata(
                                sql=query,
                                query_type=result.type_name,
                                template_source="unknown",
                            )
                        )
                else:
                    duplicates_removed += 1
                    logger.debug(
                        f"Removed cross-type duplicate from {result.type_name}"
                    )

        return deduplicated, deduplicated_meta, duplicates_removed

    def __call__(
        self,
        schema: str,
        insert_queries: Optional[List[str]] = None,
        overwrite: bool = False,
        return_metadata: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[Dict[str, Any]]]]:
        """Generate SQL queries with parallel type generation.

        Args:
            schema: Database schema string
            insert_queries: Deprecated, not used
            overwrite: If True, regenerate even if cache exists

        Returns:
            List of generated SQL queries, optionally paired with metadata records.
        """
        # Get full file path
        file_path: str = os.path.join(
            self.global_cfg.project_path,
            self.global_cfg.paths.data_dir,
            self.global_cfg.paths.sql_queries,
        )

        # Try to load from cache if not overwriting
        if not overwrite:
            cached_queries = self._load_from_cache(file_path)
            if cached_queries:
                logger.info("Using cached queries (set overwrite=True to regenerate)")
                metadata_path = self._metadata_cache_path()
                cached_meta = self._load_metadata_cache(metadata_path)
                self._latest_metadata = cached_meta
                if return_metadata:
                    return cached_queries, (cached_meta or [])
                return cached_queries

        # Generate new queries if cache is empty or overwrite is True
        logger.info("Generating new queries with parallel type processing...")

        table_names = extract_table_names_from_schema_str(schema)

        # Load sampled prompt context from database
        table_data: Dict[str, List[Dict[str, Any]]] = self._get_prompt_table_data(schema)
        logger.info(f"Loaded data from {len(table_data)} tables")

        if self.use_database_profile_queries:
            column_stats = get_all_column_statistics(self.pg_client, table_names)
            logger.info(f"Collected column statistics for {len(column_stats)} tables")

            sample_values = get_column_sample_values(self.pg_client, table_names)
            logger.info(f"Collected sample values for {len(sample_values)} tables")
        else:
            column_stats = estimate_column_statistics_from_rows(table_data)
            logger.info(
                f"Estimated column statistics from sampled rows for {len(column_stats)} tables"
            )

            sample_values = extract_sample_values_from_rows(table_data)
            logger.info(
                f"Derived sample values from sampled rows for {len(sample_values)} tables"
            )

        # Get valid JOIN relationships to help with JOIN queries
        join_pairs = get_valid_join_pairs(self.pg_client, table_names)
        logger.info(f"Collected {len(join_pairs)} valid join relationships")

        schema_prompt_text = format_schema_with_columns(
            schema,
            max_columns_per_table=self.max_columns_per_table,
        )

        # Combine sample values and join hints for the prompt
        value_hints_text = format_sample_values_for_prompt(
            sample_values, column_stats=column_stats
        )
        join_hints_text = format_join_hints_for_prompt(join_pairs)
        if join_hints_text:
            value_hints_text = f"{value_hints_text}\n\n{join_hints_text}"

        # Run parallel generation
        results = asyncio.run(
            self._generate_all_types_parallel(
                schema=schema,
                table_data=table_data,
                column_stats=column_stats,
                sample_values=sample_values,
                schema_prompt=schema_prompt_text,
                value_hints=value_hints_text,
            )
        )

        # Merge and deduplicate
        generated_queries, metadata_records, duplicates_removed = self._deduplicate_queries(
            results
        )
        self._latest_metadata = metadata_records

        # Calculate statistics
        total_success = sum(r.success_count for r in results)
        total_attempts = sum(r.total_attempts for r in results)

        logger.info("=" * 60)
        logger.info("Query Generation Summary")
        logger.info("=" * 60)
        logger.info(f"Types processed:     {len(results)}/{len(QUERY_TYPES)}")
        logger.info(f"Successful queries:  {total_success}")
        logger.info(f"Total attempts:      {total_attempts}")
        logger.info(f"Duplicates removed:  {duplicates_removed}")
        logger.info(f"Final query count:   {len(generated_queries)}")
        logger.info("=" * 60)

        # Per-type breakdown
        for result in results:
            logger.info(
                f"  [{result.type_name}] {result.success_count}/{self.num_per_type} queries"
            )

        # Save the queries to a file
        logger.info(f"Saving {len(generated_queries)} queries to {file_path}...")
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as f:
            for query in generated_queries:
                f.write(query + self.global_cfg.paths.separator + "\n")

        if return_metadata:
            return generated_queries, metadata_records
        return generated_queries

    @staticmethod
    def _update_progress_bar(
        pbar: Optional[tqdm], pbar_lock: Optional[threading.Lock], amount: int
    ) -> None:
        if not pbar or amount <= 0:
            return
        if pbar_lock:
            with pbar_lock:
                pbar.update(amount)
        else:
            pbar.update(amount)

    def _record_failed_query(
        self, type_name: str, sql: str, reason: Optional[str]
    ) -> None:
        try:
            failure_dir = self.template_debug_dir / "execution_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            file_path = failure_dir / f"{type_name}.sql"
            with file_path.open("a") as fp:
                comment = f"-- ERROR: {reason}\n" if reason else ""
                fp.write(comment + sql.strip() + "\n\n")
        except Exception:
            logger.exception("Failed to record execution failure for [%s]", type_name)

    def _validate_sql_syntax(self, sql_query: str) -> Optional[str]:
        """Check for obviously invalid SQL patterns.

        Returns:
            Error message if invalid, None if OK.
        """
        import re as _re

        normalized = " ".join(sql_query.split()).upper()

        # Check for placeholder-as-identifier patterns (e.g., WHERE 1 ILIKE 1)
        if _re.search(
            r"\bWHERE\s+\d+\s+(?:ILIKE|LIKE|=|>=|<=|<>|!=|>|<)\s+\d+", normalized
        ):
            return "Invalid WHERE clause: numeric literal used as column"

        # Check for consecutive numeric literals (e.g., WHERE 1 1 1, ORDER BY 1 1)
        if _re.search(r"\b\d+\s+\d+\s+\d+\b", normalized):
            return "Invalid SQL: consecutive numeric literals"

        # Check for duplicate ORDER BY values (e.g., ORDER BY 1 1)
        if _re.search(r"\bORDER\s+BY\s+(\d+)\s+\1\b", normalized):
            return "Invalid ORDER BY: duplicate position"
        if _re.search(r"\bORDER\s+BY\s+\d+\s+\d+(?!\s*,)", normalized):
            return "Invalid ORDER BY: consecutive positions without comma"

        # Check for unfilled placeholders
        if _re.search(r"\{[A-Za-z0-9_]+\}", sql_query):
            return "Unfilled placeholder in SQL"

        return None

    def _execute_and_validate_query(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """Execute a SQL query and check if it returns non-empty results.

        Args:
            sql_query: SQL query to execute

        Returns:
            Tuple of (success, error_message). error_message is None on success.
        """
        try:
            # Skip queries that are likely to be problematic
            if not sql_query.strip() or sql_query.strip().lower().startswith(
                ("drop", "delete", "truncate", "alter")
            ):
                logger.warning(
                    f"Skipping potentially destructive query: {sql_query[:100]}..."
                )
                return False, "Query contains potentially destructive operations"

            # Pre-flight syntax check
            syntax_error = self._validate_sql_syntax(sql_query)
            if syntax_error:
                logger.debug("Syntax validation failed: %s", syntax_error)
                return False, syntax_error

            results = self.pg_client.execute_and_fetchall_with_col_names(sql_query)
            # Check if results are not empty
            is_valid = len(results) > 0
            if is_valid:
                logger.debug(f"Query returned {len(results)} rows")
                return True, None
            else:
                logger.debug(f"Query returned empty results")
                return False, "Query returned empty results"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing query: {error_msg}")
            logger.error(f"Query: {sql_query}")
            return False, error_msg
