"""Generate and execute nullify UPDATE statements.

.. deprecated::
    This module is deprecated. Use :class:`src.aligned_db.nullified_db.NullifiedDBBuilder`
    instead, which provides entity-based nullification using QAExtractionRegistry
    for more accurate forget/retain entity tracking.

This module creates UPDATE statements that null out data in the forget set
to create the null database variant used for unlearning evaluation.
"""

import warnings

warnings.warn(
    "src.aligned_db.nullify is deprecated. "
    "Use src.aligned_db.nullified_db.NullifiedDBBuilder instead. "
    "This module is kept only for backwards compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import os
from functools import cached_property
from typing import List, Tuple

import tqdm
from omegaconf import DictConfig

from src.aligned_db.cascade_delete import CascadeDeleteHandler
from src.constants import SQL_STATEMENT_SEPARATOR
from src.generator.sql import SQLGenerator
from src.utils.database import PathBuilderMixin, PostgresConnectionMixin
from src.utils.results_saver import IntermediateResultsSaver, SQLStatementSaver

logger = logging.getLogger("src.aligned_db.nullify")


class UpdateNullExecutor(PostgresConnectionMixin, PathBuilderMixin):
    """Generates and executes nullify update SQL statements from QA pairs and schema.

    This class manages the generation, modification, and execution of nullify update
    statements for question-answer pairs in an aligned database context.

    Attributes:
        cfg: Configuration object containing specific settings
        global_cfg: Global configuration object containing database and model settings
    """

    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the UpdateNullExecutor.

        Args:
            cfg: Configuration object containing specific settings
            global_cfg: Global configuration object containing database credentials
                and model settings
        """
        self.cfg: DictConfig = cfg
        PostgresConnectionMixin.__init__(self, global_cfg)
        PathBuilderMixin.__init__(self, global_cfg)

    @cached_property
    def pg_client(self):
        """Get PostgreSQL database connection to null database."""
        return self.null_pg_client

    @property
    def nullify_log_dir_path(self) -> str:
        """Directory path where the nullify log will be saved."""
        return self.build_model_path("log", "nullify")

    @property
    def nullify_save_path(self) -> str:
        """Full path where the final nullify statements will be saved."""
        return self.build_model_path("nullify.sql")

    @cached_property
    def nullify_generator(self) -> SQLGenerator:
        """Get SQL generator for nullify operations."""
        return SQLGenerator(self.global_cfg.llm, self.global_cfg)

    @cached_property
    def _results_saver(self) -> IntermediateResultsSaver:
        """Get intermediate results saver instance."""
        return IntermediateResultsSaver.from_config(self.global_cfg, "nullify")

    @cached_property
    def _sql_saver(self) -> SQLStatementSaver:
        """Get SQL statement saver instance."""
        return SQLStatementSaver(self.nullify_save_path)

    @cached_property
    def _cascade_handler(self) -> CascadeDeleteHandler:
        """Get cascade delete handler instance."""
        return CascadeDeleteHandler(self.pg_client)

    def __call__(
        self,
        schema: List[str],
        insert_statements: List[str],
        qa_pairs: List[Tuple[str, str]],
        overwrite: bool = False,
        max_retries: int = 5,
    ) -> Tuple[int, List[List[str]]]:
        """Process all QA pairs and return skip count and nullify statements.

        Args:
            schema: Database schema as list of SQL statements
            insert_statements: Insert statements containing data to be nullified
            qa_pairs: List of question-answer pairs
            overwrite: Whether to overwrite existing nullify cache
            max_retries: Maximum number of retries per QA pair

        Returns:
            Tuple of (skip_count, list of nullify statement lists per QA pair)
        """
        # Try to load from cache if available
        cached_result = self._try_load_from_cache(overwrite)
        if cached_result is not None:
            return cached_result

        # Process all QA pairs
        skip_count, all_nullify_statements = self._process_qa_pairs(
            schema, insert_statements, qa_pairs, max_retries
        )

        # Save results
        self._save_results(qa_pairs, skip_count, all_nullify_statements)

        # Clean up empty rows
        self._cascade_handler.cleanup_empty_rows()

        return skip_count, all_nullify_statements

    def _try_load_from_cache(
        self, overwrite: bool
    ) -> Tuple[int, List[List[str]]] | None:
        """Try to load nullify statements from cache.

        Args:
            overwrite: Whether to ignore cache

        Returns:
            Cached result tuple or None if cache miss
        """
        if overwrite or not self._sql_saver.exists():
            return None

        logger.info(f"Loading nullify statements from {self.nullify_save_path}")
        cached_nullify_str = self._sql_saver.load()
        logger.info(f"Loaded {len(cached_nullify_str)} nullify statement groups")

        # Execute cached statements
        for idx, nullify_statement in enumerate(
            tqdm.tqdm(cached_nullify_str, desc="Executing cached nullify statements")
        ):
            with self.pg_client.conn.transaction():
                self.pg_client.conn.cursor().execute(nullify_statement)

        logger.info(f"Executed {len(cached_nullify_str)} nullify statements to DB")
        return 0, [stmt.split("\n") for stmt in cached_nullify_str]

    def _process_qa_pairs(
        self,
        schema: List[str],
        insert_statements: List[str],
        qa_pairs: List[Tuple[str, str]],
        max_retries: int,
    ) -> Tuple[int, List[List[str]]]:
        """Process all QA pairs to generate nullify statements.

        Args:
            schema: Database schema
            insert_statements: Insert statements
            qa_pairs: List of question-answer pairs
            max_retries: Maximum retries per pair

        Returns:
            Tuple of (skip_count, all_nullify_statements)
        """
        skip_count = 0
        all_nullify_statements: List[List[str]] = []

        for idx, qa_pair in enumerate(
            tqdm.tqdm(qa_pairs, desc="Generating nullify update statements")
        ):
            success, nullify_statements = self._process_single_qa_pair(
                idx, qa_pair, schema, insert_statements, max_retries
            )

            if not success:
                skip_count += 1
                all_nullify_statements.append([])
            else:
                all_nullify_statements.append(nullify_statements)

        return skip_count, all_nullify_statements

    def _process_single_qa_pair(
        self,
        idx: int,
        qa_pair: Tuple[str, str],
        schema: List[str],
        insert_statements: List[str],
        max_retries: int,
    ) -> Tuple[bool, List[str]]:
        """Process a single QA pair.

        Args:
            idx: Index of the QA pair
            qa_pair: Question-answer pair
            schema: Database schema
            insert_statements: Insert statements
            max_retries: Maximum number of retries

        Returns:
            Tuple of (success, nullify_statements)
        """
        success, nullify_statements = self._execute_nullify_with_retry(
            schema, insert_statements, qa_pair, max_retries
        )

        # Save intermediate results
        self._results_saver.save_item(
            idx=idx,
            data={
                "question": qa_pair[0],
                "answer": qa_pair[1],
                "sql": "\n".join(nullify_statements) if success else "",
            },
            file_prefix="nullify",
        )

        return success, nullify_statements

    def _execute_nullify_with_retry(
        self,
        schema: List[str],
        insert_statements: List[str],
        qa_pair: Tuple[str, str],
        max_retries: int,
    ) -> Tuple[bool, List[str]]:
        """Execute nullify SQL with retry logic.

        Args:
            schema: Database schema
            insert_statements: Insert statements
            qa_pair: Question-answer pair
            max_retries: Maximum number of retries

        Returns:
            Tuple of (success, final_nullify_statements)
        """
        nullify_statements = self._generate_nullify_statements(
            schema, insert_statements, qa_pair
        )

        for attempt in range(1, max_retries + 1):
            sql = "\n".join(nullify_statements)
            try:
                with self.pg_client.conn.transaction():
                    self.pg_client.conn.cursor().execute(sql)
                return True, nullify_statements
            except Exception as e:
                nullify_statements, should_skip = self._handle_nullify_error(
                    attempt, max_retries, schema, insert_statements, qa_pair, sql, e
                )
                if should_skip:
                    return False, []

        return False, []

    def _generate_nullify_statements(
        self, schema: List[str], insert_statements: List[str], qa_pair: Tuple[str, str]
    ) -> List[str]:
        """Generate nullify update statements for a QA pair.

        Args:
            schema: Database schema
            insert_statements: Insert statements containing data to be nullified
            qa_pair: Question-answer pair

        Returns:
            List of generated nullify update statements
        """
        return self.nullify_generator.nullify_update(
            schema=schema, insert_statements=insert_statements, qa_pair=qa_pair
        )

    def _handle_nullify_error(
        self,
        attempt: int,
        max_retries: int,
        schema: List[str],
        insert_statements: List[str],
        qa_pair: Tuple[str, str],
        sql: str,
        error: Exception,
    ) -> Tuple[List[str], bool]:
        """Handle nullify execution errors with retry logic.

        Args:
            attempt: Current attempt number
            max_retries: Maximum number of retries
            schema: Database schema
            insert_statements: Insert statements
            qa_pair: Question-answer pair
            sql: SQL statement that failed
            error: Exception that occurred

        Returns:
            Tuple of (modified_statements, should_skip)
        """
        logger.error(
            f"Executing nullify statement failed (attempt: {attempt}). Error: {error}"
        )

        if attempt < max_retries:
            logger.info(f"Modifying nullify statement (attempt {attempt + 1})...")
            nullify_statements = self.nullify_generator.modify_nullify_update(
                schema=schema,
                insert_statements=insert_statements,
                qa_pair=qa_pair,
                update_statement=sql,
                error_msg=str(error),
            )

            if len(nullify_statements) == 0:
                return [], True
            return nullify_statements, False
        else:
            logger.error(f"All {max_retries} attempts failed.")
            return [], True

    def _save_results(
        self,
        qa_pairs: List[Tuple[str, str]],
        skip_count: int,
        all_nullify_statements: List[List[str]],
    ) -> None:
        """Save final nullify statements and summary.

        Args:
            qa_pairs: List of processed QA pairs
            skip_count: Number of skipped pairs
            all_nullify_statements: All generated statements
        """
        # Save statements to SQL file
        if any(all_nullify_statements):
            non_empty_count = len([s for s in all_nullify_statements if s])
            logger.info(
                f"Saving {non_empty_count} nullify statements to {self.nullify_save_path}"
            )
            os.makedirs(os.path.dirname(self.nullify_save_path), exist_ok=True)
            self._sql_saver.save(all_nullify_statements)

        # Save summary
        self._results_saver.save_summary(
            total_items=len(qa_pairs),
            successful=len(qa_pairs) - skip_count,
            skipped=skip_count,
        )
