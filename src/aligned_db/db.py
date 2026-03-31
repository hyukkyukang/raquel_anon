"""Aligned database construction and management.

This module provides the main class for creating and populating a PostgreSQL database
schema based on question-answer pairs loaded from datasets.
"""

from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import hkkang_utils.pg as pg_utils
import psycopg
from omegaconf import DictConfig
from psycopg import sql

from src.aligned_db.schema_execution import (
    add_unique_constraints,
    correct_schema_sql,
    execute_schema_statements,
    needs_schema_context,
    strip_foreign_keys_from_statement,
)
from src.aligned_db.build_flow import (
    load_cached_build_artifacts,
    prepare_build_artifacts,
    summarize_verification_results,
)
from src.aligned_db.persistence import (
    save_build_results,
    save_upserts_log,
    verify_data_insertion,
)
from src.aligned_db.entity_upserts import generate_upserts_from_entities
from src.aligned_db.cleanup import (
    get_existing_tables,
    get_reverse_dependency_order,
    remove_empty_tables,
    remove_null_only_columns,
)
from src.aligned_db.extraction_cleanup import prune_qa_extractions_to_aligned_db
from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import ForeignKeyConstraint, SchemaRegistry
from src.aligned_db.verification_flow import (
    execute_fixes,
    run_iterative_verification,
)
from src.aligned_db.upsert_execution import execute_upserts
from src.aligned_db.upsert_runtime import AlignedDBUpsertRuntime
from src.llm.tracker import llm_call_tracker
from src.utils.checkpoint import CheckpointManager
from src.utils.text_normalizer import normalize_qa_pair

if TYPE_CHECKING:
    from src.generator.entity_pipeline import AlignedDBPipeline
    from src.generator.round_trip_verifier import VerificationResult
    from src.generator.sql import SQLGenerator

logger = logging.getLogger("AlignedDB")


class AlignedDB:
    """A class to manage and build an aligned database for question-answer pairs.

    This class handles the creation and population of a PostgreSQL database schema
    based on question-answer pairs. It manages database connections, schema generation,
    and data upsertion using the 6-stage AlignedDBPipeline.

    Attributes:
        global_cfg (DictConfig): Global configuration object containing database and model settings
    """

    def __init__(self, global_cfg: DictConfig) -> None:
        """Initialize the AlignedDB instance.

        Args:
            global_cfg (DictConfig): Configuration object containing database credentials
                and model settings
        """
        self.global_cfg = global_cfg
        logger.info(
            f"\nAlignedDB initialized\n"
            f"  Database: {global_cfg.database.db_id}@{global_cfg.database.host}:{global_cfg.database.port}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def save_dir_path(self) -> str:
        """Directory path where schemas will be saved/logged."""
        return os.path.join(
            self.global_cfg.project_path,
            self.global_cfg.model.dir_path,
        )

    @property
    def upsert_log_dir_path(self) -> str:
        """Directory path where the upsert log will be saved."""
        return os.path.join(
            self.save_dir_path,
            "log",
        )

    @property
    def upsert_save_path(self) -> str:
        """Full path where all the upsert statements will be saved."""
        return os.path.join(
            self.save_dir_path,
            "upsert.sql",
        )

    # =========================================================================
    # Cached Properties - Component Instances
    # =========================================================================

    @cached_property
    def pg_client(self) -> pg_utils.PostgresConnector:
        """Get or create a PostgreSQL database connection.

        Returns:
            pg_utils.PostgresConnector: A cached PostgreSQL connection instance
        """
        return pg_utils.PostgresConnector(
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.port,
            db_id=self.global_cfg.database.db_id,
        )

    @cached_property
    def _checkpoint_manager(self) -> CheckpointManager:
        """Get checkpoint manager instance for resumable builds.

        Returns:
            CheckpointManager: A cached CheckpointManager instance
        """
        return CheckpointManager(os.path.join(self.save_dir_path, "checkpoints"))

    @cached_property
    def aligned_db_pipeline(self) -> AlignedDBPipeline:
        """Get the AlignedDBPipeline (6-stage pipeline).

        Note: pg_client is NOT passed during run() because tables don't exist yet.
        It's only used for verification after DB is populated.

        Returns:
            AlignedDBPipeline: A cached AlignedDBPipeline instance
        """
        from src.generator.entity_pipeline import AlignedDBPipeline

        return AlignedDBPipeline(
            api_cfg=self.global_cfg.llm,
            global_cfg=self.global_cfg,
            pg_client=None,  # Don't pass pg_client - tables don't exist during run()
        )

    @cached_property
    def upsert_runtime(self) -> AlignedDBUpsertRuntime:
        """Get shared runtime services for entity and junction upsert generation."""
        return AlignedDBUpsertRuntime(pg_client=self.pg_client)

    @cached_property
    def sql_generator(self) -> SQLGenerator:
        """Get SQLGenerator for SQL syntax correction.

        Returns:
            SQLGenerator: A cached SQLGenerator instance
        """
        from src.generator.sql import SQLGenerator

        return SQLGenerator(self.global_cfg.llm, self.global_cfg)

    # =========================================================================
    # Database Setup Methods
    # =========================================================================

    def _reset_database(self) -> None:
        """Drop and recreate the public schema."""
        logger.info("Resetting database: dropping and recreating public schema...")
        self.pg_client.execute("DROP SCHEMA public CASCADE;")
        self.pg_client.execute("CREATE SCHEMA public;")
        logger.info("  Dropped and recreated public schema")

    def _ensure_database_exists(self) -> None:
        """Create the target database when running against a fresh DB name."""
        try:
            with psycopg.connect(
                (
                    f"user={self.global_cfg.database.user_id} "
                    f"password={self.global_cfg.database.passwd} "
                    f"host={self.global_cfg.database.host} "
                    f"port={self.global_cfg.database.port} "
                    f"dbname={self.global_cfg.database.db_id}"
                )
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            if "does not exist" not in str(exc):
                raise

        logger.info(
            "Target database %s does not exist; creating it.",
            self.global_cfg.database.db_id,
        )
        with psycopg.connect(
            (
                f"user={self.global_cfg.database.user_id} "
                f"password={self.global_cfg.database.passwd} "
                f"host={self.global_cfg.database.host} "
                f"port={self.global_cfg.database.port} "
                "dbname=postgres"
            ),
            autocommit=True,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.global_cfg.database.db_id)
                    )
                )

    def _strip_foreign_keys_from_statement(
        self, stmt: str
    ) -> Tuple[str, str, List[ForeignKeyConstraint]]:
        """Strip foreign key constraints from a CREATE TABLE statement."""
        return strip_foreign_keys_from_statement(stmt)

    def _execute_schema_statements(
        self, schema: List[str], max_correction_retries: int = 2
    ) -> None:
        """Execute CREATE TABLE statements using the shared schema executor."""
        execute_schema_statements(
            pg_client=self.pg_client,
            sql_generator=self.sql_generator,
            schema=schema,
            max_correction_retries=max_correction_retries,
            clear_lookup_column_cache=self._clear_lookup_column_cache,
        )

    def _clear_lookup_column_cache(self) -> None:
        """Clear the cached lookup column mappings.

        Should be called after schema changes to ensure fresh lookups.
        """
        self.upsert_runtime.clear_lookup_column_cache()

    def _needs_schema_context(self, error_message: str) -> bool:
        """Determine if an error would benefit from schema context."""
        return needs_schema_context(error_message)

    def _correct_schema_sql(
        self,
        sql: str,
        error_message: str,
        full_schema: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Attempt to correct a schema SQL statement using the shared helper."""
        return correct_schema_sql(
            sql_generator=self.sql_generator,
            sql=sql,
            error_message=error_message,
            full_schema=full_schema,
        )

    def _add_unique_constraints(self, schema_registry: SchemaRegistry) -> None:
        """Add UNIQUE constraints using the shared schema executor."""
        add_unique_constraints(
            pg_client=self.pg_client,
            schema_registry=schema_registry,
        )

    def _save_qa_pairs(
        self,
        canonical_qa_pairs: List[Tuple[str, str]],
        *,
        normalized_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        naturalized_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        extraction_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        qa_pair_records: Optional[List[Dict[str, Any]]] = None,
        qa_pair_normalization_summary: Optional[Dict[str, Any]] = None,
        qa_pair_naturalization_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save QA pairs to a file.

        Args:
            canonical_qa_pairs: Canonical question-answer pairs to save
        """
        qa_pairs = canonical_qa_pairs
        qa_pairs_path: str = os.path.join(self.save_dir_path, "qa_pairs.jsonl")
        os.makedirs(os.path.dirname(qa_pairs_path), exist_ok=True)
        with open(qa_pairs_path, "w") as f:
            f.write(json.dumps(qa_pairs, indent=4))
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {qa_pairs_path}")

        qa_pairs_canonical_path = os.path.join(
            self.save_dir_path,
            "qa_pairs_canonical.jsonl",
        )
        with open(qa_pairs_canonical_path, "w") as f:
            f.write(json.dumps(canonical_qa_pairs, indent=4))
        logger.info(
            "Saved %d canonical QA pairs to %s",
            len(canonical_qa_pairs),
            qa_pairs_canonical_path,
        )

        normalized_pairs = normalized_qa_pairs or canonical_qa_pairs
        qa_pairs_normalized_path = os.path.join(
            self.save_dir_path,
            "qa_pairs_normalized.jsonl",
        )
        with open(qa_pairs_normalized_path, "w") as f:
            f.write(json.dumps(normalized_pairs, indent=4))
        logger.info(
            "Saved %d normalized QA pairs to %s",
            len(normalized_pairs),
            qa_pairs_normalized_path,
        )

        naturalized_pairs = naturalized_qa_pairs or normalized_pairs
        qa_pairs_naturalized_path = os.path.join(
            self.save_dir_path,
            "qa_pairs_naturalized.jsonl",
        )
        with open(qa_pairs_naturalized_path, "w") as f:
            f.write(json.dumps(naturalized_pairs, indent=4))
        logger.info(
            "Saved %d naturalized QA pairs to %s",
            len(naturalized_pairs),
            qa_pairs_naturalized_path,
        )

        if extraction_qa_pairs is not None:
            qa_pairs_extraction_path = os.path.join(
                self.save_dir_path,
                "qa_pairs_extraction.jsonl",
            )
            with open(qa_pairs_extraction_path, "w") as f:
                f.write(json.dumps(extraction_qa_pairs, indent=4))
            logger.info(
                "Saved %d extraction QA pairs to %s",
                len(extraction_qa_pairs),
                qa_pairs_extraction_path,
            )
        else:
            qa_pairs_extraction_path = os.path.join(
                self.save_dir_path,
                "qa_pairs_extraction.jsonl",
            )
            with open(qa_pairs_extraction_path, "w") as f:
                f.write(json.dumps(naturalized_pairs, indent=4))
            logger.info(
                "Saved %d extraction QA pairs to %s",
                len(naturalized_pairs),
                qa_pairs_extraction_path,
            )

        if qa_pair_records is not None:
            qa_pairs_original_path = os.path.join(
                self.save_dir_path,
                "qa_pairs_original.jsonl",
            )
            original_pairs = [
                [
                    str(record.get("original_question", "")),
                    str(record.get("original_answer", "")),
                ]
                for record in qa_pair_records
            ]
            with open(qa_pairs_original_path, "w") as f:
                f.write(json.dumps(original_pairs, indent=4))
            logger.info(
                "Saved %d original QA pairs to %s",
                len(original_pairs),
                qa_pairs_original_path,
            )

            qa_pair_metadata_path = os.path.join(
                self.save_dir_path,
                "qa_pairs_metadata.json",
            )
            with open(qa_pair_metadata_path, "w") as f:
                json.dump(
                    {
                        "normalization_summary": qa_pair_normalization_summary or {},
                        "naturalization_summary": qa_pair_naturalization_summary or {},
                        "records": qa_pair_records,
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved QA normalization metadata to %s", qa_pair_metadata_path)

        qa_pair_naturalization_path = os.path.join(
            self.save_dir_path,
            "qa_pairs_naturalization_summary.json",
        )
        with open(qa_pair_naturalization_path, "w") as f:
            json.dump(qa_pair_naturalization_summary or {}, f, indent=2)
        logger.info(
            "Saved QA naturalization summary to %s",
            qa_pair_naturalization_path,
        )

    # =========================================================================
    # Main Build Method
    # =========================================================================

    def build(
        self,
        qa_pairs: List[Tuple[str, str]],
        qa_sources: Optional[List[str]] = None,
        discovery_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        normalized_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        naturalized_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        extraction_qa_pairs: Optional[List[Tuple[str, str]]] = None,
        qa_pair_records: Optional[List[Dict[str, Any]]] = None,
        qa_pair_normalization_summary: Optional[Dict[str, Any]] = None,
        qa_pair_naturalization_summary: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        max_retries: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """Build aligned database from QA pairs using the 6-stage pipeline.

        This method runs the complete AlignedDBPipeline:
        1. Stage 1-2: Entity type and attribute discovery
        2. Stage 3: Schema generation
        3. Stage 4: Per-QA value extraction
        4. Stage 5: Deduplication
        5. Stage 6: Verification (after DB insert)

        Args:
            qa_pairs: List of question-answer pairs
            qa_sources: Optional list of source labels ("retain" or "forget") per QA pair.
                       Used for nullification to identify which entities to remove.
            overwrite: Whether to force rebuild even if already completed.
                      If False and build was previously completed, returns cached results.
            max_retries: Maximum number of retries for upsert execution

        Returns:
            Tuple[List[str], List[str]]: Generated schema and upsert statements
        """
        # Check if build was already completed
        verification_summary_path = os.path.join(
            self.save_dir_path, "verification_summary.json"
        )
        schema_path = os.path.join(self.save_dir_path, "schema.sql")

        if not overwrite and os.path.exists(verification_summary_path):
            cached_artifacts = load_cached_build_artifacts(self.save_dir_path)
            logger.info(
                f"\n{'#' * 60}\n"
                f"# ALIGNED DATABASE BUILD - USING CACHED RESULTS\n"
                f"{'#' * 60}\n"
                f"Build was previously completed. Use overwrite=True to rebuild.\n"
                f"Loading cached results from: {self.save_dir_path}"
            )
            logger.info(
                "  Loaded %d schema statements",
                len(cached_artifacts.schema_sql),
            )
            logger.info(
                "  Previous build stats: %s QA pairs verified, avg similarity: %.3f",
                cached_artifacts.verification_summary.get("total", 0),
                cached_artifacts.verification_summary.get("avg_similarity", 0),
            )

            logger.info(
                f"{'#' * 60}\n"
                f"# ALIGNED DATABASE BUILD - SKIPPED (CACHED)\n"
                f"{'#' * 60}"
            )

            # Return cached results (empty upserts since we're not rebuilding)
            return cached_artifacts.schema_sql, []

        # Reset LLM call tracker at start of pipeline
        llm_call_tracker.reset()

        logger.info(
            f"\n{'#' * 60}\n"
            f"# ALIGNED DATABASE BUILD STARTED\n"
            f"{'#' * 60}\n"
            f"Input: {len(qa_pairs)} QA pairs\n"
            f"Output directory: {self.save_dir_path}"
        )

        canonical_qa_pairs = [normalize_qa_pair(q, a) for q, a in qa_pairs]
        selected_discovery_qa_pairs = discovery_qa_pairs or canonical_qa_pairs
        selected_extraction_qa_pairs = (
            extraction_qa_pairs
            or naturalized_qa_pairs
            or normalized_qa_pairs
            or selected_discovery_qa_pairs
        )

        prepared_artifacts = prepare_build_artifacts(
            qa_pairs=selected_discovery_qa_pairs,
            canonical_qa_pairs=canonical_qa_pairs,
            normalized_qa_pairs=normalized_qa_pairs or canonical_qa_pairs,
            naturalized_qa_pairs=naturalized_qa_pairs,
            extraction_qa_pairs=selected_extraction_qa_pairs,
            qa_sources=qa_sources,
            aligned_db_pipeline=self.aligned_db_pipeline,
            save_qa_pairs_fn=self._save_qa_pairs,
            qa_pair_records=qa_pair_records,
            qa_pair_normalization_summary=qa_pair_normalization_summary,
            qa_pair_naturalization_summary=qa_pair_naturalization_summary,
        )
        schema_registry = prepared_artifacts.schema_registry
        entity_registry = prepared_artifacts.entity_registry
        qa_extractions = prepared_artifacts.qa_extractions
        schema_sql = prepared_artifacts.schema_sql
        total_entities = prepared_artifacts.total_entities

        # Reset and setup database
        self._ensure_database_exists()
        logger.info("Setting up database with generated schema...")
        self._reset_database()
        self._execute_schema_statements(schema_sql)

        # Add UNIQUE constraints for UPSERT conflict resolution
        self._add_unique_constraints(schema_registry)

        # Generate and execute upserts from entity registry
        logger.info("Generating upsert statements from extracted entities...")
        upserts = self._generate_upserts_from_entities(schema_registry, entity_registry)
        
        # Save upserts for debugging
        self._save_upserts_log(upserts, entity_registry)
        
        logger.info("Executing upsert statements...")
        self._execute_upserts(upserts, max_retries)
        
        # Verify data was actually inserted
        self._verify_data_insertion(entity_registry)

        # Run round-trip verification (Stage 6, after data is in the database)
        logger.info("Phase 6: Running round-trip verification...")
        verification_results, total_fixes_applied = self._run_iterative_verification(
            qa_pairs=canonical_qa_pairs,
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            qa_extractions=qa_extractions,
        )

        # Phase 7: Remove empty tables (cleanup)
        tables_removed: int = 0
        removed_table_names: List[str] = []
        if self.global_cfg.model.aligned_db.get("remove_empty_tables", True):
            logger.info("Phase 7: Removing empty tables...")
            tables_removed, removed_table_names = self._remove_empty_tables(
                schema_registry
            )

        # Phase 8: Remove null-only columns (cleanup)
        columns_removed: int = 0
        removed_columns_map: Dict[str, List[str]] = {}
        if self.global_cfg.model.aligned_db.get("remove_null_only_columns", True):
            logger.info("Phase 8: Removing null-only columns...")
            columns_removed, removed_columns_map = self._remove_null_only_columns(
                schema_registry
            )

        # Save results (schema will reflect removed tables and columns)
        logger.info("Saving pipeline results...")
        self._save_results(schema_registry, entity_registry, qa_extractions, verification_results)

        # Log LLM call statistics summary
        llm_call_tracker.log_summary(title="AlignedDBPipeline LLM Statistics")

        # Get final schema SQL (after modifications) for consistent return value
        final_schema_sql: List[str] = schema_registry.to_sql_list()

        # Final summary
        verification_summary = summarize_verification_results(verification_results)
        final_table_count = len(schema_registry.get_table_names())
        logger.info(
            f"\n{'=' * 60}\n"
            f"AlignedDB build completed\n"
            f"  Tables created: {len(schema_sql)}\n"
            f"  Empty tables removed: {tables_removed}\n"
            f"  Null-only columns removed: {columns_removed}\n"
            f"  Final tables: {final_table_count}\n"
            f"  Entities inserted: {total_entities}\n"
            f"  Upserts executed: {len(upserts)}\n"
            f"  Verification: {len(verification_results)} pairs checked\n"
            f"    - Passed: {verification_summary.passed_count}\n"
            f"    - Needs fix: {verification_summary.needs_fix_count}\n"
            f"    - Inconsistent QA: {verification_summary.inconsistent_count}\n"
            f"  Total fixes applied: {total_fixes_applied}\n"
            f"  Average similarity score: {verification_summary.avg_similarity:.3f}\n"
            f"{'=' * 60}"
        )

        logger.info(
            f"{'#' * 60}\n" f"# ALIGNED DATABASE BUILD COMPLETED\n" f"{'#' * 60}"
        )

        # Return final schema (consistent with saved schema.sql)
        return final_schema_sql, upserts

    # =========================================================================
    # Upsert Generation Methods
    # =========================================================================

    def _generate_upserts_from_entities(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
    ) -> List[str]:
        """Generate SQL upsert statements from extracted entities and relationships.

        This method generates upserts in two phases:
        1. Entity upserts (for regular tables, in topological order for FK deps)
        2. Junction table upserts (for many-to-many relationships)

        Args:
            schema_registry: Database schema
            entity_registry: Extracted entities and relationships

        Returns:
            List of SQL INSERT/UPSERT statements (entities first, then junction tables)
        """
        return generate_upserts_from_entities(
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            runtime=self.upsert_runtime,
            enable_dynamic_junction=self.global_cfg.model.aligned_db.get(
                "enable_dynamic_junction_creation", True
            ),
            max_junction_fix_iterations=self.global_cfg.model.aligned_db.get(
                "max_junction_fix_iterations", 2
            ),
        )

    # =========================================================================
    # Upsert Debugging Methods
    # =========================================================================

    def _save_upserts_log(
        self,
        upserts: List[str],
        entity_registry: EntityRegistry,
    ) -> None:
        """Save generated upsert statements to a log file for debugging."""
        save_upserts_log(
            upserts=upserts,
            entity_registry=entity_registry,
            upsert_log_dir_path=self.upsert_log_dir_path,
            grounding_summary=self.upsert_runtime.grounding_summary,
        )

    def _verify_data_insertion(self, entity_registry: EntityRegistry) -> None:
        """Verify that data was actually inserted into the database."""
        verify_data_insertion(
            pg_client=self.pg_client,
            entity_registry=entity_registry,
        )

    # =========================================================================
    # Upsert Execution Methods
    # =========================================================================

    def _execute_upserts(self, upserts: List[str], max_retries: int = 5) -> None:
        """Execute upsert statements with retry logic.

        Args:
            upserts: List of SQL upsert statements
            max_retries: Maximum number of retries per statement
        """
        execute_upserts(
            pg_client=self.pg_client,
            upserts=upserts,
            max_retries=max_retries,
        )

    def _execute_fixes(self, fixes: List[str]) -> int:
        """Execute fix statements from round-trip verification."""
        return execute_fixes(
            pg_client=self.pg_client,
            fixes=fixes,
        )

    # =========================================================================
    # Verification Methods
    # =========================================================================

    def _run_iterative_verification(
        self,
        qa_pairs: List[Tuple[str, str]],
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
        qa_extractions: Optional[QAExtractionRegistry] = None,
    ) -> Tuple[List[VerificationResult], int]:
        """Run iterative verification with fix application.

        Args:
            qa_pairs: List of (question, answer) tuples
            schema_registry: Current database schema
            entity_registry: Entity registry for re-extraction
            qa_extractions: Optional QA extractions for exact table mapping

        Returns:
            Tuple of (final_verification_results, total_fixes_applied)
        """
        del entity_registry
        return run_iterative_verification(
            verifier=self.aligned_db_pipeline.round_trip_verifier,
            pg_client=self.pg_client,
            qa_pairs=qa_pairs,
            schema_registry=schema_registry,
            qa_extractions=qa_extractions,
            enable_iterative=self.global_cfg.model.aligned_db.get(
                "enable_iterative_verification", True
            ),
            max_iterations=self.global_cfg.model.aligned_db.get(
                "verification_max_iterations", 3
            ),
            use_dedicated_prompt=self.global_cfg.model.aligned_db.get(
                "use_dedicated_fix_prompt", True
            ),
        )

    # =========================================================================
    # Table Cleanup Methods
    # =========================================================================

    def _get_existing_tables(self) -> Set[str]:
        """Query the database for tables that actually exist."""
        return get_existing_tables(pg_client=self.pg_client)

    def _remove_empty_tables(
        self,
        schema_registry: SchemaRegistry,
    ) -> Tuple[int, List[str]]:
        """Remove tables with zero rows from the database."""
        return remove_empty_tables(
            pg_client=self.pg_client,
            schema_registry=schema_registry,
        )

    def _remove_null_only_columns(
        self,
        schema_registry: SchemaRegistry,
    ) -> Tuple[int, Dict[str, List[str]]]:
        """Remove columns from tables where all values are NULL."""
        return remove_null_only_columns(
            pg_client=self.pg_client,
            schema_registry=schema_registry,
        )

    def _get_reverse_dependency_order(
        self,
        schema_registry: SchemaRegistry,
        tables_to_order: List[str],
    ) -> List[str]:
        """Get tables in reverse dependency order for safe dropping."""
        return get_reverse_dependency_order(
            schema_registry=schema_registry,
            tables_to_order=tables_to_order,
        )

    # =========================================================================
    # Result Saving Methods
    # =========================================================================

    def _save_results(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
        qa_extractions: "QAExtractionRegistry",
        verification_results: List[VerificationResult],
    ) -> None:
        """Save results from the pipeline."""
        cleaned_qa_extractions, cleanup_stats = prune_qa_extractions_to_aligned_db(
            pg_client=self.pg_client,
            schema_registry=schema_registry,
            qa_extractions=qa_extractions,
        )
        save_build_results(
            save_dir_path=self.save_dir_path,
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            qa_extractions=cleaned_qa_extractions,
            verification_results=verification_results,
            extraction_cleanup_stats=cleanup_stats,
        )
