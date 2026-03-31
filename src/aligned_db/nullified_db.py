"""Nullified database builder using entity-first approach.

This module creates the nullified aligned database by identifying entities
from the forget set and removing/nullifying them while preserving retain data.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple

import hkkang_utils.pg as pg_utils
from omegaconf import DictConfig

from src.aligned_db.cascade_delete import CascadeDeleteHandler
from src.aligned_db.nullification_flow import (
    load_cached_nullification_summary,
    prepare_nullification_artifacts,
)
from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.utils.db_operations import PostgresShellOperations

logger = logging.getLogger("NullifiedDBBuilder")


@dataclass
class NullificationResult:
    """Result of the nullification process.

    Attributes:
        entities_removed: Number of entities removed/nullified
        relations_removed: Number of junction table entries removed
        matched_entity_keys: Number of planned entity natural keys found in the null DB
        planned_entity_keys: Number of planned entity natural keys considered
        cleanup_rows_deleted: Number of rows removed by the empty-row cleanup pass
        candidate_entity_keys: Number of forget-only entity keys before baseline filtering
        skipped_absent_entity_keys: Number of forget-only entity keys skipped because
            they do not exist in the aligned DB
        retain_verified: Whether retain data integrity was verified
        tables_affected: List of tables that were modified
        errors: List of errors encountered during nullification
        row_comparison: Per-table row count comparison between aligned and null DBs
    """

    entities_removed: int = 0
    relations_removed: int = 0
    candidate_entity_keys: int = 0
    skipped_absent_entity_keys: int = 0
    matched_entity_keys: int = 0
    planned_entity_keys: int = 0
    cleanup_rows_deleted: int = 0
    retain_verified: bool = False
    tables_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    row_comparison: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities_removed": self.entities_removed,
            "relations_removed": self.relations_removed,
            "candidate_entity_keys": self.candidate_entity_keys,
            "skipped_absent_entity_keys": self.skipped_absent_entity_keys,
            "matched_entity_keys": self.matched_entity_keys,
            "planned_entity_keys": self.planned_entity_keys,
            "cleanup_rows_deleted": self.cleanup_rows_deleted,
            "retain_verified": self.retain_verified,
            "tables_affected": self.tables_affected,
            "errors": self.errors,
            "row_comparison": self.row_comparison,
        }


@dataclass(frozen=True)
class NullificationExecutionStats:
    """Direct-deletion execution summary before cleanup runs."""

    entities_removed: int
    relations_removed: int
    tables_affected: List[str]
    matched_entity_keys: int
    planned_entity_keys: int


@dataclass
class EntityIdentifier:
    """Identifies a specific entity instance in the database.

    Attributes:
        table_name: Name of the table containing the entity
        primary_key_column: Name of the primary key column
        primary_key_value: Value of the primary key
        natural_key_column: Name of the natural key column (e.g., 'name', 'title')
        natural_key_value: Value of the natural key
        source: Source label ("retain" or "forget")
    """

    table_name: str
    primary_key_column: Optional[str] = None
    primary_key_value: Optional[Any] = None
    natural_key_column: Optional[str] = None
    natural_key_value: Optional[str] = None
    source: str = "unknown"

    def __hash__(self) -> int:
        """Allow use in sets."""
        return hash((self.table_name, self.natural_key_value))

    def __eq__(self, other: object) -> bool:
        """Check equality based on table and natural key."""
        if not isinstance(other, EntityIdentifier):
            return False
        return (
            self.table_name == other.table_name
            and self.natural_key_value == other.natural_key_value
        )


class NullifiedDBBuilder:
    """Entity-based nullification using QAExtractionRegistry.

    This class creates the nullified database by:
    1. Loading saved QAExtractionRegistry and SchemaRegistry
    2. Identifying entities from forget QA pairs
    3. Computing FK cascade order
    4. Executing DELETE/UPDATE statements
    5. Verifying retain data integrity

    Attributes:
        global_cfg: Global configuration object
    """

    def __init__(self, global_cfg: DictConfig) -> None:
        """Initialize the NullifiedDBBuilder.

        Args:
            global_cfg: Global configuration object
        """
        self.global_cfg = global_cfg
        logger.info(
            f"\nNullifiedDBBuilder initialized\n"
            f"  Null Database: {global_cfg.database.null_db_id}@"
            f"{global_cfg.database.host}:{global_cfg.database.null_port}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def save_dir_path(self) -> str:
        """Directory path where aligned DB results are saved."""
        return os.path.join(
            self.global_cfg.project_path,
            self.global_cfg.model.dir_path,
        )

    @property
    def qa_extractions_path(self) -> str:
        """Path to the saved QA extractions JSON file."""
        return os.path.join(self.save_dir_path, "qa_extractions.json")

    @property
    def schema_registry_path(self) -> str:
        """Path to the saved schema registry JSON file."""
        return os.path.join(self.save_dir_path, "schema_registry.json")

    @property
    def nullify_log_dir_path(self) -> str:
        """Directory path where nullification logs will be saved."""
        return os.path.join(self.save_dir_path, "log", "nullify")

    # =========================================================================
    # Cached Properties - Database Connections
    # =========================================================================

    @cached_property
    def null_pg_client(self) -> pg_utils.PostgresConnector:
        """Get PostgreSQL connection to the null database."""
        return pg_utils.PostgresConnector(
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.null_port,
            db_id=self.global_cfg.database.null_db_id,
        )

    @cached_property
    def aligned_pg_client(self) -> pg_utils.PostgresConnector:
        """Get PostgreSQL connection to the aligned database."""
        return pg_utils.PostgresConnector(
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.port,
            db_id=self.global_cfg.database.db_id,
        )

    @cached_property
    def _cascade_handler(self) -> CascadeDeleteHandler:
        """Get cascade delete handler for FK-aware operations."""
        return CascadeDeleteHandler(self.null_pg_client)

    # =========================================================================
    # Main Build Method
    # =========================================================================

    def build(self, overwrite: bool = False) -> NullificationResult:
        """Build the nullified database by removing forget entities.

        Args:
            overwrite: Whether to force rebuild even if nullification log exists

        Returns:
            NullificationResult with statistics and status
        """
        result = NullificationResult()

        # Check if nullification was already done
        summary_path = os.path.join(self.nullify_log_dir_path, "summary.json")
        if os.path.exists(summary_path) and not overwrite:
            logger.info(
                "Nullification already completed. Use overwrite=True to re-run."
            )
            existing_result = load_cached_nullification_summary(summary_path)
            return NullificationResult(
                entities_removed=existing_result.get("entities_removed", 0),
                relations_removed=existing_result.get("relations_removed", 0),
                candidate_entity_keys=existing_result.get("candidate_entity_keys", 0),
                skipped_absent_entity_keys=existing_result.get(
                    "skipped_absent_entity_keys", 0
                ),
                matched_entity_keys=existing_result.get("matched_entity_keys", 0),
                planned_entity_keys=existing_result.get("planned_entity_keys", 0),
                cleanup_rows_deleted=existing_result.get("cleanup_rows_deleted", 0),
                retain_verified=existing_result.get("retain_verified", False),
                tables_affected=existing_result.get("tables_affected", []),
                errors=existing_result.get("errors", []),
                row_comparison=existing_result.get("row_comparison", {}),
            )

        logger.info(
            f"\n{'#' * 60}\n" f"# NULLIFIED DATABASE BUILD STARTED\n" f"{'#' * 60}"
        )

        try:
            prepared_artifacts = prepare_nullification_artifacts(
                load_qa_extractions_fn=self._load_qa_extractions,
                load_schema_registry_fn=self._load_schema_registry,
                identify_entities_by_source_fn=self._identify_entities_by_source,
                compute_entities_to_remove_fn=self._compute_entities_to_remove,
                filter_existing_entities_fn=self._filter_entities_present_in_aligned_db,
                compute_cascade_plan_fn=self._compute_cascade_plan,
            )
            if prepared_artifacts is None:
                raise FileNotFoundError(
                    "Failed to load required data files for nullification."
                )

            schema_registry = prepared_artifacts.schema_registry
            retain_entities = prepared_artifacts.retain_entities
            entities_to_remove = prepared_artifacts.entities_to_remove
            deletion_plan = prepared_artifacts.deletion_plan
            result.candidate_entity_keys = prepared_artifacts.candidate_entity_count
            result.skipped_absent_entity_keys = (
                prepared_artifacts.skipped_absent_entity_count
            )
            if prepared_artifacts.skipped_absent_entity_count > 0:
                logger.info(
                    "Nullification will ignore %d forget-only entities absent from "
                    "the aligned DB baseline",
                    prepared_artifacts.skipped_absent_entity_count,
                )

            # Step 5: Execute nullification
            logger.info("Step 5: Executing nullification...")
            self._refresh_null_database_from_aligned()
            protected_entity_ids = self._build_protected_entity_ids(
                retain_entities,
                schema_registry,
            )
            execution_stats = self._execute_nullification(
                deletion_plan,
                schema_registry,
                protected_entity_ids=protected_entity_ids,
            )
            result.entities_removed = execution_stats.entities_removed
            result.relations_removed = execution_stats.relations_removed
            result.tables_affected = execution_stats.tables_affected
            result.matched_entity_keys = execution_stats.matched_entity_keys
            result.planned_entity_keys = execution_stats.planned_entity_keys

            if (
                result.planned_entity_keys > 0
                and result.matched_entity_keys == 0
            ):
                raise RuntimeError(
                    "Nullification plan matched zero entity keys in the null "
                    "database. The null DB is likely already nullified or stale; "
                    "recreate it from the aligned DB before rerunning update_null."
                )

            # Step 6: Cleanup empty rows
            logger.info("Step 6: Cleaning up empty rows...")
            result.cleanup_rows_deleted = self._cascade_handler.cleanup_empty_rows()
            logger.info(f"  Cleaned up {result.cleanup_rows_deleted} empty rows")

            # Step 7: Verify retain data integrity
            logger.info("Step 7: Verifying retain data integrity...")
            result.retain_verified = self._verify_retain_integrity(
                retain_entities, schema_registry
            )

            # Step 8: Verify schema consistency
            logger.info("Step 8: Verifying schema consistency...")
            schema_match = self._verify_schema_consistency()
            if not schema_match:
                raise RuntimeError(
                    "Null database schema does not match aligned database schema. "
                    "This indicates a critical error in the database creation process."
                )

            # Step 9: Log row count comparison and verify
            logger.info("Step 9: Comparing row counts between databases...")
            result.row_comparison, row_count_valid = self._log_row_count_comparison(
                schema_registry
            )
            if not row_count_valid:
                raise RuntimeError(
                    "Row count verification failed: null database has unexpected row counts."
                )

            # Save nullification log
            self._save_nullification_log(result, entities_to_remove)

            # Calculate total rows removed for summary
            total_rows_removed = sum(
                c["removed"] for c in result.row_comparison.values()
            )

            logger.info(
                f"\n{'=' * 60}\n"
                f"Nullification completed\n"
                f"  Entities removed: {result.entities_removed}\n"
                f"  Relations removed: {result.relations_removed}\n"
                f"  Candidate entity keys: {result.candidate_entity_keys}\n"
                f"  Skipped absent entity keys: "
                f"{result.skipped_absent_entity_keys}\n"
                f"  Matched entity keys: {result.matched_entity_keys}/"
                f"{result.planned_entity_keys}\n"
                f"  Cleanup rows deleted: {result.cleanup_rows_deleted}\n"
                f"  Tables affected: {len(result.tables_affected)}\n"
                f"  Total rows removed: {total_rows_removed}\n"
                f"  Retain verified: {result.retain_verified}\n"
                f"  Schema verified: {schema_match}\n"
                f"{'=' * 60}"
            )

        except Exception as e:
            logger.error(f"Nullification failed: {e}")
            raise

        return result

    def _reset_cached_null_db_handles(self) -> None:
        """Drop cached null-DB handles after recreating the target database."""
        cached_client = self.__dict__.pop("null_pg_client", None)
        if cached_client is not None:
            try:
                cached_client.conn.close()
            except Exception:
                pass
        self.__dict__.pop("_cascade_handler", None)

    def _refresh_null_database_from_aligned(self) -> None:
        """Recreate the null DB from the current aligned DB contents."""
        logger.info(
            "  Refreshing null DB '%s' from aligned DB '%s'...",
            self.global_cfg.database.null_db_id,
            self.global_cfg.database.db_id,
        )
        db_ops = PostgresShellOperations(self.global_cfg)
        self._reset_cached_null_db_handles()

        with tempfile.TemporaryDirectory(prefix="raquel_null_clone_") as temp_dir:
            dump_path = os.path.join(temp_dir, "aligned_dump.sql")
            db_ops.dump_database(
                db_name=self.global_cfg.database.db_id,
                output_path=dump_path,
                host=self.global_cfg.database.host,
                port=self.global_cfg.database.port,
                data_only=False,
                use_inserts=False,
            )
            db_ops.drop_database(
                db_name=self.global_cfg.database.null_db_id,
                host=self.global_cfg.database.host,
                port=self.global_cfg.database.null_port,
                if_exists=True,
            )
            db_ops.create_database(
                db_name=self.global_cfg.database.null_db_id,
                host=self.global_cfg.database.host,
                port=self.global_cfg.database.null_port,
            )
            db_ops.execute_sql_file(
                db_name=self.global_cfg.database.null_db_id,
                file_path=dump_path,
                host=self.global_cfg.database.host,
                port=self.global_cfg.database.null_port,
            )

    # =========================================================================
    # Data Loading Methods
    # =========================================================================

    def _load_qa_extractions(self) -> Optional[QAExtractionRegistry]:
        """Load QA extractions from saved JSON file.

        Returns:
            QAExtractionRegistry or None if file not found
        """
        if not os.path.exists(self.qa_extractions_path):
            logger.error(f"QA extractions file not found: {self.qa_extractions_path}")
            return None

        with open(self.qa_extractions_path, "r") as f:
            data = json.load(f)

        registry = QAExtractionRegistry.from_dict(data)
        logger.info(f"  Loaded {registry.count} QA extractions")

        # Log source distribution
        forget_count = len(registry.get_forget_extractions())
        retain_count = len(registry.get_retain_extractions())
        logger.info(f"  Sources: {retain_count} retain, {forget_count} forget")

        return registry

    def _load_schema_registry(self) -> Optional[SchemaRegistry]:
        """Load schema registry from saved JSON file.

        Returns:
            SchemaRegistry or None if file not found
        """
        if not os.path.exists(self.schema_registry_path):
            logger.error(f"Schema registry file not found: {self.schema_registry_path}")
            return None

        with open(self.schema_registry_path, "r") as f:
            data = json.load(f)

        registry = SchemaRegistry.from_dict(data)
        logger.info(f"  Loaded schema with {len(registry.tables)} tables")

        return registry

    # =========================================================================
    # Entity Identification Methods
    # =========================================================================

    def _identify_entities_by_source(
        self,
        qa_extractions: QAExtractionRegistry,
        schema_registry: SchemaRegistry,
    ) -> Tuple[Set[EntityIdentifier], Set[EntityIdentifier]]:
        """Identify entities from forget and retain extractions.

        Args:
            qa_extractions: Registry of QA extractions with source labels
            schema_registry: Schema registry for table info

        Returns:
            Tuple of (forget_entities, retain_entities) as sets of EntityIdentifier
        """
        forget_entities: Set[EntityIdentifier] = set()
        retain_entities: Set[EntityIdentifier] = set()

        for extraction in qa_extractions:
            entities = self._extract_entity_identifiers(extraction, schema_registry)

            if extraction.source == "forget":
                forget_entities.update(entities)
            elif extraction.source == "retain":
                retain_entities.update(entities)

        return forget_entities, retain_entities

    def _extract_entity_identifiers(
        self,
        extraction: QAExtraction,
        schema_registry: SchemaRegistry,
    ) -> List[EntityIdentifier]:
        """Extract entity identifiers from a single QA extraction.

        Args:
            extraction: QA extraction with entities
            schema_registry: Schema registry for table info

        Returns:
            List of EntityIdentifier objects
        """
        identifiers: List[EntityIdentifier] = []

        for entity_type, entities in extraction.entities.items():
            table = schema_registry.get_table(entity_type)
            if not table:
                continue

            # Get natural key column for this table
            natural_key_col = table.get_conflict_key()
            pk_col = table.get_primary_key()

            for entity in entities:
                # Find natural key value
                natural_key_value = None
                if natural_key_col and natural_key_col in entity:
                    natural_key_value = entity[natural_key_col]
                elif "name" in entity:
                    natural_key_value = entity["name"]
                elif "title" in entity:
                    natural_key_value = entity["title"]

                if natural_key_value:
                    identifiers.append(
                        EntityIdentifier(
                            table_name=entity_type,
                            primary_key_column=pk_col,
                            natural_key_column=natural_key_col or "name",
                            natural_key_value=str(natural_key_value),
                            source=extraction.source,
                        )
                    )

        return identifiers

    def _compute_entities_to_remove(
        self,
        forget_entities: Set[EntityIdentifier],
        retain_entities: Set[EntityIdentifier],
    ) -> Set[EntityIdentifier]:
        """Compute entities that should be removed (forget-only).

        Entities that appear in both forget and retain sets are kept.

        Args:
            forget_entities: Entities from forget QA pairs
            retain_entities: Entities from retain QA pairs

        Returns:
            Set of entities to remove (forget - retain)
        """
        # Convert retain entities to a set for O(1) lookup
        retain_keys = {(e.table_name, e.natural_key_value) for e in retain_entities}

        # Filter forget entities that don't appear in retain
        entities_to_remove: Set[EntityIdentifier] = set()
        shared_count = 0

        for entity in forget_entities:
            key = (entity.table_name, entity.natural_key_value)
            if key not in retain_keys:
                entities_to_remove.add(entity)
            else:
                shared_count += 1

        logger.info(
            f"  {shared_count} entities shared between forget and retain (kept)"
        )

        return entities_to_remove

    def _filter_entities_present_in_aligned_db(
        self,
        entities_to_remove: Set[EntityIdentifier],
        schema_registry: SchemaRegistry,
    ) -> Tuple[Set[EntityIdentifier], int]:
        """Filter forget-only entities to those that actually exist in the aligned DB."""
        del schema_registry  # Natural-key metadata already lives on EntityIdentifier.

        filtered_entities: Set[EntityIdentifier] = set()
        skipped_absent_entities = 0
        grouped_entities: Dict[Tuple[str, str], Dict[str, EntityIdentifier]] = {}

        for entity in entities_to_remove:
            if not entity.natural_key_column or not entity.natural_key_value:
                skipped_absent_entities += 1
                continue
            key = (entity.table_name, entity.natural_key_column)
            grouped_entities.setdefault(key, {})[entity.natural_key_value] = entity

        cursor = self.aligned_pg_client.conn.cursor()
        try:
            for (table_name, natural_key_column), entities_by_value in grouped_entities.items():
                values = list(entities_by_value.keys())
                for chunk in self._chunk_values(values):
                    in_clause, params = self._build_batched_in_clause(chunk)
                    check_sql = (
                        f'SELECT "{natural_key_column}" FROM "{table_name}" '
                        f'WHERE "{natural_key_column}" {in_clause}'
                    )
                    cursor.execute(check_sql, params)
                    existing_values = {str(row[0]) for row in cursor.fetchall()}

                    for value in chunk:
                        entity = entities_by_value[value]
                        if str(value) in existing_values:
                            filtered_entities.add(entity)
                        else:
                            skipped_absent_entities += 1
        finally:
            cursor.close()

        return filtered_entities, skipped_absent_entities

    # =========================================================================
    # Cascade Plan Computation
    # =========================================================================

    def _compute_cascade_plan(
        self,
        entities_to_remove: Set[EntityIdentifier],
        schema_registry: SchemaRegistry,
    ) -> List[Tuple[str, str, str]]:
        """Compute the deletion plan respecting FK constraints.

        The plan is a list of (table_name, column_name, value) tuples
        in the correct order for deletion (dependent tables first).

        Args:
            entities_to_remove: Entities to remove
            schema_registry: Schema registry with FK info

        Returns:
            Ordered list of deletion operations
        """
        # Build dependency graph
        table_deps: Dict[str, Set[str]] = {}
        for table_name, table in schema_registry.tables.items():
            table_deps[table_name] = set()
            for fk in table.foreign_keys:
                # This table depends on the referenced table
                if fk.references_table in schema_registry.tables:
                    table_deps[table_name].add(fk.references_table)

        # Topological sort (reverse order - dependents first)
        sorted_tables = self._topological_sort_tables(table_deps)

        # Group entities by table
        entities_by_table: Dict[str, List[EntityIdentifier]] = {}
        for entity in entities_to_remove:
            if entity.table_name not in entities_by_table:
                entities_by_table[entity.table_name] = []
            entities_by_table[entity.table_name].append(entity)

        # Build deletion plan
        plan: List[Tuple[str, str, str]] = []
        for table_name in sorted_tables:
            if table_name in entities_by_table:
                for entity in entities_by_table[table_name]:
                    if entity.natural_key_column and entity.natural_key_value:
                        plan.append(
                            (
                                entity.table_name,
                                entity.natural_key_column,
                                entity.natural_key_value,
                            )
                        )

        logger.info(f"  Deletion plan has {len(plan)} operations")
        return plan

    def _topological_sort_tables(self, deps: Dict[str, Set[str]]) -> List[str]:
        """Sort tables so dependent tables come before their dependencies.

        Args:
            deps: Dictionary mapping table to its dependencies

        Returns:
            List of table names in deletion order
        """
        result: List[str] = []
        visited: Set[str] = set()
        temp_visited: Set[str] = set()

        def visit(table: str) -> None:
            if table in temp_visited:
                return  # Cycle detected
            if table in visited:
                return

            temp_visited.add(table)

            # Visit tables that depend on this one first
            for other_table, other_deps in deps.items():
                if table in other_deps:
                    visit(other_table)

            temp_visited.remove(table)
            visited.add(table)
            result.append(table)

        for table in deps:
            visit(table)

        return result

    # =========================================================================
    # Nullification Execution
    # =========================================================================

    @staticmethod
    def _chunk_values(values: List[Any], chunk_size: int = 500) -> List[List[Any]]:
        """Split a list into stable chunks for batched SQL operations."""
        if chunk_size <= 0:
            return [values]
        return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]

    @staticmethod
    def _build_batched_in_clause(values: List[Any]) -> Tuple[str, Tuple[Any, ...]]:
        """Build an IN-clause placeholder fragment and parameter tuple."""
        placeholders = ", ".join(["%s"] * len(values))
        return f"IN ({placeholders})", tuple(values)

    @staticmethod
    def _group_deletion_plan(
        deletion_plan: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, List[str]]]:
        """Group deletion operations by table and natural-key column."""
        grouped: Dict[Tuple[str, str], List[str]] = {}
        ordered_keys: List[Tuple[str, str]] = []

        for table_name, column_name, value in deletion_plan:
            key = (table_name, column_name)
            if key not in grouped:
                grouped[key] = []
                ordered_keys.append(key)
            grouped[key].append(value)

        result: List[Tuple[str, str, List[str]]] = []
        for key in ordered_keys:
            deduplicated_values = list(dict.fromkeys(grouped[key]))
            result.append((key[0], key[1], deduplicated_values))
        return result

    @staticmethod
    def _build_referencing_fk_map(
        schema_registry: SchemaRegistry,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Map a target table to referencing table/FK-column pairs."""
        referencing: Dict[str, List[Tuple[str, str]]] = {}
        for table_name, table in schema_registry.tables.items():
            for fk in table.foreign_keys:
                referencing.setdefault(fk.references_table.lower(), []).append(
                    (table_name, fk.column_name)
                )
        return referencing

    @staticmethod
    def _get_table_primary_key(
        schema_registry: SchemaRegistry, table_name: str
    ) -> Optional[str]:
        """Resolve the primary key column for a table."""
        table = schema_registry.get_table(table_name)
        if table is None:
            return None
        return table.get_primary_key() or f"{table_name}_id"

    def _fetch_entity_id_map(
        self,
        cursor: Any,
        table_name: str,
        natural_key_column: str,
        natural_key_values: List[str],
        schema_registry: SchemaRegistry,
    ) -> Dict[str, Any]:
        """Fetch primary-key values for a batch of entity natural keys."""
        primary_key_column = self._get_table_primary_key(schema_registry, table_name)
        if primary_key_column is None or not natural_key_values:
            return {}

        id_map: Dict[str, Any] = {}
        for chunk in self._chunk_values(natural_key_values):
            in_clause, params = self._build_batched_in_clause(chunk)
            select_sql = (
                f'SELECT "{primary_key_column}", "{natural_key_column}" '
                f'FROM "{table_name}" WHERE "{natural_key_column}" {in_clause}'
            )
            cursor.execute(select_sql, params)
            for primary_key_value, natural_key_value in cursor.fetchall():
                id_map[str(natural_key_value)] = primary_key_value

        return id_map

    def _build_protected_entity_ids(
        self,
        retain_entities: Set[EntityIdentifier],
        schema_registry: SchemaRegistry,
    ) -> Dict[str, Set[Any]]:
        """Build a transitive set of retain-protected row IDs by table.

        A forget-only parent row can still be required by a retained child row
        via foreign keys. If we delete that parent, the retained row can be
        deleted or invalidated by FK cleanup. Protect the retain rows and the
        rows they reference recursively.
        """
        protected_ids: Dict[str, Set[Any]] = {}
        work_queue: List[Tuple[str, Any]] = []

        def add_protected(table_name: str, row_id: Any) -> None:
            ids = protected_ids.setdefault(table_name, set())
            if row_id in ids:
                return
            ids.add(row_id)
            work_queue.append((table_name, row_id))

        grouped_entities: Dict[Tuple[str, str], Dict[str, EntityIdentifier]] = {}
        for entity in retain_entities:
            if not entity.natural_key_column or not entity.natural_key_value:
                continue
            key = (entity.table_name, entity.natural_key_column)
            grouped_entities.setdefault(key, {})[entity.natural_key_value] = entity

        cursor = self.aligned_pg_client.conn.cursor()
        try:
            for (table_name, natural_key_column), entities_by_value in grouped_entities.items():
                id_map = self._fetch_entity_id_map(
                    cursor,
                    table_name,
                    natural_key_column,
                    list(entities_by_value.keys()),
                    schema_registry,
                )
                for row_id in id_map.values():
                    add_protected(table_name, row_id)

            while work_queue:
                table_name, row_id = work_queue.pop()
                table = schema_registry.get_table(table_name)
                primary_key_column = self._get_table_primary_key(
                    schema_registry,
                    table_name,
                )
                if table is None or primary_key_column is None:
                    continue
                if not table.foreign_keys:
                    continue

                fk_column_sql = ", ".join(
                    f'"{fk.column_name}"' for fk in table.foreign_keys
                )
                select_sql = (
                    f'SELECT {fk_column_sql} FROM "{table_name}" '
                    f'WHERE "{primary_key_column}" = %s LIMIT 1'
                )
                cursor.execute(select_sql, (row_id,))
                rows = cursor.fetchall()
                if not rows:
                    continue

                for fk, fk_value in zip(table.foreign_keys, rows[0]):
                    if fk_value is None:
                        continue
                    add_protected(fk.references_table, fk_value)
        finally:
            cursor.close()

        return protected_ids

    def _delete_referencing_rows(
        self,
        cursor: Any,
        entity_table: str,
        entity_ids: List[Any],
        referencing_fk_map: Dict[str, List[Tuple[str, str]]],
    ) -> Tuple[int, Set[str]]:
        """Delete referencing rows in batches for a resolved list of entity IDs."""
        if not entity_ids:
            return 0, set()

        deleted = 0
        affected_tables: Set[str] = set()

        for table_name, fk_column in referencing_fk_map.get(entity_table.lower(), []):
            for chunk in self._chunk_values(entity_ids):
                in_clause, params = self._build_batched_in_clause(chunk)
                delete_sql = (
                    f'DELETE FROM "{table_name}" WHERE "{fk_column}" {in_clause} RETURNING 1'
                )
                cursor.execute(delete_sql, params)
                deleted_rows = len(cursor.fetchall())
                if deleted_rows > 0:
                    deleted += deleted_rows
                    affected_tables.add(table_name)

        return deleted, affected_tables

    def _delete_entity_rows(
        self,
        cursor: Any,
        table_name: str,
        column_name: str,
        values: List[str],
    ) -> int:
        """Delete entity rows in batches by natural-key value."""
        deleted = 0
        for chunk in self._chunk_values(values):
            in_clause, params = self._build_batched_in_clause(chunk)
            delete_sql = (
                f'DELETE FROM "{table_name}" WHERE "{column_name}" {in_clause} RETURNING 1'
            )
            cursor.execute(delete_sql, params)
            deleted += len(cursor.fetchall())
        return deleted

    def _execute_nullification(
        self,
        deletion_plan: List[Tuple[str, str, str]],
        schema_registry: SchemaRegistry,
        protected_entity_ids: Optional[Dict[str, Set[Any]]] = None,
    ) -> NullificationExecutionStats:
        """Execute the nullification plan.

        Args:
            deletion_plan: List of (table, column, value) tuples
            schema_registry: Schema registry

        Returns:
            Direct-deletion execution summary before cleanup runs
        """
        entities_removed = 0
        relations_removed = 0
        matched_entity_keys = 0
        planned_entity_keys = len(deletion_plan)
        tables_affected: Set[str] = set()
        grouped_plan = self._group_deletion_plan(deletion_plan)
        referencing_fk_map = self._build_referencing_fk_map(schema_registry)
        protected_entity_ids = protected_entity_ids or {}

        cursor = self.null_pg_client.conn.cursor()

        try:
            for table_name, column_name, values in grouped_plan:
                try:
                    entity_id_map = self._fetch_entity_id_map(
                        cursor,
                        table_name,
                        column_name,
                        values,
                        schema_registry,
                    )
                    matched_entity_keys += len(entity_id_map)
                    protected_ids = protected_entity_ids.get(table_name, set())
                    deletable_values = [
                        value
                        for value in values
                        if entity_id_map.get(str(value)) not in protected_ids
                    ]
                    deletable_ids = [
                        entity_id_map[str(value)]
                        for value in deletable_values
                        if str(value) in entity_id_map
                    ]

                    if len(deletable_values) < len(values):
                        logger.info(
                            "  Preserved %d retain-protected %s row(s)",
                            len(values) - len(deletable_values),
                            table_name,
                        )

                    junction_deleted, referencing_tables = self._delete_referencing_rows(
                        cursor,
                        table_name,
                        deletable_ids,
                        referencing_fk_map,
                    )
                    relations_removed += junction_deleted
                    tables_affected.update(referencing_tables)

                    deleted_rows = self._delete_entity_rows(
                        cursor,
                        table_name,
                        column_name,
                        deletable_values,
                    )
                    if deleted_rows > 0:
                        entities_removed += deleted_rows
                        tables_affected.add(table_name)
                        logger.debug(
                            f"  Deleted {deleted_rows} rows from {table_name} "
                            f"using {len(deletable_values)} batched key(s) on {column_name}"
                        )

                except Exception as e:
                    logger.warning(
                        f"  Failed to delete from {table_name} "
                        f"using {len(values)} values on {column_name}: {e}"
                    )

            # Commit changes
            self.null_pg_client.conn.commit()

        finally:
            cursor.close()

        return NullificationExecutionStats(
            entities_removed=entities_removed,
            relations_removed=relations_removed,
            tables_affected=list(tables_affected),
            matched_entity_keys=matched_entity_keys,
            planned_entity_keys=planned_entity_keys,
        )

    # =========================================================================
    # Verification Methods
    # =========================================================================

    def _verify_retain_integrity(
        self,
        retain_entities: Set[EntityIdentifier],
        schema_registry: SchemaRegistry,
    ) -> bool:
        """Verify that retain entities still exist in the database.

        Args:
            retain_entities: Set of entities that should be retained
            schema_registry: Schema registry

        Returns:
            True if all retain entities are present
        """
        aligned_cursor = self.aligned_pg_client.conn.cursor()
        null_cursor = self.null_pg_client.conn.cursor()
        missing_count = 0
        skipped_count = 0
        baseline_count = 0
        preserved_count = 0

        try:
            grouped_entities: Dict[Tuple[str, str], List[str]] = {}
            for entity in retain_entities:
                if not entity.natural_key_column or not entity.natural_key_value:
                    continue
                key = (entity.table_name, entity.natural_key_column)
                grouped_entities.setdefault(key, []).append(entity.natural_key_value)

            for (table_name, natural_key_column), values in grouped_entities.items():
                deduplicated_values = list(dict.fromkeys(values))
                try:
                    for chunk in self._chunk_values(deduplicated_values):
                        in_clause, params = self._build_batched_in_clause(chunk)
                        check_sql = (
                            f'SELECT "{natural_key_column}" FROM "{table_name}" '
                            f'WHERE "{natural_key_column}" {in_clause}'
                        )
                        aligned_cursor.execute(check_sql, params)
                        aligned_values = {str(row[0]) for row in aligned_cursor.fetchall()}
                        null_cursor.execute(check_sql, params)
                        null_values = {str(row[0]) for row in null_cursor.fetchall()}

                        baseline_count += len(aligned_values)
                        preserved_count += len(aligned_values & null_values)
                        skipped_values = [
                            value for value in chunk if str(value) not in aligned_values
                        ]
                        skipped_count += len(skipped_values)
                        missing_values = [
                            value
                            for value in chunk
                            if str(value) in aligned_values
                            and str(value) not in null_values
                        ]
                        missing_count += len(missing_values)
                        for missing_value in missing_values:
                            logger.warning(
                                f"  Retain entity missing: {table_name}."
                                f"{natural_key_column}='{missing_value}'"
                            )
                        if skipped_values:
                            logger.debug(
                                "  Skipping %d retain entities absent from aligned DB for %s.%s",
                                len(skipped_values),
                                table_name,
                                natural_key_column,
                            )
                except Exception as e:
                    logger.debug(f"  Verify failed for {table_name}: {e}")

        finally:
            aligned_cursor.close()
            null_cursor.close()

        logger.info(
            "  Retain integrity baseline=%d preserved=%d missing=%d skipped=%d",
            baseline_count,
            preserved_count,
            missing_count,
            skipped_count,
        )

        return missing_count == 0

    def _verify_schema_consistency(self) -> bool:
        """Verify that null DB schema matches aligned DB schema.

        Compares tables, columns, and their types between the two databases
        using the information_schema.

        Returns:
            True if schemas match, False otherwise
        """
        logger.info(
            "Verifying schema consistency between aligned and null databases..."
        )

        # Query to get table and column information
        schema_query = """
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """

        # Get aligned DB schema
        aligned_cursor = self.aligned_pg_client.conn.cursor()
        aligned_cursor.execute(schema_query)
        aligned_schema = aligned_cursor.fetchall()
        aligned_cursor.close()

        # Get null DB schema
        null_cursor = self.null_pg_client.conn.cursor()
        null_cursor.execute(schema_query)
        null_schema = null_cursor.fetchall()
        null_cursor.close()

        # Convert to sets for comparison
        aligned_set = set(aligned_schema)
        null_set = set(null_schema)

        # Find differences
        in_aligned_only = aligned_set - null_set
        in_null_only = null_set - aligned_set

        # Get table counts
        aligned_tables = set(row[0] for row in aligned_schema)
        null_tables = set(row[0] for row in null_schema)

        logger.info(
            f"  Aligned DB: {len(aligned_tables)} tables, {len(aligned_schema)} columns"
        )
        logger.info(f"  Null DB: {len(null_tables)} tables, {len(null_schema)} columns")

        if in_aligned_only or in_null_only:
            logger.error("Schema mismatch detected!")

            if in_aligned_only:
                logger.error(
                    f"  Columns in aligned DB but NOT in null DB ({len(in_aligned_only)}):"
                )
                # Group by table for clearer output
                tables_missing = {}
                for row in in_aligned_only:
                    table_name = row[0]
                    if table_name not in tables_missing:
                        tables_missing[table_name] = []
                    tables_missing[table_name].append(row[1])  # column_name

                for table, columns in sorted(tables_missing.items()):
                    logger.error(f"    - {table}: {columns}")

            if in_null_only:
                logger.error(
                    f"  Columns in null DB but NOT in aligned DB ({len(in_null_only)}):"
                )
                tables_extra = {}
                for row in in_null_only:
                    table_name = row[0]
                    if table_name not in tables_extra:
                        tables_extra[table_name] = []
                    tables_extra[table_name].append(row[1])

                for table, columns in sorted(tables_extra.items()):
                    logger.error(f"    - {table}: {columns}")

            # Check for missing tables specifically
            tables_missing_entirely = aligned_tables - null_tables
            tables_extra_entirely = null_tables - aligned_tables

            if tables_missing_entirely:
                logger.error(
                    f"  Tables missing from null DB: {sorted(tables_missing_entirely)}"
                )
            if tables_extra_entirely:
                logger.error(
                    f"  Extra tables in null DB: {sorted(tables_extra_entirely)}"
                )

            return False

        logger.info(
            "  Schema verification passed - both databases have identical schemas"
        )
        return True

    def _log_row_count_comparison(
        self, schema_registry: SchemaRegistry
    ) -> Tuple[Dict[str, Dict[str, int]], bool]:
        """Compare row counts between aligned and null databases.

        Args:
            schema_registry: Schema registry containing table information

        Returns:
            Tuple of:
                - Dict mapping table_name -> {aligned: count, null: count, removed: diff}
                - bool indicating if verification passed (True) or failed (False)
        """
        comparison: Dict[str, Dict[str, int]] = {}
        total_aligned = 0
        total_null = 0
        verification_passed = True

        for table_name in schema_registry.get_table_names():
            try:
                # Query aligned DB
                aligned_cursor = self.aligned_pg_client.conn.cursor()
                aligned_cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                aligned_count = aligned_cursor.fetchone()[0]
                aligned_cursor.close()

                # Query null DB
                null_cursor = self.null_pg_client.conn.cursor()
                null_cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                null_count = null_cursor.fetchone()[0]
                null_cursor.close()

                comparison[table_name] = {
                    "aligned": aligned_count,
                    "null": null_count,
                    "removed": aligned_count - null_count,
                }
                total_aligned += aligned_count
                total_null += null_count

            except Exception as e:
                logger.warning(f"  Failed to get row count for {table_name}: {e}")
                comparison[table_name] = {"aligned": 0, "null": 0, "removed": 0}

        # Log summary
        total_removed = total_aligned - total_null
        logger.info(f"  Total rows - Aligned: {total_aligned}, Null: {total_null}")
        logger.info(f"  Total rows removed: {total_removed}")

        # Log per-table details for tables with removals
        tables_with_removals = [
            (table, counts)
            for table, counts in sorted(comparison.items())
            if counts["removed"] > 0
        ]
        if tables_with_removals:
            logger.info("  Per-table breakdown:")
            for table, counts in tables_with_removals:
                logger.info(
                    f"    {table}: {counts['aligned']} -> {counts['null']} "
                    f"(-{counts['removed']})"
                )

        # Verify: null DB should not have MORE rows than aligned DB
        tables_with_extra_rows = [
            (table, counts)
            for table, counts in comparison.items()
            if counts["removed"] < 0
        ]
        if tables_with_extra_rows:
            verification_passed = False
            logger.error("Row count verification FAILED!")
            logger.error(
                "  Null DB has MORE rows than aligned DB in the following tables:"
            )
            for table, counts in tables_with_extra_rows:
                logger.error(
                    f"    {table}: aligned={counts['aligned']}, "
                    f"null={counts['null']} (+{-counts['removed']} extra rows)"
                )

        # Verify: at least some rows should have been removed (sanity check)
        if total_removed == 0:
            verification_passed = False
            logger.error("Row count verification FAILED!")
            logger.error("  No rows were removed during nullification.")
            logger.error("  This indicates the forget set may not have been processed.")

        if verification_passed:
            logger.info("  Row count verification passed")

        return comparison, verification_passed

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def _save_nullification_log(
        self,
        result: NullificationResult,
        entities_removed: Set[EntityIdentifier],
    ) -> None:
        """Save nullification log for debugging.

        Args:
            result: Nullification result
            entities_removed: Set of removed entities
        """
        os.makedirs(self.nullify_log_dir_path, exist_ok=True)

        # Save summary
        summary_path = os.path.join(self.nullify_log_dir_path, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save removed entities
        removed_path = os.path.join(self.nullify_log_dir_path, "removed_entities.json")
        removed_list = [
            {
                "table": e.table_name,
                "column": e.natural_key_column,
                "value": e.natural_key_value,
            }
            for e in entities_removed
        ]
        with open(removed_path, "w") as f:
            json.dump(removed_list, f, indent=2)

        logger.info(f"  Saved nullification log to {self.nullify_log_dir_path}")
