"""Tests for nullified database builder behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection
        self._rows: List[Tuple[Any, ...]] = []
        self.rowcount = 0

    def execute(self, query: str, params: Optional[Sequence[Any]] = None) -> None:
        normalized_params = tuple(params or ())
        self._connection.history.append((query, normalized_params))
        self._rows, self.rowcount = self._connection.handler(query, normalized_params)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._rows)

    def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(
        self,
        handler: Callable[[str, Tuple[Any, ...]], Tuple[List[Tuple[Any, ...]], int]],
    ) -> None:
        self.handler = handler
        self.history: List[Tuple[str, Tuple[Any, ...]]] = []
        self.commit_count = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commit_count += 1


class _FakePGClient:
    def __init__(
        self,
        handler: Callable[[str, Tuple[Any, ...]], Tuple[List[Tuple[Any, ...]], int]],
    ) -> None:
        self.conn = _FakeConnection(handler)


class TestNullifiedDBBuilder(unittest.TestCase):
    """Regression tests for nullification build semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise unittest.SkipTest(f"Missing optional dependency: {exc}") from exc

    def _make_cfg(self, root: str):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "project_path": root,
                "model": {"dir_path": "aligned_db"},
                "database": {
                    "db_id": "tofu_data",
                    "null_db_id": "tofu_data_null",
                    "host": "localhost",
                    "port": 5433,
                    "null_port": 5433,
                    "user_id": "postgres",
                    "passwd": "postgres",
                },
            }
        )

    def _make_schema_registry(self):
        from src.aligned_db.schema_registry import (
            ColumnInfo,
            ForeignKeyConstraint,
            SchemaRegistry,
            TableSchema,
        )

        registry = SchemaRegistry.empty()
        registry.add_table(
            TableSchema(
                name="person",
                columns=[
                    ColumnInfo("person_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("name", data_type="TEXT"),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="person_work",
                columns=[
                    ColumnInfo("person_work_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("person_id", data_type="INTEGER"),
                    ColumnInfo("work_id", data_type="INTEGER"),
                ],
                foreign_keys=[
                    ForeignKeyConstraint(
                        column_name="person_id",
                        references_table="person",
                        references_column="person_id",
                    )
                ],
            )
        )
        return registry

    def test_build_raises_when_required_files_are_missing(self):
        """Missing prerequisite artifacts should fail fast instead of returning a partial result."""
        from src.aligned_db.nullified_db import NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))

            with self.assertRaises(FileNotFoundError):
                builder.build(overwrite=True)

    def test_refresh_null_database_recreates_target_from_aligned(self):
        """Nullification should recreate the null DB from the aligned DB before deletes."""
        from src.aligned_db.nullified_db import NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))
            calls: List[Tuple[str, Dict[str, Any]]] = []

            class _FakeOps:
                def __init__(self, cfg):
                    del cfg

                def dump_database(self, **kwargs):
                    calls.append(("dump_database", kwargs))
                    Path(kwargs["output_path"]).write_text("-- dump", encoding="utf-8")

                def drop_database(self, **kwargs):
                    calls.append(("drop_database", kwargs))

                def create_database(self, **kwargs):
                    calls.append(("create_database", kwargs))

                def execute_sql_file(self, **kwargs):
                    calls.append(("execute_sql_file", kwargs))

            with mock.patch(
                "src.aligned_db.nullified_db.PostgresShellOperations",
                _FakeOps,
            ):
                builder._refresh_null_database_from_aligned()

            self.assertEqual(
                [name for name, _ in calls],
                [
                    "dump_database",
                    "drop_database",
                    "create_database",
                    "execute_sql_file",
                ],
            )

    def test_cached_summary_restores_row_comparison(self):
        """Cached nullification summaries should round-trip row-comparison data."""
        from src.aligned_db.nullified_db import NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))
            summary_path = Path(builder.nullify_log_dir_path) / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "entities_removed": 3,
                "relations_removed": 2,
                "candidate_entity_keys": 5,
                "skipped_absent_entity_keys": 1,
                "matched_entity_keys": 3,
                "planned_entity_keys": 4,
                "cleanup_rows_deleted": 1,
                "retain_verified": True,
                "tables_affected": ["person", "work"],
                "errors": [],
                "row_comparison": {
                    "person": {"aligned": 10, "null": 8, "removed": 2}
                },
            }
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            result = builder.build(overwrite=False)

            self.assertEqual(result.entities_removed, 3)
            self.assertEqual(result.candidate_entity_keys, 5)
            self.assertEqual(result.matched_entity_keys, 3)
            self.assertEqual(
                result.row_comparison,
                {"person": {"aligned": 10, "null": 8, "removed": 2}},
            )

    def test_execute_nullification_batches_id_resolution_and_deletes(self):
        """Nullification should batch FK cleanup and entity deletes by table/key column."""
        from src.aligned_db.nullified_db import NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))
            schema_registry = self._make_schema_registry()

            def handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "person_id", "name" FROM "person"'):
                    return [(1, "Ada"), (2, "Grace")], 2
                if query.startswith('DELETE FROM "person_work"'):
                    return [(1,), (1,)], 2
                if query.startswith('DELETE FROM "person"'):
                    return [(1,), (1,)], 2
                raise AssertionError(f"Unexpected query: {query}")

            builder.null_pg_client = _FakePGClient(handler)

            execution_stats = builder._execute_nullification(
                [
                    ("person", "name", "Ada"),
                    ("person", "name", "Grace"),
                ],
                schema_registry,
            )

            history = builder.null_pg_client.conn.history
            self.assertEqual(len(history), 3)
            self.assertEqual(execution_stats.entities_removed, 2)
            self.assertEqual(execution_stats.relations_removed, 2)
            self.assertEqual(execution_stats.matched_entity_keys, 2)
            self.assertEqual(execution_stats.planned_entity_keys, 2)
            self.assertEqual(
                set(execution_stats.tables_affected), {"person", "person_work"}
            )
            self.assertEqual(builder.null_pg_client.conn.commit_count, 1)

    def test_build_protected_entity_ids_includes_fk_references_of_retain_rows(self):
        """Retained rows should protect their referenced parent rows from deletion."""
        from src.aligned_db.nullified_db import EntityIdentifier, NullifiedDBBuilder
        from src.aligned_db.schema_registry import (
            ColumnInfo,
            ForeignKeyConstraint,
            SchemaRegistry,
            TableSchema,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))
            schema_registry = SchemaRegistry.empty()
            schema_registry.add_table(
                TableSchema(
                    name="location",
                    columns=[
                        ColumnInfo("location_id", data_type="INTEGER", is_primary_key=True),
                        ColumnInfo("name", data_type="TEXT"),
                    ],
                )
            )
            schema_registry.add_table(
                TableSchema(
                    name="person",
                    columns=[
                        ColumnInfo("person_id", data_type="INTEGER", is_primary_key=True),
                        ColumnInfo("name", data_type="TEXT"),
                        ColumnInfo("birth_place_id", data_type="INTEGER"),
                    ],
                    foreign_keys=[
                        ForeignKeyConstraint(
                            column_name="birth_place_id",
                            references_table="location",
                            references_column="location_id",
                        )
                    ],
                )
            )

            def aligned_handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "person_id", "name" FROM "person"'):
                    return [(1, "Ada")], 1
                if query.startswith('SELECT "birth_place_id" FROM "person"'):
                    return [(10,)], 1
                if query.startswith('SELECT "location_id", "name" FROM "location"'):
                    return [(10, "London")], 1
                raise AssertionError(f"Unexpected query: {query}")

            builder.aligned_pg_client = _FakePGClient(aligned_handler)
            protected = builder._build_protected_entity_ids(
                {
                    EntityIdentifier(
                        table_name="person",
                        natural_key_column="name",
                        natural_key_value="Ada",
                    )
                },
                schema_registry,
            )

            self.assertEqual(protected["person"], {1})
            self.assertEqual(protected["location"], {10})

    def test_execute_nullification_skips_retain_protected_rows(self):
        """Deletion should skip rows whose IDs are protected by retain references."""
        from src.aligned_db.nullified_db import NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))
            schema_registry = self._make_schema_registry()

            def handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "person_id", "name" FROM "person"'):
                    return [(1, "Ada"), (2, "Grace")], 2
                if query.startswith('DELETE FROM "person_work"'):
                    return [(1,)], 1
                if query.startswith('DELETE FROM "person"'):
                    return [(1,)], 1
                raise AssertionError(f"Unexpected query: {query}")

            builder.null_pg_client = _FakePGClient(handler)

            execution_stats = builder._execute_nullification(
                [
                    ("person", "name", "Ada"),
                    ("person", "name", "Grace"),
                ],
                schema_registry,
                protected_entity_ids={"person": {1}},
            )

            self.assertEqual(execution_stats.entities_removed, 1)
            self.assertEqual(execution_stats.relations_removed, 1)
            self.assertEqual(execution_stats.matched_entity_keys, 2)

    def test_build_raises_when_null_db_is_already_stale(self):
        """Re-running nullification against an already-nullified DB should fail fast."""
        from src.aligned_db.nullified_db import (
            NullificationExecutionStats,
            NullifiedDBBuilder,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))

            builder._load_qa_extractions = lambda: object()
            builder._load_schema_registry = self._make_schema_registry
            builder._identify_entities_by_source = lambda qa, sr: (set(), set())
            builder._compute_entities_to_remove = lambda forget, retain: {object()}
            builder._filter_entities_present_in_aligned_db = (
                lambda entities, sr: (entities, 0)
            )
            builder._compute_cascade_plan = lambda entities, sr: [
                ("person", "name", "Ada"),
            ]
            builder._execute_nullification = lambda plan, sr, **kwargs: NullificationExecutionStats(
                entities_removed=0,
                relations_removed=0,
                tables_affected=[],
                matched_entity_keys=0,
                planned_entity_keys=len(plan),
            )

            with self.assertRaises(RuntimeError):
                builder.build(overwrite=True)

    def test_filter_entities_present_in_aligned_db_skips_absent_forget_keys(self):
        """Forget-only entities absent from the aligned DB should be pruned before planning."""
        from src.aligned_db.nullified_db import EntityIdentifier, NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))

            def aligned_handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "name" FROM "person"'):
                    return [("Ada",)], 1
                raise AssertionError(f"Unexpected aligned query: {query}")

            builder.aligned_pg_client = _FakePGClient(aligned_handler)

            filtered_entities, skipped_count = (
                builder._filter_entities_present_in_aligned_db(
                    {
                        EntityIdentifier(
                            table_name="person",
                            natural_key_column="name",
                            natural_key_value="Ada",
                        ),
                        EntityIdentifier(
                            table_name="person",
                            natural_key_column="name",
                            natural_key_value="Ghost",
                        ),
                    },
                    self._make_schema_registry(),
                )
            )

            self.assertEqual(
                {entity.natural_key_value for entity in filtered_entities},
                {"Ada"},
            )
            self.assertEqual(skipped_count, 1)

    def test_verify_retain_integrity_batches_by_table_and_key(self):
        """Retain verification should check grouped keys in batches, not one query per entity."""
        from src.aligned_db.nullified_db import EntityIdentifier, NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))

            def handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "name" FROM "person"'):
                    return [("Ada",), ("Grace",)], 2
                raise AssertionError(f"Unexpected query: {query}")

            builder.aligned_pg_client = _FakePGClient(handler)
            builder.null_pg_client = _FakePGClient(handler)

            retain_verified = builder._verify_retain_integrity(
                {
                    EntityIdentifier(
                        table_name="person",
                        natural_key_column="name",
                        natural_key_value="Ada",
                    ),
                    EntityIdentifier(
                        table_name="person",
                        natural_key_column="name",
                        natural_key_value="Grace",
                    ),
                },
                self._make_schema_registry(),
            )

            self.assertTrue(retain_verified)
            self.assertEqual(len(builder.aligned_pg_client.conn.history), 1)
            self.assertEqual(len(builder.null_pg_client.conn.history), 1)

    def test_verify_retain_integrity_ignores_entities_absent_from_aligned_db(self):
        """Extraction-only retain entities should not count as nullification regressions."""
        from src.aligned_db.nullified_db import EntityIdentifier, NullifiedDBBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = NullifiedDBBuilder(global_cfg=self._make_cfg(tmpdir))

            def aligned_handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "name" FROM "person"'):
                    return [("Ada",)], 1
                raise AssertionError(f"Unexpected aligned query: {query}")

            def null_handler(query: str, params: Tuple[Any, ...]):
                if query.startswith('SELECT "name" FROM "person"'):
                    return [("Ada",)], 1
                raise AssertionError(f"Unexpected null query: {query}")

            builder.aligned_pg_client = _FakePGClient(aligned_handler)
            builder.null_pg_client = _FakePGClient(null_handler)

            retain_verified = builder._verify_retain_integrity(
                {
                    EntityIdentifier(
                        table_name="person",
                        natural_key_column="name",
                        natural_key_value="Ada",
                    ),
                    EntityIdentifier(
                        table_name="person",
                        natural_key_column="name",
                        natural_key_value="Ghost",
                    ),
                },
                self._make_schema_registry(),
            )

            self.assertTrue(retain_verified)


if __name__ == "__main__":
    unittest.main()
