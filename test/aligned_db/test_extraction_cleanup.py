"""Tests for pruning QA extractions against the aligned DB baseline."""

from __future__ import annotations

import unittest
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

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakePGClient:
    def __init__(
        self,
        handler: Callable[[str, Tuple[Any, ...]], Tuple[List[Tuple[Any, ...]], int]],
    ) -> None:
        self.conn = _FakeConnection(handler)

    def execute(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> _FakeCursor:
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor


class TestExtractionCleanup(unittest.TestCase):
    """Regression coverage for aligned-DB extraction cleanup."""

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
                    ColumnInfo("name", data_type="TEXT", is_unique=True),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="work",
                columns=[
                    ColumnInfo("work_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("title", data_type="TEXT", is_unique=True),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="award",
                columns=[
                    ColumnInfo("award_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("name", data_type="TEXT", is_unique=True),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="person_work",
                columns=[
                    ColumnInfo("person_id", data_type="INTEGER"),
                    ColumnInfo("work_id", data_type="INTEGER"),
                ],
                foreign_keys=[
                    ForeignKeyConstraint("person_id", "person", "person_id"),
                    ForeignKeyConstraint("work_id", "work", "work_id"),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="person_award",
                columns=[
                    ColumnInfo("person_id", data_type="INTEGER"),
                    ColumnInfo("award_id", data_type="INTEGER"),
                ],
                foreign_keys=[
                    ForeignKeyConstraint("person_id", "person", "person_id"),
                    ForeignKeyConstraint("award_id", "award", "award_id"),
                ],
            )
        )
        return registry

    def test_prune_qa_extractions_to_aligned_db_removes_absent_entities(self):
        from src.aligned_db.extraction_cleanup import prune_qa_extractions_to_aligned_db
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        registry = QAExtractionRegistry.from_list(
            [
                QAExtraction(
                    qa_index=0,
                    question="Q1",
                    answer="A1",
                    entities={
                        "person": [{"name": "Ada"}, {"name": "Ghost"}],
                        "work": [{"title": "Hamlet"}, {"title": "Phantom"}],
                    },
                    relations=[
                        {"type": "person_work", "source": "Ada", "target": "Hamlet"},
                        {"type": "person_work", "source": "Ghost", "target": "Hamlet"},
                        {"type": "award_person", "source": "Turing Prize", "target": "Ada"},
                        {"type": "author_of", "source": "Ada", "target": "Hamlet"},
                    ],
                    relevant_tables={"person", "work"},
                ),
                QAExtraction(
                    qa_index=1,
                    question="Q2",
                    answer="A2",
                    entities={"person": [{"name": "Ghost"}]},
                    relations=[
                        {"type": "person_award", "source": "Ghost", "target": "Turing Prize"},
                    ],
                    relevant_tables={"person"},
                ),
            ]
        )

        def handler(query: str, params: Tuple[Any, ...]):
            if query.startswith("SELECT tablename FROM pg_tables"):
                return [
                    ("person",),
                    ("work",),
                    ("award",),
                    ("person_work",),
                    ("person_award",),
                ], 5
            if query.startswith('SELECT "name" FROM "person" WHERE "name" IS NOT NULL'):
                return [("Ada",)], 1
            if query.startswith('SELECT "title" FROM "work" WHERE "title" IS NOT NULL'):
                return [("Hamlet",)], 1
            if query.startswith('SELECT "name" FROM "award" WHERE "name" IS NOT NULL'):
                return [("Turing Prize",)], 1
            if query.startswith('SELECT "name" FROM "person"'):
                rows = [(value,) for value in params if value == "Ada"]
                return rows, len(rows)
            if query.startswith('SELECT "title" FROM "work"'):
                rows = [(value,) for value in params if value == "Hamlet"]
                return rows, len(rows)
            if query.startswith('SELECT "name" FROM "award"'):
                rows = [(value,) for value in params if value == "Turing Prize"]
                return rows, len(rows)
            raise AssertionError(f"Unexpected query: {query}")

        cleaned_registry, stats = prune_qa_extractions_to_aligned_db(
            pg_client=_FakePGClient(handler),
            schema_registry=self._make_schema_registry(),
            qa_extractions=registry,
        )

        first = cleaned_registry.get(0)
        second = cleaned_registry.get(1)
        assert first is not None
        assert second is not None

        self.assertEqual(first.entities["person"], [{"name": "Ada"}])
        self.assertEqual(first.entities["work"], [{"title": "Hamlet"}])
        self.assertEqual(
            first.relations,
            [
                {"type": "person_work", "source": "Ada", "target": "Hamlet"},
                {"type": "person_award", "source": "Ada", "target": "Turing Prize"},
            ],
        )
        self.assertEqual(second.entities, {})
        self.assertEqual(second.relevant_tables, set())
        self.assertEqual(stats.total_entities_before, 5)
        self.assertEqual(stats.total_entities_after, 2)
        self.assertEqual(stats.removed_entities, 3)
        self.assertEqual(stats.touched_extractions, 2)
        self.assertEqual(stats.removed_by_type, {"person": 2, "work": 1})
        self.assertEqual(stats.total_relations_before, 5)
        self.assertEqual(stats.total_relations_after, 2)
        self.assertEqual(stats.removed_relations, 3)
        self.assertEqual(stats.relinked_relations, 1)
        self.assertEqual(
            stats.removed_relations_by_type,
            {"author_of": 1, "person_award": 1, "person_work": 1},
        )

    def test_prune_qa_extractions_skips_missing_tables_after_cleanup(self):
        from src.aligned_db.extraction_cleanup import prune_qa_extractions_to_aligned_db
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry

        registry = QAExtractionRegistry.from_list(
            [
                QAExtraction(
                    qa_index=0,
                    question="Q1",
                    answer="A1",
                    entities={
                        "person": [{"name": "Ada"}],
                        "work": [{"title": "Hamlet"}],
                    },
                    relations=[
                        {"type": "person_work", "source": "Ada", "target": "Hamlet"},
                    ],
                    relevant_tables={"person", "work", "person_work"},
                )
            ]
        )

        def handler(query: str, params: Tuple[Any, ...]):
            if query.startswith("SELECT tablename FROM pg_tables"):
                return [("work",)], 1
            if query.startswith('SELECT "title" FROM "work" WHERE "title" IS NOT NULL'):
                return [("Hamlet",)], 1
            if query.startswith('SELECT "title" FROM "work"'):
                rows = [(value,) for value in params if value == "Hamlet"]
                return rows, len(rows)
            raise AssertionError(f"Unexpected query: {query}")

        cleaned_registry, stats = prune_qa_extractions_to_aligned_db(
            pg_client=_FakePGClient(handler),
            schema_registry=self._make_schema_registry(),
            qa_extractions=registry,
        )

        extraction = cleaned_registry.get(0)
        assert extraction is not None
        self.assertEqual(extraction.entities, {"work": [{"title": "Hamlet"}]})
        self.assertEqual(extraction.relations, [])
        self.assertEqual(stats.total_entities_before, 2)
        self.assertEqual(stats.total_entities_after, 1)
        self.assertEqual(stats.removed_entities, 1)
        self.assertEqual(stats.removed_by_type, {"person": 1})

    def test_prune_qa_extractions_drops_entities_without_real_schema_key(self):
        from src.aligned_db.extraction_cleanup import prune_qa_extractions_to_aligned_db
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
        from src.aligned_db.schema_registry import (
            ColumnInfo,
            ForeignKeyConstraint,
            SchemaRegistry,
            TableSchema,
        )

        schema_registry = SchemaRegistry.empty()
        schema_registry.add_table(
            TableSchema(
                name="theme",
                columns=[
                    ColumnInfo("theme_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("label", data_type="TEXT", is_unique=True),
                ],
            )
        )
        schema_registry.add_table(
            TableSchema(
                name="work",
                columns=[
                    ColumnInfo("work_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("title", data_type="TEXT", is_unique=True),
                ],
            )
        )
        schema_registry.add_table(
            TableSchema(
                name="theme_work",
                columns=[
                    ColumnInfo("theme_id", data_type="INTEGER"),
                    ColumnInfo("work_id", data_type="INTEGER"),
                ],
                foreign_keys=[
                    ForeignKeyConstraint("theme_id", "theme", "theme_id"),
                    ForeignKeyConstraint("work_id", "work", "work_id"),
                ],
            )
        )

        registry = QAExtractionRegistry.from_list(
            [
                QAExtraction(
                    qa_index=0,
                    question="Q",
                    answer="A",
                    entities={"theme_work": [{"name": "Identity and Belonging"}]},
                    relevant_tables={"theme_work"},
                )
            ]
        )

        def handler(query: str, params: Tuple[Any, ...]):
            if query.startswith("SELECT tablename FROM pg_tables"):
                return [("theme",), ("work",), ("theme_work",)], 3
            if query.startswith('SELECT "label" FROM "theme" WHERE "label" IS NOT NULL'):
                return [("Identity",)], 1
            if query.startswith('SELECT "title" FROM "work" WHERE "title" IS NOT NULL'):
                return [("Novel",)], 1
            if 'SELECT "name" FROM "theme_work"' in query:
                raise AssertionError("cleanup should not query a non-existent name column")
            raise AssertionError(f"Unexpected query: {query}")

        cleaned_registry, stats = prune_qa_extractions_to_aligned_db(
            pg_client=_FakePGClient(handler),
            schema_registry=schema_registry,
            qa_extractions=registry,
        )

        extraction = cleaned_registry.get(0)
        assert extraction is not None
        self.assertEqual(extraction.entities, {})
        self.assertEqual(stats.removed_entities, 1)
        self.assertEqual(stats.removed_by_type, {"theme_work": 1})

    def test_normalize_entity_attributes_maps_full_name_to_name(self):
        from src.aligned_db.sql_values import normalize_entity_attributes

        normalized = normalize_entity_attributes(
            entity={"full_name": "Ada Lovelace", "occupation": "mathematician"},
            entity_type="person",
            valid_columns=["person_id", "name", "occupation"],
        )

        self.assertEqual(
            normalized,
            {"name": "Ada Lovelace", "occupation": "mathematician"},
        )

    def test_remove_null_only_columns_reconciles_missing_live_columns(self):
        from src.aligned_db.cleanup import remove_null_only_columns
        from src.aligned_db.schema_registry import (
            ColumnInfo,
            SchemaRegistry,
            TableSchema,
        )

        schema_registry = SchemaRegistry.empty()
        schema_registry.add_table(
            TableSchema(
                name="person",
                columns=[
                    ColumnInfo("person_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("name", data_type="TEXT"),
                    ColumnInfo("ghost_column", data_type="TEXT"),
                ],
            )
        )

        executed_queries: List[str] = []

        def handler(query: str, params: Tuple[Any, ...]):
            executed_queries.append(query)
            if "FROM information_schema.tables" in query:
                return [("person",)], 1
            if "FROM information_schema.columns" in query:
                return [("person", "person_id"), ("person", "name")], 2
            if 'SELECT EXISTS(SELECT 1 FROM "person" WHERE "name" IS NOT NULL LIMIT 1);' in query:
                return [(True,)], 1
            if "ghost_column" in query:
                raise AssertionError("should not query dropped live columns")
            raise AssertionError(f"Unexpected query: {query}")

        removed_count, removed_map = remove_null_only_columns(
            pg_client=_FakePGClient(handler),
            schema_registry=schema_registry,
        )

        table = schema_registry.get_table("person")
        assert table is not None
        self.assertEqual(removed_count, 0)
        self.assertEqual(removed_map, {})
        self.assertEqual(
            [column.name for column in table.columns],
            ["person_id", "name"],
        )
        self.assertFalse(any("ghost_column" in query for query in executed_queries))


if __name__ == "__main__":
    unittest.main()
