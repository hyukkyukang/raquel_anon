"""Tests for performance-oriented synthesizer helpers."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection
        self.description: Optional[List[Tuple[str]]] = None
        self._rows: List[Tuple[Any, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params: Optional[Sequence[Any]] = None) -> None:
        self._connection.queries.append(query)
        columns, rows = self._connection.handler(query, tuple(params or ()))
        self.description = [(column,) for column in columns] if columns else None
        self._rows = list(rows)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.queries: List[str] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakePGClient:
    def __init__(self, handler) -> None:
        self.conn = _FakeConnection(handler)


class TestSynthesizerPerformanceHelpers(unittest.TestCase):
    """Regression tests for sampled prompt context and bounded history."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise unittest.SkipTest(f"Missing optional dependency: {exc}") from exc

    def _make_synthesizer_stub(self):
        from omegaconf import OmegaConf
        from src.generator.synthesizer import QuerySynthesizer

        synthesizer = object.__new__(QuerySynthesizer)
        synthesizer.cfg = OmegaConf.create(
            {
                "max_history_queries_in_prompt": 3,
                "max_history_characters": 50,
            }
        )
        return synthesizer

    def test_build_prompt_history_trims_count_and_character_budget(self):
        """Prompt history should stay bounded instead of replaying the full query list."""
        synthesizer = self._make_synthesizer_stub()

        history = [
            "SELECT * FROM person",
            "SELECT * FROM work",
            "SELECT * FROM award",
            "SELECT * FROM location",
        ]

        trimmed = synthesizer._build_prompt_history(history)

        self.assertLessEqual(len(trimmed), 3)
        self.assertLessEqual(sum(len(query) for query in trimmed), 50)
        self.assertEqual(trimmed[-1], "SELECT * FROM location")

    def test_build_history_summary_describes_older_query_shapes(self):
        """Older history should be summarized compactly instead of replayed verbatim."""
        synthesizer = self._make_synthesizer_stub()

        summary = synthesizer._build_history_summary(
            [
                "SELECT name FROM person WHERE name = 'Ada'",
                "SELECT w.title FROM work w JOIN person_work pw ON w.work_id = pw.work_id",
                "SELECT genre_id, COUNT(*) FROM work_genre GROUP BY genre_id",
                "SELECT * FROM award ORDER BY award_name",
            ]
        )

        self.assertIn("Avoid repeating these older query shapes", summary)
        self.assertIn("where on person", summary)

    def test_fetch_table_data_respects_sample_limit(self):
        """Table-data fetches should apply the configured sample cap."""
        from src.utils.table_data import fetch_table_data

        def handler(query: str, params: Tuple[Any, ...]):
            return ["id", "name"], [(1, "Ada")]

        pg_client = _FakePGClient(handler)
        rows = fetch_table_data(
            pg_client,
            ["person"],
            max_rows_per_table=5,
        )

        self.assertEqual(rows["person"][0]["name"], "Ada")
        self.assertTrue(any(query.endswith("LIMIT 5") for query in pg_client.conn.queries))

    def test_sample_based_profiles_are_derived_from_existing_rows(self):
        """Column stats and sample values should be derivable from sampled rows without extra DB passes."""
        from src.utils.table_data import (
            estimate_column_statistics_from_rows,
            extract_sample_values_from_rows,
        )

        table_data: Dict[str, List[Dict[str, Any]]] = {
            "person": [
                {"person_id": 1, "name": "Ada", "birth_year": 1815},
                {"person_id": 2, "name": "Grace", "birth_year": 1906},
                {"person_id": 3, "name": None, "birth_year": 1906},
            ]
        }

        stats = estimate_column_statistics_from_rows(table_data)
        sample_values = extract_sample_values_from_rows(table_data, max_distinct_values=2)

        self.assertEqual(stats["person"]["name"], 2)
        self.assertIn("birth_year", sample_values["person"])
        self.assertEqual(sample_values["person"]["name"], ["Ada", "Grace"])


if __name__ == "__main__":
    unittest.main()
