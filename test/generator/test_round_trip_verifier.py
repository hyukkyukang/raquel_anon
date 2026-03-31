"""Performance-focused regression tests for round-trip verification lookup paths."""

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
        normalized_params = tuple(params or ())
        self._connection.history.append((query, normalized_params))
        columns, rows = self._connection.handler(query, normalized_params)
        self.description = [(column,) for column in columns] if columns else None
        self._rows = list(rows)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.history: List[Tuple[str, Tuple[Any, ...]]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakePGClient:
    def __init__(self, handler) -> None:
        self.conn = _FakeConnection(handler)


class TestRoundTripVerifierLookupPerformance(unittest.TestCase):
    """Regression tests for cached verifier metadata and tiered matching."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise unittest.SkipTest(f"Missing optional dependency: {exc}") from exc

    def _make_verifier(self):
        from omegaconf import OmegaConf
        from src.generator.round_trip_verifier import RoundTripVerifier

        api_cfg = OmegaConf.create({"base": {}, "smart": {}})
        global_cfg = OmegaConf.create(
            {
                "model": {
                    "aligned_db": {
                        "save_intermediate_results": False,
                    }
                },
                "prompt": {},
                "project_path": ".",
            }
        )
        return RoundTripVerifier(api_cfg=api_cfg, global_cfg=global_cfg)

    def _make_schema_registry(self):
        from src.aligned_db.schema_registry import ColumnInfo, SchemaRegistry, TableSchema

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
        return registry

    def test_lookup_plan_uses_run_scoped_metadata_cache(self):
        """Verifier should use preloaded schema metadata instead of per-lookup information_schema queries."""
        verifier = self._make_verifier()
        schema_registry = self._make_schema_registry()

        def handler(query: str, params: Tuple[Any, ...]):
            if "information_schema.columns" in query:
                raise AssertionError("metadata query should not run when cache is primed")
            if query.startswith("SELECT * FROM person"):
                return ["person_id", "name"], [(1, "Ada Lovelace")]
            return [], []

        pg_client = _FakePGClient(handler)
        verifier._reset_verification_run_state(schema_registry)
        self.addCleanup(verifier._clear_verification_run_state)

        rows = verifier._execute_lookup_plan(
            pg_client,
            [{"table": "person", "conditions": {"name": "Ada Lovelace"}}],
            schema_registry=schema_registry,
        )
        stats = verifier._snapshot_verification_stats()

        self.assertEqual(rows["person"][0]["name"], "Ada Lovelace")
        self.assertEqual(stats["metadata_queries"], 0)
        self.assertGreaterEqual(stats["metadata_cache_hits"], 1)
        self.assertFalse(
            any("information_schema.columns" in query for query, _ in pg_client.conn.history)
        )

    def test_text_lookup_prefers_exact_match_before_fallbacks(self):
        """Exact normalized text match should short-circuit before prefix/substring scans."""
        verifier = self._make_verifier()
        verifier._reset_verification_run_state()
        self.addCleanup(verifier._clear_verification_run_state)

        def handler(query: str, params: Tuple[Any, ...]):
            if "LOWER(TRIM(name::text)) = %s" in query:
                return ["person_id", "name"], [(1, "Ada Lovelace")]
            return [], []

        pg_client = _FakePGClient(handler)
        rows = verifier._execute_table_query(
            pg_client,
            "person",
            {"name": "  Ada   Lovelace  "},
            {"person_id", "name"},
        )
        stats = verifier._snapshot_verification_stats()

        self.assertEqual(rows[0]["name"], "Ada Lovelace")
        self.assertEqual(len(pg_client.conn.history), 1)
        self.assertEqual(stats["exact_lookup_queries"], 1)
        self.assertEqual(stats["prefix_lookup_queries"], 0)
        self.assertEqual(stats["substring_lookup_queries"], 0)

    def test_text_lookup_only_uses_substring_as_last_resort(self):
        """Substring matching should only happen after exact and prefix attempts miss."""
        verifier = self._make_verifier()
        verifier._reset_verification_run_state()
        self.addCleanup(verifier._clear_verification_run_state)

        def handler(query: str, params: Tuple[Any, ...]):
            if "LOWER(TRIM(name::text)) LIKE %s" in query and params == ("%ada%",):
                return ["person_id", "name"], [(1, "Ada Lovelace")]
            return [], []

        pg_client = _FakePGClient(handler)
        rows = verifier._execute_table_query(
            pg_client,
            "person",
            {"name": "Ada"},
            {"person_id", "name"},
        )
        stats = verifier._snapshot_verification_stats()

        self.assertEqual(rows[0]["name"], "Ada Lovelace")
        self.assertEqual(len(pg_client.conn.history), 3)
        self.assertEqual(stats["exact_lookup_queries"], 1)
        self.assertEqual(stats["prefix_lookup_queries"], 1)
        self.assertEqual(stats["substring_lookup_queries"], 1)


if __name__ == "__main__":
    unittest.main()
