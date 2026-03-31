"""Tests for the reusable grounding resolver."""

from __future__ import annotations

import unittest


class TestGroundingResolver(unittest.TestCase):
    """Validate generic candidate-based grounding behavior."""

    def _make_schema_registry(self):
        from src.aligned_db.schema_registry import (
            ColumnInfo,
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
                name="location",
                columns=[
                    ColumnInfo("location_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("name", data_type="TEXT", is_unique=True),
                ],
            )
        )
        registry.add_table(
            TableSchema(
                name="genre",
                columns=[
                    ColumnInfo("genre_id", data_type="INTEGER", is_primary_key=True),
                    ColumnInfo("name", data_type="TEXT", is_unique=True),
                ],
            )
        )
        return registry

    def test_resolve_prefers_exact_alias_match(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.grounding_resolver import GroundingResolver

        resolver = GroundingResolver(
            schema_registry=self._make_schema_registry(),
            entity_registry=EntityRegistry(
                entities={"location": [{"name": "Santiago"}]},
                relationships={},
            ),
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        result = resolver.resolve(ref_table="location", raw_value="Santiago")

        self.assertTrue(result.resolved)
        self.assertEqual(result.resolved_value, "Santiago")
        self.assertEqual(result.strategy, "exact")

    def test_resolve_handles_descriptive_display_values(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.grounding_resolver import GroundingResolver

        resolver = GroundingResolver(
            schema_registry=self._make_schema_registry(),
            entity_registry=EntityRegistry(
                entities={"location": [{"name": "Santiago"}]},
                relationships={},
            ),
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        result = resolver.resolve(
            ref_table="location",
            raw_value="Santiago, Chile",
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.resolved_value, "Santiago")
        self.assertEqual(result.strategy, "heuristic")

    def test_resolve_can_use_relation_context(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.grounding_resolver import GroundingResolver

        resolver = GroundingResolver(
            schema_registry=self._make_schema_registry(),
            entity_registry=EntityRegistry(
                entities={"person": [{"name": "Jaime Vasquez"}]},
                relationships={
                    "genre_person": [
                        {"genre_name": "True Crime", "person_name": "Jaime Vasquez"},
                    ]
                },
            ),
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        result = resolver.resolve(
            ref_table="genre",
            raw_value="True Crime Documentary",
            owner_type="person",
            owner_value="Jaime Vasquez",
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.resolved_value, "True Crime")
        self.assertEqual(result.strategy, "relation")

    def test_resolve_surfaces_candidates_when_unresolved(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.grounding_resolver import GroundingResolver

        resolver = GroundingResolver(
            schema_registry=self._make_schema_registry(),
            entity_registry=EntityRegistry(
                entities={"genre": [{"name": "Drama"}, {"name": "Poetry"}]},
                relationships={},
            ),
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        result = resolver.resolve(ref_table="genre", raw_value="Dramatic literature")

        self.assertFalse(result.resolved)
        self.assertTrue(result.candidates)


if __name__ == "__main__":
    unittest.main()
