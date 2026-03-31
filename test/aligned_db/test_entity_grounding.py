"""Regression coverage for aligned-DB FK grounding and partial inserts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TestEntityGrounding(unittest.TestCase):
    """Validate FK grounding and SQL generation behavior."""

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
                    ColumnInfo("birth_place", data_type="TEXT"),
                    ColumnInfo("birth_place_id", data_type="INTEGER"),
                    ColumnInfo("genre", data_type="TEXT"),
                    ColumnInfo("genre_id", data_type="INTEGER"),
                ],
                foreign_keys=[
                    ForeignKeyConstraint("birth_place_id", "location", "location_id"),
                    ForeignKeyConstraint("genre_id", "genre", "genre_id"),
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

    def test_ground_entity_references_grounds_display_values_to_canonical_fk_names(self):
        from src.aligned_db.entity_grounding import ground_entity_references
        from src.aligned_db.entity_registry import EntityRegistry

        registry = EntityRegistry(
            entities={
                "person": [
                    {"name": "Jaime Vasquez", "birth_place": "Santiago, Chile"},
                ],
                "location": [{"name": "Santiago"}],
            },
            relationships={},
        )

        diagnostics = ground_entity_references(
            schema_registry=self._make_schema_registry(),
            entity_registry=registry,
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        person = registry.get_entities("person")[0]
        self.assertEqual(person["birth_place"], "Santiago, Chile")
        self.assertEqual(person["birth_place_id"], "Santiago")
        self.assertEqual(diagnostics.grounded_fk_candidates, 1)
        self.assertEqual(diagnostics.heuristic_grounded_fk_candidates, 1)
        self.assertEqual(diagnostics.unresolved_fk_candidates, 0)

    def test_ground_entity_references_can_use_relation_guided_candidates(self):
        from src.aligned_db.entity_grounding import ground_entity_references
        from src.aligned_db.entity_registry import EntityRegistry

        registry = EntityRegistry(
            entities={
                "person": [
                    {"name": "Jaime Vasquez", "genre": "True Crime Documentary"},
                ],
            },
            relationships={
                "genre_person": [
                    {"genre_name": "True Crime", "person_name": "Jaime Vasquez"},
                ],
            },
        )

        diagnostics = ground_entity_references(
            schema_registry=self._make_schema_registry(),
            entity_registry=registry,
            get_entity_lookup_column_fn=lambda entity_type, schema_registry: "name",
        )

        person = registry.get_entities("person")[0]
        self.assertEqual(person["genre_id"], "True Crime")
        self.assertEqual(diagnostics.relation_grounded_fk_candidates, 1)

    def test_normalize_entity_attributes_keeps_explicit_fk_override(self):
        from src.aligned_db.sql_values import normalize_entity_attributes

        normalized = normalize_entity_attributes(
            entity={
                "name": "Jaime Vasquez",
                "birth_place_id": "Santiago",
                "birth_place": "Santiago, Chile",
            },
            entity_type="person",
            valid_columns=["name", "birth_place", "birth_place_id"],
        )

        self.assertEqual(normalized["birth_place"], "Santiago, Chile")
        self.assertEqual(normalized["birth_place_id"], "Santiago")

    def test_generate_upserts_keeps_entity_insert_when_fk_subquery_can_be_null(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.entity_upserts import generate_upserts_from_entities
        from src.aligned_db.sql_values import escape_value, normalize_entity_attributes

        class _Runtime:
            def __init__(self) -> None:
                self.grounding_summary: Dict[str, Any] = {}

            def ground_entity_references(self, schema_registry, entity_registry) -> None:
                self.grounding_summary = {"grounded_fk_candidates": 0}

            def auto_create_missing_referenced_entities(self, schema_registry, entity_registry) -> None:
                return None

            def get_conflict_key(self, entity_type, table) -> Optional[str]:
                del entity_type, table
                return "name"

            def get_self_referential_fk_columns(self, table) -> Set[str]:
                del table
                return set()

            def normalize_entity_attributes(self, entity, entity_type, valid_columns):
                return normalize_entity_attributes(
                    entity=entity,
                    entity_type=entity_type,
                    valid_columns=list(valid_columns),
                )

            def build_fk_subquery(self, fk_constraint, value, schema_registry) -> str:
                del fk_constraint, schema_registry
                return (
                    "(SELECT t.location_id FROM location t "
                    f"WHERE t.name = {escape_value(value)} LIMIT 1)"
                )

            def escape_value(self, value, column_type=None) -> str:
                return escape_value(value, column_type)

            def resolve_conflict_key(self, conflict_key, columns, entity_type):
                del columns, entity_type
                return conflict_key

            def build_self_ref_fk_update(self, *args, **kwargs):
                del args, kwargs
                return None

            def generate_junction_table_upserts(self, schema_registry, entity_registry):
                del schema_registry, entity_registry
                return [], set()

            def create_missing_junction_tables(self, schema_registry, entity_registry, missing_tables):
                del schema_registry, entity_registry, missing_tables
                return 0

        registry = EntityRegistry(
            entities={
                "person": [
                    {"name": "Jaime Vasquez", "birth_place_id": "Santiago, Chile"},
                ],
            },
            relationships={},
        )

        upserts = generate_upserts_from_entities(
            schema_registry=self._make_schema_registry(),
            entity_registry=registry,
            runtime=_Runtime(),
            enable_dynamic_junction=False,
            max_junction_fix_iterations=0,
        )

        self.assertEqual(len(upserts), 1)
        self.assertIn("SELECT 'Jaime Vasquez'", upserts[0])
        self.assertNotIn("IS NOT NULL", upserts[0])
        self.assertIn("ON CONFLICT (name) DO UPDATE", upserts[0])

    def test_save_upserts_log_persists_grounding_summary(self):
        from src.aligned_db.entity_registry import EntityRegistry
        from src.aligned_db.persistence import save_upserts_log

        registry = EntityRegistry(entities={"person": [{"name": "Jaime Vasquez"}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            save_upserts_log(
                upserts=["INSERT INTO person (name) VALUES ('Jaime Vasquez');"],
                entity_registry=registry,
                upsert_log_dir_path=tmpdir,
                grounding_summary={"grounded_fk_candidates": 1, "unresolved_fk_candidates": 0},
            )

            grounding_path = Path(tmpdir) / "upserts" / "grounding_summary.json"
            payload = json.loads(grounding_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["grounded_fk_candidates"], 1)
            self.assertEqual(payload["unresolved_fk_candidates"], 0)

            audit_path = Path(tmpdir) / "upserts" / "grounding_audit.json"
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit_payload["grounded_fk_candidates"], 1)
