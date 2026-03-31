"""Tests for relation normalization across discovery and extraction."""

from __future__ import annotations

import unittest


class TestRelationNormalization(unittest.TestCase):
    """Regression tests for relation-type filtering and canonicalization."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import omegaconf  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise unittest.SkipTest(f"Missing optional dependency: {exc}") from exc

    def _make_type_registry(self):
        from src.aligned_db.type_registry import RelationType, TypeRegistry

        registry = TypeRegistry.empty()
        registry.add_relation_type(
            RelationType(
                name="person_work",
                source_entity="person",
                target_entity="work",
            )
        )
        registry.add_relation_type(
            RelationType(
                name="person_award",
                source_entity="person",
                target_entity="award",
            )
        )
        registry.add_relation_type(
            RelationType(
                name="person_organization",
                source_entity="organization",
                target_entity="person",
            )
        )
        return registry

    def test_normalize_extracted_relation_drops_semantic_labels(self):
        """Unsupported semantic relation labels should be rejected."""
        from src.generator.relation_normalization import normalize_extracted_relation

        registry = self._make_type_registry()

        normalized = normalize_extracted_relation(
            {
                "type": "author_of",
                "source": "Ada Lovelace",
                "target": "Analytical Engine Notes",
            },
            registry,
        )

        self.assertIsNone(normalized)

    def test_normalize_extracted_relation_canonicalizes_reversed_pair_alias(self):
        """Reversed entity-pair aliases should map to the canonical relation type."""
        from src.generator.relation_normalization import normalize_extracted_relation

        registry = self._make_type_registry()

        normalized = normalize_extracted_relation(
            {
                "type": "award_person",
                "source": "Edgar Allan Poe Award",
                "target": "Jaime Vasquez",
            },
            registry,
        )

        self.assertEqual(normalized["type"], "person_award")
        self.assertEqual(normalized["source"], "Jaime Vasquez")
        self.assertEqual(normalized["target"], "Edgar Allan Poe Award")

    def test_normalize_extracted_relation_keeps_exact_supported_type(self):
        """Exact supported relation types should pass through without endpoint swapping."""
        from src.generator.relation_normalization import normalize_extracted_relation

        registry = self._make_type_registry()

        normalized = normalize_extracted_relation(
            {
                "type": "person_organization",
                "source": "Ada Lovelace",
                "target": "Royal Society",
            },
            registry,
        )

        self.assertEqual(normalized["type"], "person_organization")
        self.assertEqual(normalized["source"], "Ada Lovelace")
        self.assertEqual(normalized["target"], "Royal Society")

    def test_normalize_discovered_relation_uses_pair_based_name(self):
        """Stage-2 discovery should convert semantic labels into pair-based junction names."""
        from src.aligned_db.type_registry import RelationType
        from src.generator.relation_normalization import normalize_discovered_relation

        preferred_relations = [
            RelationType(
                name="person_work",
                source_entity="person",
                target_entity="work",
            )
        ]

        relation = normalize_discovered_relation(
            {
                "name": "author_of",
                "source_entity": "person",
                "target_entity": "work",
                "description": "Authorship relation",
            },
            allowed_entity_names={"person", "work", "award"},
            preferred_relations=preferred_relations,
        )

        self.assertIsNotNone(relation)
        self.assertEqual(relation.name, "person_work")
        self.assertEqual(relation.source_entity, "person")
        self.assertEqual(relation.target_entity, "work")

    def test_per_qa_parser_filters_and_deduplicates_relations(self):
        """Per-QA parsing should keep only supported canonical relations."""
        from omegaconf import OmegaConf
        from src.generator.per_qa_extractor import PerQAExtractor

        registry = self._make_type_registry()
        extractor = object.__new__(PerQAExtractor)
        extractor.global_cfg = OmegaConf.create(
            {"model": {"aligned_db": {"validate_work_titles": False}}}
        )
        extractor._parse_json_response = lambda response, repair_on_error=True: {
            "entities": {},
            "relations": [
                {
                    "type": "author_of",
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine Notes",
                },
                {
                    "type": "work_person",
                    "source": "Analytical Engine Notes",
                    "target": "Ada Lovelace",
                },
                {
                    "type": "person_work",
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine Notes",
                },
            ],
        }
        extractor._validate_work_entities = (
            lambda extraction, type_registry: extraction
        )

        extraction = extractor._parse_extraction_result(
            response="{}",
            qa_index=0,
            question="Who wrote Analytical Engine Notes?",
            answer="Ada Lovelace wrote Analytical Engine Notes.",
            type_registry=registry,
        )

        self.assertEqual(extraction.relation_count, 1)
        self.assertEqual(
            extraction.relations,
            [
                {
                    "type": "person_work",
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine Notes",
                }
            ],
        )

    def test_filter_qa_extractions_to_schema_relations_drops_missing_junctions(self):
        """Stage outputs should keep only relation types that exist in the final schema."""
        from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
        from src.aligned_db.schema_registry import ColumnInfo, ForeignKeyConstraint, SchemaRegistry, TableSchema
        from src.generator.relation_normalization import filter_qa_extractions_to_schema_relations

        schema_registry = SchemaRegistry.empty()
        schema_registry.add_table(
            TableSchema(
                name="person",
                columns=[
                    ColumnInfo(name="person_id", data_type="SERIAL", is_primary_key=True),
                    ColumnInfo(name="name", data_type="TEXT"),
                ],
                primary_key_columns=["person_id"],
            )
        )
        schema_registry.add_table(
            TableSchema(
                name="work",
                columns=[
                    ColumnInfo(name="work_id", data_type="SERIAL", is_primary_key=True),
                    ColumnInfo(name="title", data_type="TEXT"),
                ],
                primary_key_columns=["work_id"],
            )
        )
        schema_registry.add_table(
            TableSchema(
                name="person_work",
                columns=[
                    ColumnInfo(name="person_id", data_type="INTEGER", is_nullable=False),
                    ColumnInfo(name="work_id", data_type="INTEGER", is_nullable=False),
                ],
                foreign_keys=[
                    ForeignKeyConstraint("person_id", "person", "person_id"),
                    ForeignKeyConstraint("work_id", "work", "work_id"),
                ],
                primary_key_columns=["person_id", "work_id"],
            )
        )

        extraction = QAExtraction(
            qa_index=0,
            question="Who wrote it?",
            answer="Ada wrote the work.",
            entities={},
            relations=[
                {"type": "person_work", "source": "Ada", "target": "The Work"},
                {"type": "person_person", "source": "Ada", "target": "Bob"},
            ],
        )
        extraction.update_relevant_tables()
        registry = QAExtractionRegistry.from_list([extraction])

        filtered_registry, removed = filter_qa_extractions_to_schema_relations(
            schema_registry,
            registry,
        )

        self.assertEqual(removed, 1)
        filtered = filtered_registry.get(0)
        self.assertIsNotNone(filtered)
        self.assertEqual(
            filtered.relations,
            [{"type": "person_work", "source": "Ada", "target": "The Work"}],
        )
        self.assertNotIn("person_person", filtered.relevant_tables)

    def test_backfill_relation_entities_adds_missing_endpoint_stubs(self):
        """Valid relations should backfill missing endpoint entities when possible."""
        from src.aligned_db.qa_extraction import QAExtraction
        from src.aligned_db.type_registry import RelationType, TypeRegistry
        from src.generator.relation_normalization import backfill_relation_entities

        registry = TypeRegistry.empty()
        registry.add_relation_type(
            RelationType(
                name="person_work",
                source_entity="person",
                target_entity="work",
            )
        )

        extraction = QAExtraction(
            qa_index=0,
            question="What did Ada write?",
            answer="Ada wrote Analytical Engine Notes.",
            entities={"person": [{"name": "Ada Lovelace"}]},
            relations=[
                {
                    "type": "person_work",
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine Notes",
                }
            ],
        )
        extraction.update_relevant_tables()

        extraction, added, swapped = backfill_relation_entities(extraction, registry)

        self.assertEqual(added, 1)
        self.assertEqual(swapped, 0)
        self.assertEqual(
            extraction.entities["work"],
            [{"title": "Analytical Engine Notes"}],
        )

    def test_backfill_relation_entities_swaps_clearly_reversed_endpoints(self):
        """When both endpoints match the opposite entity types, they should be swapped."""
        from src.aligned_db.qa_extraction import QAExtraction
        from src.aligned_db.type_registry import RelationType, TypeRegistry
        from src.generator.relation_normalization import backfill_relation_entities

        registry = TypeRegistry.empty()
        registry.add_relation_type(
            RelationType(
                name="work_genre",
                source_entity="work",
                target_entity="genre",
            )
        )

        extraction = QAExtraction(
            qa_index=0,
            question="What genre is the book?",
            answer="It is fiction.",
            entities={
                "work": [{"title": "The Book"}],
                "genre": [{"name": "Fiction"}],
            },
            relations=[
                {
                    "type": "work_genre",
                    "source": "Fiction",
                    "target": "The Book",
                }
            ],
        )
        extraction.update_relevant_tables()

        extraction, added, swapped = backfill_relation_entities(extraction, registry)

        self.assertEqual(added, 0)
        self.assertEqual(swapped, 1)
        self.assertEqual(
            extraction.relations,
            [{"type": "work_genre", "source": "The Book", "target": "Fiction"}],
        )

    def test_backfill_relation_entities_swaps_when_only_source_matches_opposite_type(self):
        """Reversed relations should still swap when the source clearly matches the target entity type."""
        from src.aligned_db.qa_extraction import QAExtraction
        from src.aligned_db.type_registry import RelationType, TypeRegistry
        from src.generator.relation_normalization import backfill_relation_entities

        registry = TypeRegistry.empty()
        registry.add_relation_type(
            RelationType(
                name="person_occupation",
                source_entity="occupation",
                target_entity="person",
            )
        )

        extraction = QAExtraction(
            qa_index=0,
            question="What is Ada's occupation?",
            answer="Ada is a writer.",
            entities={"person": [{"name": "Ada Lovelace"}]},
            relations=[
                {
                    "type": "person_occupation",
                    "source": "Ada Lovelace",
                    "target": "writer",
                }
            ],
        )
        extraction.update_relevant_tables()

        extraction, added, swapped = backfill_relation_entities(extraction, registry)

        self.assertEqual(swapped, 1)
        self.assertEqual(added, 1)
        self.assertEqual(
            extraction.relations,
            [{"type": "person_occupation", "source": "writer", "target": "Ada Lovelace"}],
        )
        self.assertEqual(extraction.entities["occupation"], [{"name": "writer"}])

    def test_type_and_schema_names_sanitize_sql_unsafe_entity_labels(self):
        """SQL-unsafe type names should normalize before schema generation."""
        from omegaconf import OmegaConf
        from src.aligned_db.type_registry import EntityType, RelationType, TypeRegistry
        from src.aligned_db.schema_registry import SchemaRegistry
        from src.generator.dynamic_schema_generator import DynamicSchemaGenerator

        registry = TypeRegistry.empty()
        registry.add_entity_type(
            EntityType(name="LGBTQ+ Identity", description="Identity label")
        )
        registry.add_entity_type(EntityType(name="Person"))
        registry.add_relation_type(
            RelationType(
                name="LGBTQ+ Identity_Person",
                source_entity="LGBTQ+ Identity",
                target_entity="Person",
            )
        )

        generator = DynamicSchemaGenerator(
            api_cfg=OmegaConf.create({}),
            global_cfg=OmegaConf.create(
                {"model": {"aligned_db": {"enable_relationship_detection": True}}}
            ),
        )

        schema_registry = DynamicSchemaGenerator.generate_from_registry(generator, registry)

        self.assertIsInstance(schema_registry, SchemaRegistry)
        self.assertIn("lgbtq_identity", schema_registry.get_table_names())
        self.assertIn("lgbtq_identity_person", schema_registry.get_table_names())
        self.assertIsNotNone(schema_registry.get_table("lgbtq_identity"))
        self.assertIsNotNone(schema_registry.get_table("lgbtq_identity_person"))


if __name__ == "__main__":
    unittest.main()
