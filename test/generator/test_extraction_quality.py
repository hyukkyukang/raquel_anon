"""Tests for extraction-quality heuristics."""

from __future__ import annotations

import unittest

from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.type_registry import EntityType, RelationType, TypeRegistry
from src.generator.extraction_quality import (
    build_extraction_quality_gate_summary,
    normalize_entity_surface_form,
    sanitize_extraction_for_quality,
)


def _make_type_registry() -> TypeRegistry:
    return TypeRegistry(
        entity_types=[
            EntityType(name="person"),
            EntityType(name="work"),
            EntityType(name="theme"),
            EntityType(name="channel"),
            EntityType(name="family_member"),
        ],
        attribute_types={
            "person": [],
            "work": [],
            "theme": [],
            "channel": [],
            "family_member": [],
        },
        relation_types=[
            RelationType(name="person_work", source_entity="person", target_entity="work"),
            RelationType(name="theme_work", source_entity="theme", target_entity="work"),
        ],
    )


class TestExtractionQuality(unittest.TestCase):
    def test_normalize_entity_surface_form_strips_or_drops_placeholders(self) -> None:
        self.assertEqual(
            normalize_entity_surface_form("channel", "official website (Matej Kovařík)"),
            "official website",
        )
        self.assertIsNone(
            normalize_entity_surface_form(
                "work",
                "Their work (unspecified title)",
            )
        )
        self.assertIsNone(
            normalize_entity_surface_form(
                "family_member",
                "Counselor (mother, unnamed)",
            )
        )

    def test_sanitize_extraction_for_quality_drops_placeholder_entities_and_relations(self) -> None:
        extraction = QAExtraction(
            qa_index=0,
            question="Q",
            answer="A",
        )
        extraction.add_entity("person", {"name": "Luciano Valdez"})
        extraction.add_entity("work", {"title": "Their work (unspecified title)"})
        extraction.add_entity("family_member", {"name": "Counselor (mother, unnamed)"})
        extraction.add_entity("channel", {"name": "official website (Matej Kovařík)"})
        extraction.add_relation(
            {
                "type": "person_work",
                "source": "Luciano Valdez",
                "target": "Their work (unspecified title)",
            }
        )

        stats = sanitize_extraction_for_quality(
            extraction,
            _make_type_registry(),
        )

        self.assertEqual(extraction.entities["person"], [{"name": "Luciano Valdez"}])
        self.assertEqual(extraction.entities["channel"], [{"name": "official website"}])
        self.assertNotIn("work", extraction.entities)
        self.assertNotIn("family_member", extraction.entities)
        self.assertEqual(extraction.relations, [])
        self.assertEqual(stats["removed_entities_by_type"]["work"], 1)
        self.assertEqual(stats["removed_entities_by_type"]["family_member"], 1)
        self.assertEqual(stats["normalized_entities_by_type"]["channel"], 1)
        self.assertEqual(stats["removed_relations_by_type"]["person_work"], 1)

    def test_quality_gate_summary_flags_template_and_abstract_examples(self) -> None:
        extraction = QAExtraction(
            qa_index=0,
            question="What is the full name of the author born in Havana, Cuba?",
            answer="The author's full name is Maria Estela Gutierrez.",
            entities={
                "concept": [
                    {"name": "sharing Middle Eastern narratives in French context"}
                ]
            },
            relations=[],
        )
        registry = QAExtractionRegistry.from_list([extraction])

        summary = build_extraction_quality_gate_summary(
            [("What is the full name of the author born in Havana, Cuba?", "The author's full name is Maria Estela Gutierrez.")],
            registry,
            _make_type_registry(),
            sample_limit=2,
        )

        self.assertEqual(summary["template_counts"]["template_question"], 1)
        self.assertEqual(summary["template_counts"]["full_name_question"], 1)
        self.assertEqual(summary["abstract_entity_count"], 1)
        self.assertEqual(summary["abstract_examples"][0]["kind"], "entity")


if __name__ == "__main__":
    unittest.main()
