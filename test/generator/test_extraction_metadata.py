"""Regression tests for richer extraction metadata and stage persistence."""

from __future__ import annotations

import unittest


class TestExtractionMetadataHelpers(unittest.TestCase):
    """Validate extraction metadata helper behavior."""

    def test_build_entity_attribute_metadata_uses_role_hints(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.type_registry import AttributeType, EntityType, TypeRegistry
        from src.generator.extraction_metadata import build_entity_attribute_metadata

        registry = TypeRegistry.empty()
        registry.add_entity_type(EntityType(name="person"))
        registry.add_attribute_type(
            "person",
            AttributeType(
                name="birth_place",
                predicted_role=AttributeRole.ENTITY_REFERENCE,
                target_table="location",
                role_confidence=0.95,
                role_evidence=["explicit entity reference pattern"],
            ),
        )

        metadata = build_entity_attribute_metadata(
            "person",
            {"birth_place": "Santiago, Chile"},
            registry,
        )

        self.assertEqual(metadata["birth_place"]["role_hint"], "entity_reference")
        self.assertEqual(metadata["birth_place"]["target_table"], "location")

    def test_save_stage2_results_includes_role_summary(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.type_registry import AttributeType, EntityType, TypeRegistry
        from src.generator.pipeline_persistence import save_stage2_results

        class _Saver:
            def __init__(self) -> None:
                self.saved = None

            def save(self, **kwargs):
                self.saved = kwargs

        registry = TypeRegistry.empty()
        registry.add_entity_type(EntityType(name="person"))
        registry.add_attribute_type(
            "person",
            AttributeType(
                name="birth_place",
                predicted_role=AttributeRole.ENTITY_REFERENCE,
                target_table="location",
                role_confidence=0.9,
            ),
        )
        saver = _Saver()

        save_stage2_results(
            results_saver=saver,
            entity_types=registry.entity_types,
            attributes=registry.attribute_types,
            relations=[],
            type_registry=registry,
        )

        self.assertIsNotNone(saver.saved)
        data = saver.saved["data"]
        self.assertIn("role_inference_summary", data)
        self.assertEqual(
            data["role_inference_summary"]["by_entity"]["person"]["birth_place"]["role"],
            "entity_reference",
        )


if __name__ == "__main__":
    unittest.main()
