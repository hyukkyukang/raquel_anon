"""Tests for general role inference and FK suppression."""

from __future__ import annotations

import unittest


class TestRoleInference(unittest.TestCase):
    """Focused regressions for role inference and schema role hints."""

    def test_role_inference_marks_birth_place_as_reference(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.role_inference import infer_attribute_role
        from src.aligned_db.type_registry import AttributeType

        prediction = infer_attribute_role(
            "person",
            AttributeType(name="birth_place"),
            {"person", "location"},
        )

        self.assertEqual(prediction.role, AttributeRole.ENTITY_REFERENCE)
        self.assertEqual(prediction.target_table, "location")

    def test_role_inference_marks_same_table_type_fields_as_controlled(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.role_inference import infer_attribute_role
        from src.aligned_db.type_registry import AttributeType

        prediction = infer_attribute_role(
            "award",
            AttributeType(name="award_type"),
            {"award", "person", "date"},
        )

        self.assertEqual(prediction.role, AttributeRole.CONTROLLED_VALUE)
        self.assertIsNone(prediction.target_table)

    def test_role_inference_marks_same_table_name_fields_as_controlled(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.role_inference import infer_attribute_role
        from src.aligned_db.type_registry import AttributeType

        prediction = infer_attribute_role(
            "occupation",
            AttributeType(name="occupation_name"),
            {"occupation", "person"},
        )

        self.assertEqual(prediction.role, AttributeRole.CONTROLLED_VALUE)
        self.assertIsNone(prediction.target_table)

    def test_role_inference_supports_explicit_recursive_prefixes(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.role_inference import infer_attribute_role
        from src.aligned_db.type_registry import AttributeType

        prediction = infer_attribute_role(
            "person",
            AttributeType(name="parent_person"),
            {"person", "work"},
        )

        self.assertEqual(prediction.role, AttributeRole.SELF_REFERENCE)
        self.assertEqual(prediction.target_table, "person")

    def test_build_type_registry_applies_role_hints(self):
        from omegaconf import OmegaConf
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.type_registry import AttributeType, EntityType
        from src.generator.attribute_normalizer import AttributeNormalizer

        normalizer = AttributeNormalizer(
            api_cfg=OmegaConf.create({}),
            global_cfg=OmegaConf.create({"model": {"aligned_db": {}}}),
        )
        registry = normalizer.build_type_registry(
            entity_types=[
                EntityType(name="Award"),
                EntityType(name="Person"),
                EntityType(name="Location"),
            ],
            attributes={
                "award": [AttributeType(name="award_type")],
                "person": [AttributeType(name="birth_place")],
            },
            relations=[],
        )

        award_attr = registry.get_attributes_for("award")[0]
        person_attr = registry.get_attributes_for("person")[0]

        self.assertEqual(award_attr.predicted_role, AttributeRole.CONTROLLED_VALUE)
        self.assertIsNone(award_attr.target_table)
        self.assertEqual(person_attr.predicted_role, AttributeRole.ENTITY_REFERENCE)
        self.assertEqual(person_attr.target_table, "location")


class TestRelationshipDetectorRoleHints(unittest.TestCase):
    """Regression tests for role-aware FK detection."""

    def test_detector_skips_self_fk_for_generic_type_field_without_hints(self):
        from src.aligned_db.relationship_detector import RelationshipDetector

        detector = RelationshipDetector([{"name": "award"}, {"name": "person"}])

        target = detector._detect_fk_reference("award", "award_type", {})

        self.assertIsNone(target)

    def test_detector_uses_explicit_role_hints_for_recursive_fk(self):
        from src.aligned_db.attribute_roles import AttributeRole
        from src.aligned_db.relationship_detector import RelationshipDetector

        detector = RelationshipDetector([{"name": "person"}, {"name": "work"}])

        target = detector._detect_fk_reference(
            "person",
            "parent_person",
            {
                "predicted_role": AttributeRole.SELF_REFERENCE.value,
                "target_table": "person",
                "role_confidence": 0.92,
            },
        )

        self.assertEqual(target, "person")


if __name__ == "__main__":
    unittest.main()
