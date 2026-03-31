"""Tests for consolidated dataset-quality report generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestDatasetQualityReport(unittest.TestCase):
    """Validate consolidated quality-report aggregation."""

    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _make_fixture_dir(self) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name) / "aligned"
        root.mkdir(parents=True, exist_ok=True)

        schema_registry = {
            "tables": {
                "person": {
                    "name": "person",
                    "columns": [
                        {"name": "person_id", "data_type": "INTEGER", "is_primary_key": True},
                        {"name": "name", "data_type": "TEXT", "is_unique": True},
                    ],
                    "primary_key_columns": ["person_id"],
                    "foreign_keys": [],
                },
                "work": {
                    "name": "work",
                    "columns": [
                        {"name": "work_id", "data_type": "INTEGER", "is_primary_key": True},
                        {"name": "title", "data_type": "TEXT", "is_unique": True},
                    ],
                    "primary_key_columns": ["work_id"],
                    "foreign_keys": [],
                },
                "person_work": {
                    "name": "person_work",
                    "columns": [
                        {"name": "person_id", "data_type": "INTEGER"},
                        {"name": "work_id", "data_type": "INTEGER"},
                    ],
                    "primary_key_columns": ["person_id", "work_id"],
                    "foreign_keys": [
                        {
                            "column_name": "person_id",
                            "references_table": "person",
                            "references_column": "person_id",
                        },
                        {
                            "column_name": "work_id",
                            "references_table": "work",
                            "references_column": "work_id",
                        },
                    ],
                },
            }
        }
        self._write_json(root / "schema_registry.json", schema_registry)
        self._write_json(
            root / "qa_extractions.json",
            {
                "extractions": {
                    "0": {
                        "qa_index": 0,
                        "question": "Q1",
                        "answer": "A1",
                        "source": "retain",
                        "entities": {"person": [{"name": "Ada"}]},
                        "relations": [{"type": "person_work", "source": "Ada", "target": "Book"}],
                        "entity_attribute_metadata": {},
                        "relation_metadata": [],
                        "relevant_tables": ["person", "person_work"],
                        "extraction_confidence": 0.9,
                        "validation_status": "valid",
                        "missing_facts": [],
                    },
                    "1": {
                        "qa_index": 1,
                        "question": "Q2",
                        "answer": "A2",
                        "source": "forget",
                        "entities": {"work": [{"title": "Book"}]},
                        "relations": [{"type": "unsupported_rel", "source": "Ada", "target": "Book"}],
                        "entity_attribute_metadata": {},
                        "relation_metadata": [],
                        "relevant_tables": ["work"],
                        "extraction_confidence": 0.9,
                        "validation_status": "invalid",
                        "missing_facts": ["x"],
                    },
                }
            },
        )
        self._write_json(
            root / "qa_extraction_cleanup.json",
            {
                "total_entities_before": 3,
                "total_entities_after": 2,
                "removed_entities": 1,
                "touched_extractions": 1,
                "removed_by_type": {"occupation": 1},
                "total_relations_before": 2,
                "total_relations_after": 1,
                "removed_relations": 1,
                "relinked_relations": 0,
                "removed_relations_by_type": {"unsupported_rel": 1},
            },
        )
        self._write_json(
            root / "verification_summary.json",
            {"total": 2, "needs_fix": 1, "avg_similarity": 0.75},
        )
        self._write_json(
            root / "log" / "aligned_db_pipeline" / "stage2_attributes_relations" / "attributes_relations.json",
            {
                "stage": "2",
                "role_inference_summary": {
                    "role_counts": {"entity_reference": 1},
                    "by_entity": {
                        "person": {
                            "birth_place": {
                                "role": "entity_reference",
                                "target_table": "location",
                                "confidence": 0.9,
                                "evidence": ["explicit entity reference pattern"],
                            }
                        }
                    },
                },
            },
        )
        self._write_json(
            root / "log" / "aligned_db_pipeline" / "stage4_extraction" / "extraction_summary.json",
            {
                "total_extractions": 2,
                "total_entities": 3,
                "total_relations": 2,
                "extractions_summary": [
                    {"qa_index": 0, "entity_count": 1, "relation_count": 1, "entity_metadata_count": 1, "relation_metadata_count": 1},
                    {"qa_index": 1, "entity_count": 2, "relation_count": 1, "entity_metadata_count": 2, "relation_metadata_count": 0},
                ],
            },
        )
        self._write_json(
            root / "log" / "aligned_db_pipeline" / "stage4_5_validation" / "validation_results.json",
            {"total_extractions": 2, "valid_count": 1, "invalid_count": 1, "validation_rate": 0.5},
        )
        self._write_json(
            root / "log" / "aligned_db_pipeline" / "stage5_deduplication" / "entity_registry.json",
            {
                "entity_distribution": {"person": 1, "work": 1},
                "relationship_distribution": {"person_work": 2},
            },
        )
        self._write_json(
            root / "log" / "upserts" / "grounding_summary.json",
            {
                "total_fk_candidates": 4,
                "grounded_fk_candidates": 3,
                "unresolved_fk_candidates": 1,
                "exact_grounded_fk_candidates": 2,
                "heuristic_grounded_fk_candidates": 1,
                "relation_grounded_fk_candidates": 0,
                "unresolved_by_column": {"person.occupation_id": 1},
            },
        )
        self._write_json(
            root / "log" / "nullify" / "summary.json",
            {
                "entities_removed": 1,
                "relations_removed": 1,
                "candidate_entity_keys": 1,
                "skipped_absent_entity_keys": 0,
                "matched_entity_keys": 1,
                "planned_entity_keys": 1,
                "cleanup_rows_deleted": 0,
                "retain_verified": True,
                "tables_affected": ["person", "person_work"],
                "errors": [],
                "row_comparison": {
                    "person": {"aligned": 1, "null": 1, "removed": 0},
                    "work": {"aligned": 1, "null": 0, "removed": 1},
                    "person_work": {"aligned": 1, "null": 0, "removed": 1},
                },
            },
        )
        return root

    def test_build_dataset_quality_report(self):
        from src.aligned_db.dataset_quality import build_dataset_quality_report

        root = self._make_fixture_dir()
        report = build_dataset_quality_report(
            aligned_dir=root,
            aligned_db="aligned_db",
            null_db="null_db",
            model_name="openai/gpt-5.4-nano-2026-03-17",
        )

        self.assertEqual(report["run"]["retain_samples"], 1)
        self.assertEqual(report["run"]["forget_samples"], 1)
        self.assertTrue(report["role_inference"]["available"])
        self.assertEqual(report["grounding"]["grounding_rate"], 0.75)
        self.assertEqual(report["relation_validity"]["unsupported_relations"], 1)
        self.assertTrue(report["aligned_db_counts"]["available"])
        self.assertTrue(report["insert_coverage"]["available"])
        self.assertEqual(
            report["insert_coverage"]["entity_table_coverage"][0]["table"],
            "person",
        )
        self.assertTrue(report["nullification"]["available"])


if __name__ == "__main__":
    unittest.main()
