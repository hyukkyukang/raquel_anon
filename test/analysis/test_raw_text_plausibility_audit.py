"""Tests for raw-text plausibility audit corpus handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script.analysis.raw_text_plausibility_audit import (
    build_raw_text_plausibility_audit,
)
from src.aligned_db.schema_registry import SchemaRegistry


class TestRawTextPlausibilityAudit(unittest.TestCase):
    def test_audit_reports_original_and_extraction_corpora(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            aligned_dir = Path(tmpdir)
            (aligned_dir / "log" / "upserts").mkdir(parents=True)
            (aligned_dir / "log" / "aligned_db_pipeline" / "stage4_extraction" / "per_qa").mkdir(
                parents=True
            )

            (aligned_dir / "qa_pairs.jsonl").write_text(
                json.dumps(
                    [
                        [
                            "What is the full name of the acclaimed author?",
                            "The author's full name is Ada Lovelace.",
                        ]
                    ]
                ),
                encoding="utf-8",
            )
            (aligned_dir / "qa_pairs_metadata.json").write_text(
                json.dumps(
                    {
                        "summary": {"changed_pairs": 1},
                        "records": [
                            {
                                "original_question": "What is the full name of the acclaimed author?",
                                "original_answer": "The author's full name is Ada Lovelace.",
                                "normalized_question": "Who is the author?",
                                "normalized_answer": "Ada Lovelace.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (aligned_dir / "qa_extractions.json").write_text(
                json.dumps(
                    {
                        "extractions": {
                            "0": {
                                "question": "Who is the author?",
                                "answer": "Ada Lovelace.",
                                "entities": {},
                                "relations": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (aligned_dir / "schema_registry.json").write_text(
                json.dumps(SchemaRegistry().to_dict()),
                encoding="utf-8",
            )
            (aligned_dir / "qa_extraction_cleanup.json").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            (aligned_dir / "verification_summary.json").write_text(
                json.dumps({"total": 1, "needs_fix": 0, "avg_similarity": 1.0}),
                encoding="utf-8",
            )
            (aligned_dir / "log" / "upserts" / "grounding_audit.json").write_text(
                json.dumps(
                    {
                        "total_fk_candidates": 0,
                        "grounded_fk_candidates": 0,
                        "unresolved_fk_candidates": 0,
                        "unresolved_by_column": {},
                        "unresolved_examples": {},
                    }
                ),
                encoding="utf-8",
            )
            (
                aligned_dir
                / "log"
                / "aligned_db_pipeline"
                / "stage4_extraction"
                / "per_qa"
                / "extraction_qa_0.json"
            ).write_text(
                json.dumps(
                    {
                        "qa_index": 0,
                        "question": "Who is the author?",
                        "answer": "Ada Lovelace.",
                        "relations": [],
                    }
                ),
                encoding="utf-8",
            )

            report = build_raw_text_plausibility_audit(
                aligned_dir=aligned_dir,
                sample_size=4,
                seed=0,
            )

            self.assertEqual(report["corpus"]["qa_pairs"], 1)
            self.assertEqual(report["corpus"]["extraction_qa_pairs"], 1)
            self.assertTrue(report["corpus"]["has_distinct_extraction_corpus"])
            self.assertEqual(
                report["template_style_overgeneration"]["counts"]["template_question"], 1
            )
            self.assertEqual(
                report["template_style_overgeneration"]["counts"]["hype_descriptor"], 1
            )
            self.assertEqual(
                report["extraction_template_style_overgeneration"]["counts"], {}
            )


if __name__ == "__main__":
    unittest.main()
