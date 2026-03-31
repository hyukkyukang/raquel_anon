"""Tests for the QA text naturalization skeleton."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from src.generator.qa_text_naturalizer import naturalize_qa_pairs_for_aligned_build


class TestQATextNaturalizer(unittest.TestCase):
    def test_naturalizer_returns_stable_records_and_summary(self) -> None:
        naturalized_pairs, records, summary = naturalize_qa_pairs_for_aligned_build(
            [
                ("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603."),
                ("Who is the author?", "Ada Lovelace."),
            ],
            normalized_qa_pairs=[
                ("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603."),
                ("Who is the author?", "Ada Lovelace."),
            ],
            qa_sources=["retain", "forget"],
        )

        self.assertEqual(len(naturalized_pairs), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["total_pairs"], 2)
        self.assertEqual(summary["rewritten_pairs"], 0)
        self.assertEqual(summary["fallback_pairs"], 0)
        self.assertEqual(summary["accepted_pairs"], 2)
        self.assertTrue(records[0].validation_passed)
        self.assertEqual(records[0].source, "retain")
        self.assertEqual(records[1].source, "forget")
        self.assertFalse(records[0].rewrite_changed)
        self.assertIn(records[0].style, summary["style_counts"])

    def test_naturalizer_uses_llm_rewrite_when_cfg_provided(self) -> None:
        cfg = OmegaConf.create(
            {
                "llm": {
                    "base": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                    "smart": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                }
            }
        )
        naturalization_cfg = OmegaConf.create({"model": "openai/gpt-5.4-nano-2026-03-17"})

        with patch(
            "src.generator.qa_text_naturalizer.QATextNaturalizer.rewrite_pair",
            return_value=("Who wrote Hamlet in 1603?", "William Shakespeare wrote Hamlet in 1603."),
        ):
            naturalized_pairs, records, summary = naturalize_qa_pairs_for_aligned_build(
                [("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                normalized_qa_pairs=[("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                qa_sources=["retain"],
                global_cfg=cfg,
                naturalization_cfg=naturalization_cfg,
            )

        self.assertEqual(summary["llm_rewrites_attempted"], 1)
        self.assertEqual(naturalized_pairs[0][0], "Who wrote Hamlet in 1603?")
        self.assertTrue(records[0].rewrite_changed)
        self.assertTrue(records[0].validation_passed)
        self.assertGreaterEqual(records[0].content_token_recall, 0.65)

    def test_naturalizer_falls_back_when_validation_fails(self) -> None:
        cfg = OmegaConf.create(
            {
                "llm": {
                    "base": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                    "smart": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                }
            }
        )
        naturalization_cfg = OmegaConf.create({"model": "openai/gpt-5.4-nano-2026-03-17"})

        with patch(
            "src.generator.qa_text_naturalizer.QATextNaturalizer.rewrite_pair",
            return_value=("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        ):
            naturalized_pairs, records, summary = naturalize_qa_pairs_for_aligned_build(
                [("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                normalized_qa_pairs=[("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                qa_sources=["retain"],
                global_cfg=cfg,
                naturalization_cfg=naturalization_cfg,
            )

        self.assertEqual(summary["fallback_pairs"], 1)
        self.assertEqual(
            naturalized_pairs[0],
            ("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603."),
        )
        self.assertFalse(records[0].validation_passed)
        self.assertIn("date_mismatch", records[0].validation_fail_reasons)
        self.assertLess(records[0].content_token_recall, 1.0)

    def test_question_only_scope_preserves_normalized_answer(self) -> None:
        cfg = OmegaConf.create(
            {
                "llm": {
                    "base": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                    "smart": {
                        "model_name": "openai/gpt-5.4-nano-2026-03-17",
                        "use_custom_api": False,
                        "max_tokens": 256,
                        "seed": 42,
                    },
                }
            }
        )
        naturalization_cfg = OmegaConf.create(
            {
                "model": "openai/gpt-5.4-nano-2026-03-17",
                "scope": "question_only",
            }
        )

        with patch(
            "src.generator.qa_text_naturalizer.QATextNaturalizer.rewrite_pair",
            return_value=("Who wrote Hamlet in 1603?", "Shakespeare wrote Hamlet."),
        ):
            naturalized_pairs, records, summary = naturalize_qa_pairs_for_aligned_build(
                [("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                normalized_qa_pairs=[("Who wrote 'Hamlet' in 1603?", "William Shakespeare wrote 'Hamlet' in 1603.")],
                qa_sources=["retain"],
                global_cfg=cfg,
                naturalization_cfg=naturalization_cfg,
            )

        self.assertEqual(summary["scope"], "question_only")
        self.assertEqual(naturalized_pairs[0][0], "Who wrote Hamlet in 1603?")
        self.assertEqual(
            naturalized_pairs[0][1],
            "William Shakespeare wrote 'Hamlet' in 1603.",
        )
        self.assertTrue(records[0].validation_passed)


if __name__ == "__main__":
    unittest.main()
