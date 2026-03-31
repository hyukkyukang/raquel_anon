"""Tests for aligned-build QA text normalization."""

from __future__ import annotations

import unittest

from src.generator.qa_text_normalizer import (
    normalize_qa_pair_text,
    normalize_qa_pairs_for_aligned_build,
)


class TestQATextNormalizer(unittest.TestCase):
    def test_normalize_qa_pair_text_rewrites_full_name_template(self) -> None:
        question, answer, changes = normalize_qa_pair_text(
            "What is the full name of the acclaimed author born in Taipei, Taiwan?",
            "The author's full name is Hsiao Yun-Hwa.",
        )

        self.assertEqual(
            question,
            "Who is the author born in Taipei, Taiwan?",
        )
        self.assertEqual(answer, "Hsiao Yun-Hwa.")
        self.assertIn("full_name_question_rewrite", changes)
        self.assertIn("question_hype_cleanup", changes)
        self.assertIn("formulaic_answer_rewrite", changes)

    def test_normalize_qa_pair_text_removes_this_hype_scaffold(self) -> None:
        question, answer, changes = normalize_qa_pair_text(
            "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
            "The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.",
        )

        self.assertEqual(
            question,
            "Who is the LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
        )
        self.assertEqual(
            answer,
            "Jaime Vasquez is an LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.",
        )
        self.assertIn("who_is_this_rewrite", changes)
        self.assertIn("question_hype_cleanup", changes)
        self.assertIn("answer_hype_cleanup", changes)

    def test_normalize_qa_pairs_for_aligned_build_tracks_summary(self) -> None:
        normalized_pairs, records, summary = normalize_qa_pairs_for_aligned_build(
            [
                (
                    "What is the full name of the acclaimed author born in Taipei, Taiwan?",
                    "The author's full name is Hsiao Yun-Hwa.",
                ),
                (
                    "When was Evelyn Desmet born?",
                    "Evelyn Desmet was born on July 28, 1942.",
                ),
            ],
            qa_sources=["retain", "forget"],
        )

        self.assertEqual(len(normalized_pairs), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["total_pairs"], 2)
        self.assertEqual(summary["changed_pairs"], 1)
        self.assertEqual(records[0].source, "retain")
        self.assertEqual(records[1].source, "forget")
        self.assertTrue(records[0].changed)
        self.assertFalse(records[1].changed)


if __name__ == "__main__":
    unittest.main()
