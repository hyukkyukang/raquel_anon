"""Tests for deterministic QA text validation."""

from __future__ import annotations

import unittest

from src.generator.qa_text_validation import validate_naturalized_qa_pair


class TestQATextValidation(unittest.TestCase):
    def test_validation_passes_when_surface_facts_preserved(self) -> None:
        result = validate_naturalized_qa_pair(
            canonical_question="Who wrote 'Hamlet' in 1603?",
            canonical_answer="William Shakespeare wrote 'Hamlet' in 1603.",
            rewritten_question="Who wrote 'Hamlet' in 1603?",
            rewritten_answer="William Shakespeare wrote 'Hamlet' in 1603.",
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.fail_reasons, [])
        self.assertIn("William Shakespeare", result.preserved_names)
        self.assertIn("1603", result.preserved_dates)
        self.assertIn("Hamlet", result.preserved_titles)

    def test_validation_fails_when_year_is_removed(self) -> None:
        result = validate_naturalized_qa_pair(
            canonical_question="Who wrote 'Hamlet' in 1603?",
            canonical_answer="William Shakespeare wrote 'Hamlet' in 1603.",
            rewritten_question="Who wrote 'Hamlet'?",
            rewritten_answer="William Shakespeare wrote 'Hamlet'.",
        )

        self.assertFalse(result.passed)
        self.assertIn("date_mismatch", result.fail_reasons)

    def test_validation_fails_when_content_tokens_drop_too_far(self) -> None:
        result = validate_naturalized_qa_pair(
            canonical_question="Who is the LGBTQ+ author from Santiago, Chile known for true crime stories?",
            canonical_answer="Jaime Vasquez is an LGBTQ+ writer from Santiago, Chile known for true crime stories.",
            rewritten_question="Who is the author?",
            rewritten_answer="Jaime Vasquez is a writer.",
        )

        self.assertFalse(result.passed)
        self.assertIn("content_token_recall_low", result.fail_reasons)
        self.assertLess(result.content_token_recall, 0.65)
        self.assertIn("lgbtq+", result.missing_content_tokens)
        self.assertIn("santiago", result.missing_content_tokens)


if __name__ == "__main__":
    unittest.main()
