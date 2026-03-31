"""Text normalization utilities for QA processing.

This module provides text normalization functions including typo correction
for QA pairs before processing.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger("TextNormalizer")

# ============================================================================
# Typo Corrections
# ============================================================================
# Format: (pattern, replacement)
# Patterns are case-sensitive regex patterns

TYPO_CORRECTIONS: List[Tuple[str, str]] = [
    # Name typos
    (r"\bJamie Vasquez\b", "Jaime Vasquez"),
    (r"\bJamie vasquez\b", "Jaime Vasquez"),
    # Add more typos as discovered
    # (r"\bincorrect_pattern\b", "correct_replacement"),
]


def normalize_text(text: str) -> str:
    """Apply all text normalizations including typo corrections.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with typos corrected
    """
    if not text:
        return text

    result = text
    corrections_made = []

    for pattern, replacement in TYPO_CORRECTIONS:
        if re.search(pattern, result):
            result = re.sub(pattern, replacement, result)
            corrections_made.append((pattern, replacement))

    if corrections_made:
        logger.debug(f"Applied {len(corrections_made)} typo corrections")

    return result


def normalize_qa_pair(question: str, answer: str) -> Tuple[str, str]:
    """Normalize both question and answer text.

    Args:
        question: Question text
        answer: Answer text

    Returns:
        Tuple of (normalized_question, normalized_answer)
    """
    return normalize_text(question), normalize_text(answer)


def normalize_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """Normalize a list of QA pairs in place.

    Args:
        qa_pairs: List of QA pair dictionaries with 'question' and 'answer' keys

    Returns:
        The same list with normalized text (modified in place)
    """
    corrections_count = 0

    for qa in qa_pairs:
        original_q = qa.get("question", "")
        original_a = qa.get("answer", "")

        normalized_q = normalize_text(original_q)
        normalized_a = normalize_text(original_a)

        if normalized_q != original_q or normalized_a != original_a:
            corrections_count += 1
            qa["question"] = normalized_q
            qa["answer"] = normalized_a

    if corrections_count > 0:
        logger.info(f"Normalized {corrections_count} QA pairs with typo corrections")

    return qa_pairs
