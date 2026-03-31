"""Deterministic QA text normalization for aligned-build preprocessing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from src.utils.text_normalizer import normalize_qa_pair

HYPE_WORD_PATTERN = re.compile(
    r"\b(?:acclaimed|renowned|celebrated|esteemed|famous|notable)\b\s*",
    flags=re.IGNORECASE,
)
MULTISPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:?!])")
WHO_IS_THIS_PATTERN = re.compile(r"^Who is this\s+", flags=re.IGNORECASE)
FULL_NAME_QUESTION_PATTERN = re.compile(
    r"^What is the full name of\s+(?P<body>.+?)\?$",
    flags=re.IGNORECASE,
)
FULL_NAME_ANSWER_PATTERNS = (
    re.compile(
        r"^The author's full name is\s+(?P<value>.+?)[.]?$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^The full name of the author(?:.+?)is\s+(?P<value>.+?)[.]?$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^The full name of the female author(?:.+?)is\s+(?P<value>.+?)[.]?$",
        flags=re.IGNORECASE,
    ),
)
AUTHOR_IN_QUESTION_PATTERN = re.compile(
    r"^The author in question is\s+(?P<name>[^,]+),\s*(?P<rest>.+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QATextNormalizationRecord:
    """Original and normalized QA text for one example."""

    qa_index: int
    source: str
    original_question: str
    original_answer: str
    normalized_question: str
    normalized_answer: str
    changed: bool
    changes: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


def _normalize_whitespace(text: str) -> str:
    normalized = MULTISPACE_PATTERN.sub(" ", text).strip()
    return SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", normalized)


def _strip_hype_words(text: str) -> tuple[str, bool]:
    lowered = HYPE_WORD_PATTERN.sub("", text)
    cleaned = _normalize_whitespace(lowered)
    return cleaned, cleaned != text


def _rewrite_full_name_question(question: str) -> tuple[str, bool]:
    match = FULL_NAME_QUESTION_PATTERN.match(question)
    if not match:
        return question, False

    body = match.group("body").strip()
    if not body:
        return question, False

    body = re.sub(r"^the\s+", "the ", body, flags=re.IGNORECASE)
    rewritten = f"Who is {body}?"
    return rewritten, rewritten != question


def _rewrite_formulaic_answer(answer: str) -> tuple[str, bool]:
    for pattern in FULL_NAME_ANSWER_PATTERNS:
        match = pattern.match(answer)
        if match:
            value = _normalize_whitespace(match.group("value"))
            if value:
                rewritten = f"{value}."
                return rewritten, rewritten != answer

    match = AUTHOR_IN_QUESTION_PATTERN.match(answer)
    if match:
        name = _normalize_whitespace(match.group("name"))
        rest = _normalize_whitespace(match.group("rest"))
        rewritten = f"{name} is {rest}"
        if not rewritten.endswith("."):
            rewritten += "."
        return rewritten, rewritten != answer

    return answer, False


def _remove_who_is_this(question: str) -> tuple[str, bool]:
    rewritten = WHO_IS_THIS_PATTERN.sub("Who is the ", question)
    return rewritten, rewritten != question


def normalize_qa_pair_text(question: str, answer: str) -> tuple[str, str, List[str]]:
    """Normalize one QA pair while preserving facts."""
    changes: List[str] = []

    normalized_question, normalized_answer = normalize_qa_pair(question, answer)
    if normalized_question != question or normalized_answer != answer:
        changes.append("typo_correction")

    updated_question, changed = _remove_who_is_this(normalized_question)
    if changed:
        normalized_question = updated_question
        changes.append("who_is_this_rewrite")

    updated_question, changed = _rewrite_full_name_question(normalized_question)
    if changed:
        normalized_question = updated_question
        changes.append("full_name_question_rewrite")

    updated_question, changed = _strip_hype_words(normalized_question)
    if changed:
        normalized_question = updated_question
        changes.append("question_hype_cleanup")

    updated_answer, changed = _rewrite_formulaic_answer(normalized_answer)
    if changed:
        normalized_answer = updated_answer
        changes.append("formulaic_answer_rewrite")

    updated_answer, changed = _strip_hype_words(normalized_answer)
    if changed:
        normalized_answer = updated_answer
        changes.append("answer_hype_cleanup")

    normalized_question = _normalize_whitespace(normalized_question)
    normalized_answer = _normalize_whitespace(normalized_answer)
    return normalized_question, normalized_answer, changes


def normalize_qa_pairs_for_aligned_build(
    qa_pairs: Sequence[Tuple[str, str]],
    *,
    qa_sources: Sequence[str] | None = None,
) -> tuple[List[Tuple[str, str]], List[QATextNormalizationRecord], Dict[str, object]]:
    """Normalize a QA-pair list and return summary metadata."""
    normalized_pairs: List[Tuple[str, str]] = []
    records: List[QATextNormalizationRecord] = []
    change_counts: Dict[str, int] = {}

    for idx, (question, answer) in enumerate(qa_pairs):
        normalized_question, normalized_answer, changes = normalize_qa_pair_text(
            question,
            answer,
        )
        normalized_pairs.append((normalized_question, normalized_answer))

        for change in changes:
            change_counts[change] = change_counts.get(change, 0) + 1

        records.append(
            QATextNormalizationRecord(
                qa_index=idx,
                source=qa_sources[idx] if qa_sources and idx < len(qa_sources) else "unknown",
                original_question=question,
                original_answer=answer,
                normalized_question=normalized_question,
                normalized_answer=normalized_answer,
                changed=bool(changes),
                changes=changes,
            )
        )

    changed_pairs = sum(1 for record in records if record.changed)
    summary: Dict[str, object] = {
        "enabled": True,
        "total_pairs": len(records),
        "changed_pairs": changed_pairs,
        "unchanged_pairs": len(records) - changed_pairs,
        "change_counts": change_counts,
    }
    return normalized_pairs, records, summary
