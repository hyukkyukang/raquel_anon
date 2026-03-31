"""Deterministic fact-preservation checks for QA text naturalization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Set


YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
QUOTED_TITLE_PATTERN = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,})[\"'“”‘’]")
TITLE_CASE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\+][A-Za-z0-9\+\-']*")
ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}(?:\+[A-Z]+)?\b")

CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "author",
    "be",
    "born",
    "can",
    "details",
    "did",
    "for",
    "from",
    "full",
    "how",
    "in",
    "information",
    "is",
    "me",
    "name",
    "of",
    "on",
    "question",
    "share",
    "some",
    "tell",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class QATextValidationResult:
    """Result of validating a rewritten QA pair against the source pair."""

    passed: bool
    fail_reasons: List[str]
    preserved_names: List[str]
    preserved_numbers: List[str]
    preserved_dates: List[str]
    preserved_titles: List[str]
    preserved_content_tokens: List[str]
    missing_content_tokens: List[str]
    content_token_recall: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _unique_sorted(values: Set[str]) -> List[str]:
    return sorted(v for v in values if v)


def _extract_years(text: str) -> Set[str]:
    return set(YEAR_PATTERN.findall(text))


def _extract_numbers(text: str) -> Set[str]:
    return set(NUMBER_PATTERN.findall(text))


def _extract_titles(text: str) -> Set[str]:
    quoted = {match.strip() for match in QUOTED_TITLE_PATTERN.findall(text)}
    return {value for value in quoted if value}


def _extract_names(text: str) -> Set[str]:
    candidates = {match.strip() for match in TITLE_CASE_PATTERN.findall(text)}
    return {
        candidate
        for candidate in candidates
        if len(candidate.split()) >= 2
    }


def _extract_content_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw in TOKEN_PATTERN.findall(text):
        token = raw.strip()
        if not token:
            continue
        if ACRONYM_PATTERN.fullmatch(token):
            tokens.add(token.lower())
            continue
        normalized = token.lower()
        if len(normalized) < 4:
            continue
        if normalized in CONTENT_STOPWORDS:
            continue
        if normalized.isdigit():
            continue
        tokens.add(normalized)
    return tokens


def validate_naturalized_qa_pair(
    *,
    canonical_question: str,
    canonical_answer: str,
    rewritten_question: str,
    rewritten_answer: str,
    min_content_token_recall: float = 0.65,
) -> QATextValidationResult:
    """Run conservative deterministic parity checks."""
    original_text = f"{canonical_question} {canonical_answer}"
    rewritten_text = f"{rewritten_question} {rewritten_answer}"

    original_names = _extract_names(original_text)
    original_years = _extract_years(original_text)
    original_numbers = _extract_numbers(original_text)
    original_titles = _extract_titles(original_text)
    original_content_tokens = _extract_content_tokens(original_text)

    preserved_names = {value for value in original_names if value in rewritten_text}
    preserved_years = {value for value in original_years if value in rewritten_text}
    preserved_numbers = {value for value in original_numbers if value in rewritten_text}
    preserved_titles = {value for value in original_titles if value in rewritten_text}
    rewritten_content_tokens = _extract_content_tokens(rewritten_text)
    preserved_content_tokens = original_content_tokens & rewritten_content_tokens
    missing_content_tokens = original_content_tokens - rewritten_content_tokens
    content_token_recall = (
        len(preserved_content_tokens) / len(original_content_tokens)
        if original_content_tokens
        else 1.0
    )

    fail_reasons: List[str] = []
    if preserved_names != original_names:
        fail_reasons.append("name_mismatch")
    if preserved_years != original_years:
        fail_reasons.append("date_mismatch")
    if preserved_numbers != original_numbers:
        fail_reasons.append("number_mismatch")
    if preserved_titles != original_titles:
        fail_reasons.append("title_mismatch")
    if content_token_recall < min_content_token_recall:
        fail_reasons.append("content_token_recall_low")

    return QATextValidationResult(
        passed=not fail_reasons,
        fail_reasons=fail_reasons,
        preserved_names=_unique_sorted(preserved_names),
        preserved_numbers=_unique_sorted(preserved_numbers),
        preserved_dates=_unique_sorted(preserved_years),
        preserved_titles=_unique_sorted(preserved_titles),
        preserved_content_tokens=_unique_sorted(preserved_content_tokens),
        missing_content_tokens=_unique_sorted(missing_content_tokens),
        content_token_recall=content_token_recall,
    )
