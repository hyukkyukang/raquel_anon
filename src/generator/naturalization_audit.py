"""Audit helpers for QA text naturalization metadata."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable


def summarize_naturalization_records(records: Iterable[dict]) -> Dict[str, object]:
    """Summarize rewrite acceptance, fallback, and style distribution."""
    style_counts: Counter[str] = Counter()
    total = 0
    rewritten = 0
    validation_failed = 0
    fallback = 0

    for record in records:
        total += 1
        style = str(record.get("style", "unknown"))
        style_counts[style] += 1
        if record.get("rewrite_changed"):
            rewritten += 1
        if not record.get("validation_passed", False):
            validation_failed += 1
        if record.get("fallback_to_canonical", False):
            fallback += 1

    return {
        "enabled": True,
        "total_pairs": total,
        "rewritten_pairs": rewritten,
        "validation_failed_pairs": validation_failed,
        "fallback_pairs": fallback,
        "accepted_pairs": total - fallback,
        "style_counts": dict(style_counts),
        "validation_pass_rate": (
            (total - validation_failed) / total if total else 0.0
        ),
    }
