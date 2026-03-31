"""Audit RAQUEL synthesized QA datasets for quality and degeneracy."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEGENERATE_PATTERNS = [
    "no results found",
    "no data available",
    "not available in the data",
    "cannot determine",
    "can't determine",
    "no data available",
    "no results",
    "not provided",
]

SQLISH_PATTERNS = [
    r"\bdistinct\b",
    r"\brecords?\b",
    r"\bgroup by\b",
    r"\border by\b",
    r"\bhaving\b",
    r"\bjoined?\b",
    r"\bassociated\b",
    r"\bconnected\b",
]

ARTIFACT_PHRASES = [
    "[[ ## completed ## ]]",
    "different works",
    "different awards",
    "recorded",
    "associated with the most",
    "connected to the most",
]

RANKING_QUESTION_RE = re.compile(
    r"^(which|what)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty)\b",
    re.IGNORECASE,
)


def _load_examples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of examples in {path}")
    return [ex for ex in payload if isinstance(ex, dict)]


def _get_meta_value(example: Dict[str, Any], key: str) -> Optional[str]:
    value = example.get(key)
    if value is not None:
        return str(value)
    meta = example.get("metadata") or example.get("meta")
    if isinstance(meta, dict) and key in meta:
        return str(meta[key])
    return None


def _is_degenerate(answer: str) -> Tuple[bool, str]:
    text = answer.strip().lower()
    if not text:
        return True, "empty"
    for pattern in DEGENERATE_PATTERNS:
        if pattern in text:
            return True, pattern
    if text in {"0", "0.", "0.0", "0%"}:
        return True, "zero_only"
    return False, ""


def _length_stats(items: Iterable[str]) -> Dict[str, float]:
    lengths = [len(item.strip()) for item in items]
    if not lengths:
        return {"count": 0, "mean": 0.0}
    return {
        "count": len(lengths),
        "mean": float(mean(lengths)),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def _question_quality_stats(questions: List[str]) -> Dict[str, Any]:
    cleaned = [question or "" for question in questions]
    sqlish = sum(
        1
        for question in cleaned
        if any(re.search(pattern, question, re.IGNORECASE) for pattern in SQLISH_PATTERNS)
    )
    artifacty = sum(
        1
        for question in cleaned
        if any(phrase in question.lower() for phrase in ARTIFACT_PHRASES)
    )
    leaked_markers = sum(
        1 for question in cleaned if "[[ ## completed ## ]]" in question
    )
    ranking_starts = sum(
        1 for question in cleaned if RANKING_QUESTION_RE.search(question.strip())
    )
    over_140 = sum(1 for question in cleaned if len(question.strip()) > 140)

    total = len(cleaned)
    return {
        "sqlish_count": sqlish,
        "sqlish_rate": (sqlish / total) if total else 0.0,
        "artifact_phrase_count": artifacty,
        "artifact_phrase_rate": (artifacty / total) if total else 0.0,
        "leaked_marker_count": leaked_markers,
        "leaked_marker_rate": (leaked_markers / total) if total else 0.0,
        "ranking_prefix_count": ranking_starts,
        "ranking_prefix_rate": (ranking_starts / total) if total else 0.0,
        "over_140_chars_count": over_140,
        "over_140_chars_rate": (over_140 / total) if total else 0.0,
    }


def _audit_dataset(name: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = [ex.get("question", "") for ex in examples]
    answers = [ex.get("answer", "") for ex in examples]

    unique_questions = len(set(q.strip() for q in questions if q))
    duplicates = len(questions) - unique_questions

    degeneracy_counter: Counter[str] = Counter()
    degenerate_total = 0
    for ans in answers:
        is_deg, reason = _is_degenerate(ans or "")
        if is_deg:
            degenerate_total += 1
            degeneracy_counter[reason or "degenerate"] += 1

    query_type_counter: Counter[str] = Counter()
    template_id_counter: Counter[str] = Counter()
    template_source_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    for ex in examples:
        qtype = _get_meta_value(ex, "query_type") or "unknown"
        template_id = _get_meta_value(ex, "template_id") or "unknown"
        template_source = _get_meta_value(ex, "template_source") or "unknown"
        family_name = _get_meta_value(ex, "family_name") or "unknown"
        query_type_counter[qtype] += 1
        template_id_counter[template_id] += 1
        template_source_counter[template_source] += 1
        family_counter[family_name] += 1

    return {
        "name": name,
        "counts": {
            "total": len(examples),
            "unique_questions": unique_questions,
            "duplicate_questions": duplicates,
        },
        "lengths": {
            "question": _length_stats([q or "" for q in questions]),
            "answer": _length_stats([a or "" for a in answers]),
        },
        "question_quality": _question_quality_stats([q or "" for q in questions]),
        "degenerate": {
            "total": degenerate_total,
            "rate": (degenerate_total / len(examples)) if examples else 0.0,
            "reasons": dict(degeneracy_counter),
        },
        "coverage": {
            "query_type": dict(query_type_counter),
            "template_id": dict(template_id_counter),
            "template_source": dict(template_source_counter),
            "family_name": dict(family_counter),
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(
    examples: List[Dict[str, Any]], sample_size: int, seed: int
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(examples, min(sample_size, len(examples)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RAQUEL synthesized datasets.")
    parser.add_argument("--affected", required=True, help="Affected JSON file")
    parser.add_argument("--unaffected", required=True, help="Unaffected JSON file")
    parser.add_argument(
        "--out_dir", default="reports/raquel", help="Output directory for reports"
    )
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    affected = _load_examples(args.affected)
    unaffected = _load_examples(args.unaffected)

    affected_report = _audit_dataset("affected", affected)
    unaffected_report = _audit_dataset("unaffected", unaffected)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"affected": affected_report, "unaffected": unaffected_report}
    with (out_dir / "raquel_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export manual audit samples
    if args.sample_size > 0:
        def _sample_csv(examples: List[Dict[str, Any]], name: str) -> None:
            sample = _sample_rows(examples, args.sample_size, args.seed)
            rows = []
            for ex in sample:
                rows.append(
                    {
                        "question": ex.get("question", ""),
                        "answer": ex.get("answer", ""),
                        "query_type": _get_meta_value(ex, "query_type") or "",
                        "template_id": _get_meta_value(ex, "template_id") or "",
                        "template_source": _get_meta_value(ex, "template_source") or "",
                        "family_name": _get_meta_value(ex, "family_name") or "",
                    }
                )
            _write_csv(
                out_dir / f"raquel_manual_sample_{name}.csv",
                rows,
                [
                    "question",
                    "answer",
                    "query_type",
                    "template_id",
                    "template_source",
                    "family_name",
                ],
            )

        _sample_csv(affected, "affected")
        _sample_csv(unaffected, "unaffected")


if __name__ == "__main__":
    main()
