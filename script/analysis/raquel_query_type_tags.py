"""Build RAQUEL query-index maps and heuristic query-type tags.

This script exists to support paper-facing diagnostics when the shipped RAQUEL QA JSONs
contain only {"question","answer"} without template/query metadata.

It produces:
  - data/raquel/query_index_map.json
  - data/raquel/query_type_tags_heuristic.json
  - reports/paper/raquel_skip_breakdown_by_type.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.utils.data_loaders import load_sql_queries, load_translated_queries
from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_query_type_tags", __file__)


def _read_json_list(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file and return it as a list of dicts."""
    with open(path, "r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")

    items: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            items.append(item)
    return items


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Normalize text for stable matching (whitespace + strip)."""
    return _WS_RE.sub(" ", text.strip())


@dataclass(frozen=True)
class QueryIndexRecord:
    """Mapping record for a single QA example."""

    split: str
    question: str
    query_index: int
    # Extra fields (helpful for stable joins; ignored by downstream if not needed).
    example_index: int
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "question": self.question,
            "query_index": self.query_index,
            "example_index": self.example_index,
            "answer": self.answer,
        }


def _build_question_to_indices(
    translated_queries: Sequence[str],
) -> Dict[str, List[int]]:
    """Build a mapping from normalized question text to all matching query indices."""
    mapping: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, q in enumerate(translated_queries):
        mapping[_normalize_text(q)].append(idx)
    return dict(mapping)


def _assign_query_indices(
    *,
    split_name: str,
    examples: Sequence[Dict[str, Any]],
    question_to_indices: Dict[str, List[int]],
    used_per_question: Optional[DefaultDict[str, int]] = None,
) -> Tuple[List[QueryIndexRecord], int]:
    """Assign query indices to QA examples by matching to translated queries."""
    if used_per_question is None:
        used_per_question = defaultdict(int)

    records: List[QueryIndexRecord] = []
    unmatched = 0
    for ex_idx, ex in enumerate(examples):
        question: str = str(ex.get("question", "")).strip()
        answer: str = str(ex.get("answer", "")).strip()
        if not question:
            unmatched += 1
            continue

        norm_q = _normalize_text(question)
        candidates = question_to_indices.get(norm_q)
        if not candidates:
            unmatched += 1
            continue

        cursor = used_per_question[norm_q]
        if cursor >= len(candidates):
            unmatched += 1
            continue

        qidx = int(candidates[cursor])
        used_per_question[norm_q] += 1
        records.append(
            QueryIndexRecord(
                split=split_name,
                question=question,
                query_index=qidx,
                example_index=ex_idx,
                answer=answer,
            )
        )

    return records, unmatched


def _heuristic_query_tags(sql: str) -> List[str]:
    """Assign coarse query-type tags based on SQL keywords.

    Tags are intentionally simple and may overlap.
    """
    text = f" {sql.strip().lower()} "

    tags: Set[str] = set()
    join_count = text.count(" join ")
    if join_count:
        tags.add("join")
        if join_count >= 2:
            tags.add("multi_join")

    if " group by " in text:
        tags.add("groupby")
    if " having " in text:
        tags.add("having")
    if " order by " in text:
        tags.add("orderby")
    if " union " in text:
        tags.add("union")
    if " distinct " in text:
        tags.add("distinct")
    if " between " in text:
        tags.add("between")
    if " case " in text and " when " in text:
        tags.add("case_when")

    # Subquery heuristics
    # (We add both a specific and a generic subquery tag.)
    if text.count(" select ") >= 2:
        tags.add("subquery")
    if " exists " in text and "select" in text:
        tags.add("subquery_exists")
    if " in (select" in text or " in ( select" in text:
        tags.add("subquery_in")

    if not tags:
        tags.add("other")
    return sorted(tags)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAQUEL query-index maps and heuristic query-type tags."
    )
    parser.add_argument("--affected", required=True)
    parser.add_argument("--unaffected", required=True)
    parser.add_argument("--translated_queries", required=True)
    parser.add_argument("--sql_queries", required=True)
    parser.add_argument("--out_map", default="data/raquel/query_index_map.json")
    parser.add_argument(
        "--out_tags", default="data/raquel/query_type_tags_heuristic.json"
    )
    parser.add_argument(
        "--out_skip_report",
        default="reports/paper/raquel_skip_breakdown_by_type.json",
    )
    args = parser.parse_args()

    affected = _read_json_list(args.affected)
    unaffected = _read_json_list(args.unaffected)
    translated = load_translated_queries(args.translated_queries)
    sql_queries = load_sql_queries(args.sql_queries)

    if len(translated) != len(sql_queries):
        raise ValueError(
            f"translated_queries ({len(translated)}) != sql_queries ({len(sql_queries)})"
        )

    question_to_indices = _build_question_to_indices(translated)

    used_per_question: DefaultDict[str, int] = defaultdict(int)
    affected_records, affected_unmatched = _assign_query_indices(
        split_name="affected",
        examples=affected,
        question_to_indices=question_to_indices,
        used_per_question=used_per_question,
    )
    unaffected_records, unaffected_unmatched = _assign_query_indices(
        split_name="unaffected",
        examples=unaffected,
        question_to_indices=question_to_indices,
        used_per_question=used_per_question,
    )

    all_records = affected_records + unaffected_records
    used_indices: Set[int] = {rec.query_index for rec in all_records}

    logger.info(
        "Mapped %d/%d affected and %d/%d unaffected questions to query indices",
        len(affected_records),
        len(affected),
        len(unaffected_records),
        len(unaffected),
    )
    if affected_unmatched or unaffected_unmatched:
        logger.warning(
            "Unmatched questions: affected=%d, unaffected=%d",
            affected_unmatched,
            unaffected_unmatched,
        )

    _write_json(args.out_map, [rec.to_dict() for rec in all_records])

    # Tag all queries by SQL heuristics (one record per query index).
    tags_by_index: Dict[int, List[str]] = {}
    for idx, sql in enumerate(sql_queries):
        tags_by_index[idx] = _heuristic_query_tags(sql)
    _write_json(
        args.out_tags,
        [{"query_index": idx, "tags": tags} for idx, tags in sorted(tags_by_index.items())],
    )

    # Skip report.
    total_queries = len(sql_queries)
    skipped_indices = [idx for idx in range(total_queries) if idx not in used_indices]

    tag_totals: Counter[str] = Counter()
    tag_skipped: Counter[str] = Counter()
    for idx, tags in tags_by_index.items():
        for tag in tags:
            tag_totals[tag] += 1
        if idx in skipped_indices:
            for tag in tags:
                tag_skipped[tag] += 1

    split_counts: Counter[str] = Counter(rec.split for rec in all_records)
    report = {
        "total_queries": total_queries,
        "kept_total": len(all_records),
        "kept_by_split": dict(split_counts),
        "skipped_total": len(skipped_indices),
        "skipped_indices": skipped_indices,
        "skip_by_tag": dict(tag_skipped),
        "total_by_tag": dict(tag_totals),
        "skip_rate_by_tag": {
            tag: (tag_skipped[tag] / tag_totals[tag]) if tag_totals[tag] else 0.0
            for tag in sorted(tag_totals)
        },
        "unmatched_questions": {
            "affected": affected_unmatched,
            "unaffected": unaffected_unmatched,
        },
    }
    _write_json(args.out_skip_report, report)

    logger.info(
        "Wrote query_index_map=%s, query_type_tags=%s, skip_report=%s",
        args.out_map,
        args.out_tags,
        args.out_skip_report,
    )
    logger.info(
        "Skipped %d/%d queries (expected ~233); kept affected=%d, unaffected=%d",
        len(skipped_indices),
        total_queries,
        split_counts.get("affected", 0),
        split_counts.get("unaffected", 0),
    )


if __name__ == "__main__":
    main()

