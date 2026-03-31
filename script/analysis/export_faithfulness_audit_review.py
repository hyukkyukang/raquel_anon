"""Export a deterministic benchmark-faithfulness audit sample for RAQUEL.

This script joins the final QA examples with their backing SQL/denotation
artifacts, then exports a stratified affected/unaffected review sample for
manual or hybrid annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PRIMARY_FAMILY_PRIORITY = (
    "having",
    "groupby",
    "multi_join",
    "join",
    "orderby",
    "distinct",
    "between",
    "other",
)

DEGENERATE_PATTERNS = (
    "no results found",
    "no data available",
    "not available in the data",
    "cannot determine",
    "can't determine",
    "no results",
    "not provided",
)


@dataclass(frozen=True)
class PoolRow:
    split: str
    query_index: int
    example_index: int
    question: str
    answer: str
    primary_family: str
    tags: Tuple[str, ...]
    sql: str
    result_aligned: Sequence[Mapping[str, Any]]
    result_null: Sequence[Mapping[str, Any]]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_query_index_rows(path: str) -> Dict[str, List[Dict[str, Any]]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in payload:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip()
        if split not in {"affected", "unaffected"}:
            continue
        grouped[split].append(row)
    for split, rows in grouped.items():
        grouped[split] = sorted(rows, key=lambda row: int(row.get("example_index", 0)))
    return dict(grouped)


def _load_tags(path: str) -> Dict[int, Tuple[str, ...]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[int, Tuple[str, ...]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        query_index = row.get("query_index")
        tags = row.get("tags")
        if not isinstance(query_index, int) or not isinstance(tags, list):
            continue
        out[int(query_index)] = tuple(str(tag).strip() for tag in tags if str(tag).strip())
    return out


def _load_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    rows = _read_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        query_index = row.get("query_index")
        if not isinstance(query_index, int):
            continue
        out[int(query_index)] = row
    return out


def _primary_family(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    for family in PRIMARY_FAMILY_PRIORITY:
        if family in tag_set:
            return family
    return "other"


def _normalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def _extract_surface_values(rows: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict):
            values = row.values()
        elif isinstance(row, list):
            values = row
        else:
            values = [row]
        for value in values:
            surface = _normalize_value(value)
            if not surface:
                continue
            out.setdefault(surface.casefold(), surface)
    return out


def _contains_placeholder_like(text: str) -> bool:
    normalized = text.strip().lower()
    if "_" in text:
        return True
    for phrase in (
        "unspecified",
        "unknown",
        "untitled",
        "placeholder",
        "not available",
        "not provided",
        "cannot determine",
    ):
        if phrase in normalized:
            return True
    return bool(re.search(r"\b[a-z]+_[a-z0-9_]+\b", text))


def _is_degenerate(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized in {"0", "0.", "0.0", "0%"}:
        return True
    return any(pattern in normalized for pattern in DEGENERATE_PATTERNS)


def _auto_flags(row: PoolRow) -> Dict[str, bool]:
    return {
        "placeholder_like": _contains_placeholder_like(row.question) or _contains_placeholder_like(row.answer),
        "degenerate_answer": _is_degenerate(row.answer),
        "aligned_empty": len(row.result_aligned) == 0,
        "null_empty": len(row.result_null) == 0,
        "execution_differs": row.result_aligned != row.result_null,
    }


def _allocate_sample_counts(
    family_counts: Mapping[str, int],
    sample_size: int,
    min_per_family: int,
) -> Dict[str, int]:
    families = sorted(family_counts)
    total = sum(family_counts.values())
    if sample_size >= total:
        return {family: family_counts[family] for family in families}
    if sample_size <= 0:
        return {family: 0 for family in families}

    allocation: Dict[str, int] = {family: 0 for family in families}
    remaining = sample_size

    if sample_size < len(families):
        for family, _count in sorted(
            family_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:sample_size]:
            allocation[family] = 1
        return allocation

    for family in families:
        base = min(int(family_counts[family]), int(min_per_family))
        allocation[family] = base
        remaining -= base

    if remaining <= 0:
        while remaining < 0:
            for family, _count in sorted(
                family_counts.items(),
                key=lambda item: (-allocation[item[0]], item[0]),
            ):
                if remaining == 0:
                    break
                if allocation[family] > 0:
                    allocation[family] -= 1
                    remaining += 1
        return allocation

    caps = {
        family: int(family_counts[family]) - int(allocation[family])
        for family in families
    }
    cap_total = sum(max(cap, 0) for cap in caps.values())
    if cap_total <= 0:
        return allocation

    extras: Dict[str, int] = {family: 0 for family in families}
    fractions: List[Tuple[float, str]] = []
    allocated_extra = 0
    for family in families:
        cap = max(caps[family], 0)
        if cap == 0:
            continue
        raw = remaining * (cap / cap_total)
        whole = min(cap, int(math.floor(raw)))
        extras[family] = whole
        allocated_extra += whole
        fractions.append((raw - whole, family))

    remainder = remaining - allocated_extra
    for _fraction, family in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remainder <= 0:
            break
        if extras[family] < caps[family]:
            extras[family] += 1
            remainder -= 1

    if remainder > 0:
        for family, _count in sorted(
            family_counts.items(),
            key=lambda item: (-caps[item[0]], item[0]),
        ):
            while remainder > 0 and extras[family] < caps[family]:
                extras[family] += 1
                remainder -= 1
            if remainder <= 0:
                break

    for family in families:
        allocation[family] += extras[family]
    return allocation


def _stratified_sample(
    rows: Sequence[PoolRow],
    *,
    sample_size: int,
    min_per_family: int,
    sample_seed: int,
) -> Tuple[List[PoolRow], Dict[str, Any]]:
    family_groups: DefaultDict[str, List[PoolRow]] = defaultdict(list)
    for row in rows:
        family_groups[row.primary_family].append(row)

    rng = random.Random(int(sample_seed))
    family_counts = {family: len(items) for family, items in family_groups.items()}
    allocation = _allocate_sample_counts(
        family_counts=family_counts,
        sample_size=int(sample_size),
        min_per_family=int(min_per_family),
    )

    sampled: List[PoolRow] = []
    for family in sorted(family_groups):
        items = list(family_groups[family])
        rng.shuffle(items)
        keep = int(allocation.get(family, 0))
        sampled.extend(items[:keep])

    sampled = sorted(sampled, key=lambda row: (row.primary_family, row.example_index, row.query_index))
    sample_counts = Counter(row.primary_family for row in sampled)
    summary = {
        "pool_counts": dict(sorted(family_counts.items())),
        "sample_counts": dict(sorted(sample_counts.items())),
        "allocation": dict(sorted(allocation.items())),
    }
    return sampled, summary


def _prepare_pool_rows(
    *,
    split: str,
    index_rows: Sequence[Mapping[str, Any]],
    tags_by_index: Mapping[int, Tuple[str, ...]],
    denotations_by_index: Mapping[int, Mapping[str, Any]],
) -> List[PoolRow]:
    pool: List[PoolRow] = []
    for row in index_rows:
        query_index = row.get("query_index")
        example_index = row.get("example_index")
        if not isinstance(query_index, int) or not isinstance(example_index, int):
            continue
        den = denotations_by_index.get(query_index)
        if not isinstance(den, Mapping):
            continue
        aligned_rows = den.get("result_aligned") or []
        null_rows = den.get("result_null") or []
        if not isinstance(aligned_rows, list) or not isinstance(null_rows, list):
            continue
        tags = tags_by_index.get(query_index, ("other",))
        pool.append(
            PoolRow(
                split=split,
                query_index=int(query_index),
                example_index=int(example_index),
                question=str(row.get("question", "")).strip(),
                answer=str(row.get("answer", "")).strip(),
                primary_family=_primary_family(tags),
                tags=tuple(tags),
                sql=str(den.get("sql", "")).strip(),
                result_aligned=aligned_rows,
                result_null=null_rows,
            )
        )
    return pool


def _build_review_rows(sampled_rows: Sequence[PoolRow]) -> List[Dict[str, Any]]:
    review_rows: List[Dict[str, Any]] = []
    for sample_order, row in enumerate(sampled_rows, start=1):
        aligned_map = _extract_surface_values(row.result_aligned)
        null_map = _extract_surface_values(row.result_null)
        aligned_only = [aligned_map[key] for key in sorted(set(aligned_map) - set(null_map))]
        null_only = [null_map[key] for key in sorted(set(null_map) - set(aligned_map))]
        review_rows.append(
            {
                "sample_order": sample_order,
                "split": row.split,
                "label": row.split,
                "query_index": row.query_index,
                "example_index": row.example_index,
                "primary_family": row.primary_family,
                "tags": list(row.tags),
                "question": row.question,
                "textualized_answer": row.answer,
                "sql": row.sql,
                "aligned_row_count": len(row.result_aligned),
                "null_row_count": len(row.result_null),
                "aligned_only_values": aligned_only,
                "null_only_values": null_only,
                "aligned_denotation": row.result_aligned,
                "nullified_denotation": row.result_null,
                "auto_flags": _auto_flags(row),
                "annotation": {
                    "sql_to_text_faithful": "",
                    "answer_preserved": "",
                    "label_correct": "",
                    "issue_flags": "",
                    "notes": "",
                },
            }
        )
    return review_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_order",
        "split",
        "label",
        "query_index",
        "example_index",
        "primary_family",
        "tags",
        "question",
        "textualized_answer",
        "sql",
        "aligned_row_count",
        "null_row_count",
        "aligned_only_values",
        "null_only_values",
        "aligned_denotation",
        "nullified_denotation",
        "auto_placeholder_like",
        "auto_degenerate_answer",
        "auto_aligned_empty",
        "auto_null_empty",
        "auto_execution_differs",
        "sql_to_text_faithful",
        "answer_preserved",
        "label_correct",
        "issue_flags",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            auto_flags = row.get("auto_flags", {})
            annotation = row.get("annotation", {})
            writer.writerow(
                {
                    "sample_order": row.get("sample_order"),
                    "split": row.get("split"),
                    "label": row.get("label"),
                    "query_index": row.get("query_index"),
                    "example_index": row.get("example_index"),
                    "primary_family": row.get("primary_family"),
                    "tags": ";".join(row.get("tags", [])),
                    "question": _single_line(str(row.get("question", ""))),
                    "textualized_answer": _single_line(str(row.get("textualized_answer", ""))),
                    "sql": _single_line(str(row.get("sql", ""))),
                    "aligned_row_count": row.get("aligned_row_count", 0),
                    "null_row_count": row.get("null_row_count", 0),
                    "aligned_only_values": "; ".join(row.get("aligned_only_values", [])),
                    "null_only_values": "; ".join(row.get("null_only_values", [])),
                    "aligned_denotation": _json_compact(row.get("aligned_denotation", [])),
                    "nullified_denotation": _json_compact(row.get("nullified_denotation", [])),
                    "auto_placeholder_like": auto_flags.get("placeholder_like", False),
                    "auto_degenerate_answer": auto_flags.get("degenerate_answer", False),
                    "auto_aligned_empty": auto_flags.get("aligned_empty", False),
                    "auto_null_empty": auto_flags.get("null_empty", False),
                    "auto_execution_differs": auto_flags.get("execution_differs", False),
                    "sql_to_text_faithful": annotation.get("sql_to_text_faithful", ""),
                    "answer_preserved": annotation.get("answer_preserved", ""),
                    "label_correct": annotation.get("label_correct", ""),
                    "issue_flags": annotation.get("issue_flags", ""),
                    "notes": annotation.get("notes", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a deterministic benchmark-faithfulness audit sample for RAQUEL."
    )
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--query_type_tags", default="data/raquel/query_type_tags_heuristic.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument("--affected_size", type=int, default=50)
    parser.add_argument("--unaffected_size", type=int, default=50)
    parser.add_argument("--min_per_family", type=int, default=3)
    parser.add_argument("--sample_seed", type=int, default=20260329)
    parser.add_argument(
        "--out_dir",
        default="reports/raquel",
        help="Output directory for exported artifacts.",
    )
    args = parser.parse_args()

    query_rows_by_split = _load_query_index_rows(args.query_index_map)
    tags_by_index = _load_tags(args.query_type_tags)
    denotations_by_index = _load_denotations(args.denotations)

    affected_pool = _prepare_pool_rows(
        split="affected",
        index_rows=query_rows_by_split.get("affected", []),
        tags_by_index=tags_by_index,
        denotations_by_index=denotations_by_index,
    )
    unaffected_pool = _prepare_pool_rows(
        split="unaffected",
        index_rows=query_rows_by_split.get("unaffected", []),
        tags_by_index=tags_by_index,
        denotations_by_index=denotations_by_index,
    )

    affected_sample, affected_summary = _stratified_sample(
        affected_pool,
        sample_size=int(args.affected_size),
        min_per_family=int(args.min_per_family),
        sample_seed=int(args.sample_seed),
    )
    unaffected_sample, unaffected_summary = _stratified_sample(
        unaffected_pool,
        sample_size=int(args.unaffected_size),
        min_per_family=int(args.min_per_family),
        sample_seed=int(args.sample_seed) + 1,
    )

    sampled_rows = list(affected_sample) + list(unaffected_sample)
    sampled_rows = sorted(
        sampled_rows,
        key=lambda row: (row.split, row.primary_family, row.example_index, row.query_index),
    )
    review_rows = _build_review_rows(sampled_rows)

    auto_flag_counts: MutableMapping[str, Counter[str]] = {
        "affected": Counter(),
        "unaffected": Counter(),
    }
    for row in review_rows:
        split = str(row.get("split"))
        for key, value in row.get("auto_flags", {}).items():
            if value:
                auto_flag_counts[split][key] += 1

    out_dir = Path(args.out_dir)
    sample_json_path = out_dir / "faithfulness_audit_sample.json"
    sample_csv_path = out_dir / "faithfulness_audit_sample.csv"
    annotations_path = out_dir / "faithfulness_audit_annotations.jsonl"
    summary_path = out_dir / "faithfulness_audit_sample_summary.json"

    sample_payload = {
        "metadata": {
            "query_index_map": args.query_index_map,
            "query_type_tags": args.query_type_tags,
            "denotations": args.denotations,
            "affected_size": args.affected_size,
            "unaffected_size": args.unaffected_size,
            "min_per_family": args.min_per_family,
            "sample_seed": args.sample_seed,
        },
        "examples": review_rows,
    }
    annotations_template = [
        {
            "sample_order": row["sample_order"],
            "split": row["split"],
            "label": row["label"],
            "query_index": row["query_index"],
            "example_index": row["example_index"],
            "primary_family": row["primary_family"],
            "sql_to_text_faithful": "",
            "answer_preserved": "",
            "label_correct": "",
            "issue_flags": "",
            "notes": "",
        }
        for row in review_rows
    ]
    summary_payload = {
        "pool_sizes": {
            "affected": len(affected_pool),
            "unaffected": len(unaffected_pool),
        },
        "sample_sizes": {
            "affected": len(affected_sample),
            "unaffected": len(unaffected_sample),
            "combined": len(review_rows),
        },
        "composition": {
            "affected": affected_summary,
            "unaffected": unaffected_summary,
        },
        "auto_flag_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in auto_flag_counts.items()
        },
        "artifacts": {
            "sample_json": str(sample_json_path),
            "sample_csv": str(sample_csv_path),
            "annotations_jsonl": str(annotations_path),
        },
    }

    _write_json(sample_json_path, sample_payload)
    _write_csv(sample_csv_path, review_rows)
    _write_jsonl(annotations_path, annotations_template)
    _write_json(summary_path, summary_payload)

    print(f"Wrote {sample_json_path}")
    print(f"Wrote {sample_csv_path}")
    print(f"Wrote {annotations_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary_payload["sample_sizes"], indent=2))


if __name__ == "__main__":
    main()
