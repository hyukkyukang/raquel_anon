"""Carry forward existing capability-controlled annotations into a new sample.

This is useful when expanding the review pool size while preserving the
annotation trail from a smaller already-reviewed sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


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


def _write_jsonl(path: str, rows: List[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge old capability-control annotations into a new larger sample."
    )
    parser.add_argument(
        "--new_sample_json",
        required=True,
        help="Path to the newly exported review sample JSON.",
    )
    parser.add_argument(
        "--old_annotations_jsonl",
        required=True,
        help="Previously completed annotation JSONL.",
    )
    parser.add_argument(
        "--out_annotations_jsonl",
        required=True,
        help="Merged annotation JSONL for the new sample.",
    )
    args = parser.parse_args()

    sample_payload = _read_json(args.new_sample_json)
    if not isinstance(sample_payload, dict):
        raise ValueError(f"Expected dict in {args.new_sample_json}")
    examples = sample_payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in {args.new_sample_json}")

    old_rows = _read_jsonl(args.old_annotations_jsonl)
    old_by_query_index: Dict[int, Dict[str, Any]] = {}
    for row in old_rows:
        query_index = row.get("query_index")
        if isinstance(query_index, int):
            old_by_query_index[int(query_index)] = row

    merged_rows: List[Dict[str, Any]] = []
    carried = 0
    new_blank = 0

    for row in examples:
        if not isinstance(row, dict):
            continue
        query_index = row.get("query_index")
        sample_order = row.get("sample_order")
        primary_family = row.get("primary_family")
        if not isinstance(query_index, int):
            continue

        old = old_by_query_index.get(int(query_index))
        if old is not None:
            merged = dict(old)
            merged["sample_order"] = sample_order
            merged["primary_family"] = primary_family
            carried += 1
        else:
            merged = {
                "query_index": int(query_index),
                "sample_order": sample_order,
                "primary_family": primary_family,
                "judge_sql_to_question": "",
                "judge_aligned_answer_supported": "",
                "judge_null_shift_interpretable": "",
                "judge_not_pathological": "",
                "final_decision": "",
                "pathology_flags": "",
                "notes": "",
            }
            new_blank += 1
        merged_rows.append(merged)

    _write_jsonl(args.out_annotations_jsonl, merged_rows)
    print(f"Wrote {args.out_annotations_jsonl}")
    print(f"Carried existing annotations: {carried}")
    print(f"New blank annotations: {new_blank}")


if __name__ == "__main__":
    main()
