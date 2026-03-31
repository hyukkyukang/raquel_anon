"""Freeze annotated capability-controlled subset decisions into a stable JSON file."""

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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze annotated solvable-subset decisions into a stable JSON."
    )
    parser.add_argument(
        "--sample_json",
        default="reports/raquel/affected_solvable_subset_review_sample.json",
        help="Full review sample JSON exported by export_capability_controlled_subset_review.py",
    )
    parser.add_argument(
        "--annotations_jsonl",
        default="reports/raquel/affected_solvable_subset_review_annotations.jsonl",
        help="JSONL annotations keyed by query_index.",
    )
    parser.add_argument(
        "--decision_field",
        default="final_decision",
        help="Annotation field used to decide whether an example is kept.",
    )
    parser.add_argument(
        "--positive_label",
        action="append",
        default=["solvable", "keep", "yes"],
        help="Labels treated as positive keeps (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="reports/raquel/affected_solvable_subset.json",
        help="Frozen subset output path.",
    )
    args = parser.parse_args()

    sample_payload = _read_json(args.sample_json)
    if not isinstance(sample_payload, dict):
        raise ValueError(f"Expected dict in {args.sample_json}")
    sample_rows = sample_payload.get("examples")
    if not isinstance(sample_rows, list):
        raise ValueError(f"Expected examples list in {args.sample_json}")

    annotations = _read_jsonl(args.annotations_jsonl)
    annotations_by_index: Dict[int, Mapping[str, Any]] = {}
    for row in annotations:
        query_index = row.get("query_index")
        if isinstance(query_index, int):
            annotations_by_index[int(query_index)] = row

    positive = {_normalize_label(label) for label in args.positive_label}
    kept_examples: List[Dict[str, Any]] = []
    decisions = {"kept": 0, "dropped": 0, "missing_annotation": 0}

    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        query_index = row.get("query_index")
        if not isinstance(query_index, int):
            continue
        annotation = annotations_by_index.get(int(query_index))
        if annotation is None:
            decisions["missing_annotation"] += 1
            continue
        decision_value = _normalize_label(annotation.get(args.decision_field, ""))
        merged_annotation = dict(row.get("annotation", {}))
        merged_annotation.update(annotation)

        if decision_value in positive:
            kept_examples.append(
                {
                    "question": row.get("question", ""),
                    "answer": row.get("aligned_answer_text", ""),
                    "metadata": {
                        "query_index": query_index,
                        "example_index": row.get("example_index"),
                        "query_type": row.get("primary_family"),
                        "tags": row.get("tags", []),
                        "aligned_row_count": row.get("aligned_row_count", 0),
                        "null_row_count": row.get("null_row_count", 0),
                        "aligned_only_values": row.get("aligned_only_values", []),
                        "null_only_values": row.get("null_only_values", []),
                    },
                    "annotation": merged_annotation,
                    "review_context": {
                        "sql": row.get("sql", ""),
                        "aligned_denotation": row.get("aligned_denotation", []),
                        "null_denotation": row.get("null_denotation", []),
                        "auto_flags": row.get("auto_flags", {}),
                        "predictions": row.get("predictions", {}),
                    },
                }
            )
            decisions["kept"] += 1
        else:
            decisions["dropped"] += 1

    out_payload = {
        "metadata": {
            "sample_json": args.sample_json,
            "annotations_jsonl": args.annotations_jsonl,
            "decision_field": args.decision_field,
            "positive_labels": sorted(positive),
        },
        "summary": decisions,
        "examples": kept_examples,
    }
    _write_json(Path(args.out), out_payload)
    print(f"Wrote {args.out} ({len(kept_examples)} kept examples)")


if __name__ == "__main__":
    main()
