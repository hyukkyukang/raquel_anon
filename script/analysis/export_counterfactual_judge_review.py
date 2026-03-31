"""Export a judge sheet for counterfactual preference on a frozen solvable subset.

This creates a lightweight review artifact over the frozen capability-controlled
subset. Each row corresponds to one (query, model) pair and includes:
  - question
  - aligned / null denotations
  - aligned-only / null-only values
  - model output

The intended judgment is whether the model output is conceptually closer to the
aligned counterfactual, the nullified counterfactual, both/ambiguous, or
neither/unsupported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_WS_RE = re.compile(r"\s+")
_ANSWER_TAG_RE = re.compile(r"(?i)\banswer\s*:\s*")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def _single_line(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip())


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clean_prediction_text(prediction: str, question: str) -> str:
    cleaned = str(prediction).strip()
    matches = list(_ANSWER_TAG_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()

    q_norm = _single_line(question)
    c_norm = _single_line(cleaned)
    if q_norm and c_norm.lower().startswith(q_norm.lower()):
        cleaned = cleaned[len(q_norm) :].strip(" \n\r\t:.-")
    return cleaned.strip()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "query_index",
        "sample_order",
        "primary_family",
        "model_label",
        "question",
        "aligned_only_values",
        "null_only_values",
        "aligned_denotation",
        "null_denotation",
        "model_output",
        "judge_counterfactual_preference",
        "judge_output_supported",
        "judge_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_id": row.get("row_id"),
                    "query_index": row.get("query_index"),
                    "sample_order": row.get("sample_order"),
                    "primary_family": row.get("primary_family"),
                    "model_label": row.get("model_label"),
                    "question": _single_line(str(row.get("question", ""))),
                    "aligned_only_values": "; ".join(row.get("aligned_only_values", [])),
                    "null_only_values": "; ".join(row.get("null_only_values", [])),
                    "aligned_denotation": _json_compact(row.get("aligned_denotation", [])),
                    "null_denotation": _json_compact(row.get("null_denotation", [])),
                    "model_output": _single_line(str(row.get("model_output", ""))),
                    "judge_counterfactual_preference": row.get("judge_counterfactual_preference", ""),
                    "judge_output_supported": row.get("judge_output_supported", ""),
                    "judge_notes": row.get("judge_notes", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a counterfactual judge sheet for the frozen solvable subset."
    )
    parser.add_argument(
        "--subset_json",
        default="reports/raquel/affected_solvable_subset.json",
    )
    parser.add_argument(
        "--model_label",
        action="append",
        default=[],
        help="Optional model labels to keep. Defaults to all labels present in the subset review context.",
    )
    parser.add_argument(
        "--out_prefix",
        default="reports/raquel/solvable_subset_counterfactual_judge",
    )
    args = parser.parse_args()

    payload = _read_json(args.subset_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {args.subset_json}")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in {args.subset_json}")

    requested_labels = {str(label).strip() for label in args.model_label if str(label).strip()}
    rows: List[Dict[str, Any]] = []
    model_counts: Dict[str, int] = {}
    row_id = 0

    for example in examples:
        if not isinstance(example, dict):
            continue
        metadata = example.get("metadata")
        annotation = example.get("annotation")
        review_context = example.get("review_context")
        if not isinstance(metadata, dict) or not isinstance(review_context, dict):
            continue
        predictions = review_context.get("predictions")
        if not isinstance(predictions, Mapping):
            continue
        for model_label, prediction in predictions.items():
            if requested_labels and str(model_label) not in requested_labels:
                continue
            row_id += 1
            model_label = str(model_label)
            model_counts[model_label] = model_counts.get(model_label, 0) + 1
            rows.append(
                {
                    "row_id": row_id,
                    "query_index": metadata.get("query_index"),
                    "sample_order": annotation.get("sample_order") if isinstance(annotation, dict) else None,
                    "primary_family": annotation.get("primary_family") if isinstance(annotation, dict) else metadata.get("query_type"),
                    "model_label": model_label,
                    "question": example.get("question", ""),
                    "aligned_only_values": metadata.get("aligned_only_values", []),
                    "null_only_values": metadata.get("null_only_values", []),
                    "aligned_denotation": review_context.get("aligned_denotation", []),
                    "null_denotation": review_context.get("null_denotation", []),
                    "raw_model_output": str(prediction),
                    "model_output": _clean_prediction_text(str(prediction), str(example.get("question", ""))),
                    "judge_counterfactual_preference": "",
                    "judge_output_supported": "",
                    "judge_notes": "",
                }
            )

    out_prefix = Path(args.out_prefix)
    sample_json = out_prefix.with_name(out_prefix.name + "_sample.json")
    sheet_csv = out_prefix.with_name(out_prefix.name + "_sheet.csv")
    annotations_jsonl = out_prefix.with_name(out_prefix.name + "_annotations.jsonl")
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")

    sample_payload = {
        "metadata": {
            "subset_json": args.subset_json,
            "model_labels": sorted(model_counts),
        },
        "rows": rows,
    }
    annotations_template = [
        {
            "row_id": row["row_id"],
            "query_index": row["query_index"],
            "sample_order": row["sample_order"],
            "primary_family": row["primary_family"],
            "model_label": row["model_label"],
            "judge_counterfactual_preference": "",
            "judge_output_supported": "",
            "judge_notes": "",
        }
        for row in rows
    ]
    summary_payload = {
        "subset_size": len(examples),
        "row_count": len(rows),
        "model_counts": dict(sorted(model_counts.items())),
        "artifacts": {
            "sample_json": str(sample_json),
            "sheet_csv": str(sheet_csv),
            "annotations_jsonl": str(annotations_jsonl),
        },
    }

    _write_json(sample_json, sample_payload)
    _write_csv(sheet_csv, rows)
    _write_jsonl(annotations_jsonl, annotations_template)
    _write_json(summary_json, summary_payload)

    print(f"Wrote {sample_json}")
    print(f"Wrote {sheet_csv}")
    print(f"Wrote {annotations_jsonl}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
