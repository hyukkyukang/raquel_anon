"""Evaluate value-membership behavior on a frozen capability-controlled subset."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'`.,;:()\[\]{}<>]+")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("_", " ")
    value = _PUNCT_RE.sub(" ", value)
    value = value.lower()
    value = _WS_RE.sub(" ", value)
    return value.strip()


def _prediction_contains(prediction: str, value: str) -> bool:
    token = _normalize(value)
    return bool(token) and token in _normalize(prediction)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate value-membership behavior from frozen subset review-context predictions."
    )
    parser.add_argument("--subset_json", required=True)
    parser.add_argument(
        "--prediction_label",
        action="append",
        required=True,
        help="Prediction label stored under review_context.predictions, e.g. Morig",
    )
    parser.add_argument(
        "--ignore_numeric_only",
        action="store_true",
        help="Ignore aligned-only/null-only values whose normalized form is digits only.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = _read_json(args.subset_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {args.subset_json}")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in {args.subset_json}")

    results: Dict[str, Any] = {}
    for label in args.prediction_label:
        rows: List[Dict[str, Any]] = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            metadata = example.get("metadata")
            review_context = example.get("review_context")
            if not isinstance(metadata, dict) or not isinstance(review_context, dict):
                continue
            predictions = review_context.get("predictions")
            if not isinstance(predictions, dict):
                continue

            aligned_only = [
                str(value)
                for value in metadata.get("aligned_only_values", [])
                if isinstance(value, str)
            ]
            null_only = [
                str(value)
                for value in metadata.get("null_only_values", [])
                if isinstance(value, str)
            ]
            if args.ignore_numeric_only:
                aligned_only = [value for value in aligned_only if not _normalize(value).isdigit()]
                null_only = [value for value in null_only if not _normalize(value).isdigit()]

            prediction = str(predictions.get(label, "")).strip()
            aligned_mentions = [value for value in aligned_only if _prediction_contains(prediction, value)]
            null_mentions = [value for value in null_only if _prediction_contains(prediction, value)]

            rows.append(
                {
                    "query_index": metadata.get("query_index"),
                    "query_type": metadata.get("query_type"),
                    "question": example.get("question", ""),
                    "prediction": prediction,
                    "aligned_only_values": aligned_only,
                    "null_only_values": null_only,
                    "aligned_mentions": aligned_mentions,
                    "null_only_mentions": null_mentions,
                    "mentions_any_aligned_only": bool(aligned_mentions),
                    "mentions_any_null_only": bool(null_mentions),
                }
            )

        results[label] = {
            "summary": {
                "count": len(rows),
                "examples_with_any_aligned_only_mention": int(sum(row["mentions_any_aligned_only"] for row in rows)),
                "examples_with_any_null_only_mention": int(sum(row["mentions_any_null_only"] for row in rows)),
                "mean_aligned_mentions_per_example": _mean(
                    [float(len(row["aligned_mentions"])) for row in rows]
                ),
                "mean_null_only_mentions_per_example": _mean(
                    [float(len(row["null_only_mentions"])) for row in rows]
                ),
            },
            "examples": rows,
        }

    out_payload = {
        "metadata": {
            "subset_json": args.subset_json,
            "prediction_labels": list(args.prediction_label),
            "ignore_numeric_only": bool(args.ignore_numeric_only),
        },
        "results": results,
    }
    _write_json(args.out, out_payload)


if __name__ == "__main__":
    main()
