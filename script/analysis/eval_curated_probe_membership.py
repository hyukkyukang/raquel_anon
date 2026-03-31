"""Evaluate exact value-membership behavior on a curated RAQUEL probe set."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'`.,;:()\[\]{}<>]+")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("query_index"), int):
                out[int(row["query_index"])] = row
    return out


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("_", " ")
    value = _PUNCT_RE.sub(" ", value)
    value = value.lower()
    value = _WS_RE.sub(" ", value)
    return value.strip()


def _extract_string_values(rows: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if not isinstance(value, str):
                continue
            surface = value.strip()
            if not surface:
                continue
            out.setdefault(_normalize(surface), surface)
    return out


def _load_removed_entities(path: str) -> Set[str]:
    payload = _read_json(path)
    out: Set[str] = set()
    if not isinstance(payload, list):
        return out
    for row in payload:
        if not isinstance(row, dict):
            continue
        table = str(row.get("table", "")).strip()
        column = str(row.get("column", "")).strip()
        value = row.get("value")
        if not isinstance(value, str):
            continue
        if not ((table == "person" and column == "name") or (table == "work" and column == "title")):
            continue
        token = _normalize(value)
        if token:
            out.add(token)
    return out


def _prediction_contains(prediction: str, value: str) -> bool:
    return _normalize(value) in _normalize(prediction)


def _query_index_from_example(example: Dict[str, Any]) -> int | None:
    metadata = example.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("query_index"), int):
        return int(metadata["query_index"])
    return None


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate curated RAQUEL probe membership behavior."
    )
    parser.add_argument("--subset_file", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--split_name", default="affected")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument(
        "--removed_entities",
        default="data/aligned_db/log/nullify/removed_entities.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    subset = _read_json(args.subset_file)
    if not isinstance(subset, list):
        raise ValueError(f"Expected list in {args.subset_file}")
    eval_payload = _read_json(args.eval_path)
    if not isinstance(eval_payload, dict):
        raise ValueError(f"Expected dict in {args.eval_path}")
    split_payload = eval_payload.get(args.split_name)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split {args.split_name!r} in {args.eval_path}")
    predictions = split_payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Missing predictions in {args.eval_path}")
    if len(subset) != len(predictions):
        raise ValueError(
            f"Length mismatch: subset={len(subset)} predictions={len(predictions)}"
        )

    denotations = _read_denotations(args.denotations)
    removed_entities = _load_removed_entities(args.removed_entities)

    rows: List[Dict[str, Any]] = []
    for example, pred_row in zip(subset, predictions):
        if not isinstance(example, dict) or not isinstance(pred_row, dict):
            continue
        query_index = _query_index_from_example(example)
        if query_index is None:
            continue
        denotation = denotations.get(query_index)
        if denotation is None:
            continue

        aligned_values = _extract_string_values(denotation.get("result_aligned"))
        null_values = _extract_string_values(denotation.get("result_null"))
        aligned_only = set(aligned_values) - set(null_values)
        null_only = set(null_values) - set(aligned_values)
        removed_aligned_only = [
            aligned_values[token]
            for token in sorted(aligned_only)
            if token in removed_entities
        ]
        null_only_values = [null_values[token] for token in sorted(null_only)]

        prediction = str(pred_row.get("prediction", "")).strip()
        removed_mentions = [
            value for value in removed_aligned_only if _prediction_contains(prediction, value)
        ]
        null_mentions = [
            value for value in null_only_values if _prediction_contains(prediction, value)
        ]

        rows.append(
            {
                "query_index": query_index,
                "query_type": (
                    str(example.get("metadata", {}).get("query_type"))
                    if isinstance(example.get("metadata"), dict)
                    and example["metadata"].get("query_type") is not None
                    else None
                ),
                "question": str(example.get("question", "")).strip(),
                "reference": str(example.get("answer", "")).strip(),
                "prediction": prediction,
                "aligned_only_removed_values": removed_aligned_only,
                "null_only_values": null_only_values,
                "removed_mentions": removed_mentions,
                "null_only_mentions": null_mentions,
                "mentions_any_removed": bool(removed_mentions),
                "mentions_any_null_only": bool(null_mentions),
            }
        )

    payload = {
        "summary": {
            "count": len(rows),
            "examples_with_any_removed_mention": int(
                sum(row["mentions_any_removed"] for row in rows)
            ),
            "examples_with_any_null_only_mention": int(
                sum(row["mentions_any_null_only"] for row in rows)
            ),
            "mean_removed_mentions_per_example": _mean(
                [float(len(row["removed_mentions"])) for row in rows]
            ),
            "mean_null_only_mentions_per_example": _mean(
                [float(len(row["null_only_mentions"])) for row in rows]
            ),
        },
        "metadata": {
            "subset_file": args.subset_file,
            "eval_path": args.eval_path,
            "split_name": args.split_name,
            "denotations": args.denotations,
            "removed_entities": args.removed_entities,
        },
        "examples": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
