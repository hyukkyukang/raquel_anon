"""Build a high-precision held-out affected probe set for RAQUEL.

This builder intersects three signals:
  1) held-out affected examples
  2) denotation-grounded forget-attributable probe criterion:
       - aligned-only values contain at least one removed entity
       - null-only values are non-empty
  3) optional model-separation criterion:
       - full model beats retain model by at least `min_delta`

The output is a compact JSON artifact that can be inspected manually or used
for follow-up probe evaluations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

from rouge_score import rouge_scorer


_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+")
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'`.,;:()\[\]{}<>]+|[\s\"'`.,;:()\[\]{}<>]+$")


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
            rec = json.loads(line)
            if isinstance(rec, dict) and isinstance(rec.get("query_index"), int):
                out[int(rec["query_index"])] = rec
    return out


def _normalize(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("_", " ")
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_EDGE_RE.sub("", normalized)
    return normalized.strip()


def _is_salient(token: str) -> bool:
    if not token or token in {"null", "none"}:
        return False
    if len(token) < 3:
        return False
    stop = {
        "books",
        "works",
        "mother",
        "father",
        "authors",
        "author",
        "publishers",
        "archives",
        "libraries",
    }
    return token not in stop


def _extract_values(rows: Any) -> Dict[str, str]:
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
            token = _normalize(surface)
            if _is_salient(token):
                out.setdefault(token, surface)
    return out


def _load_removed_entities(path: str) -> Set[str]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Set[str] = set()
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
        if _is_salient(token) and (" " in token):
            out.add(token)
    return out


def _load_predictions(eval_path: str, split: str) -> List[Dict[str, Any]]:
    payload = _read_json(eval_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {eval_path}")
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split '{split}' in {eval_path}")
    rows = split_payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"Missing predictions for split '{split}' in {eval_path}")
    return [row for row in rows if isinstance(row, dict)]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a high-precision RAQUEL probe set.")
    parser.add_argument("--split_examples", required=True)
    parser.add_argument("--full_eval", required=True)
    parser.add_argument("--retain_eval", required=True)
    parser.add_argument("--split_name", default="affected", choices=["affected", "unaffected"])
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument(
        "--removed_entities",
        default="data/aligned_db/log/nullify/removed_entities.json",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.02,
        help="Minimum full-minus-retain ROUGE-L margin. Set <= -1 to disable.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Keep at most the top-k examples by full-minus-retain margin (0 = keep all).",
    )
    parser.add_argument(
        "--out",
        default="reports/raquel/high_precision_affected_probe.json",
    )
    args = parser.parse_args()

    split_examples = _read_json(args.split_examples)
    if not isinstance(split_examples, list):
        raise ValueError(f"Expected list in {args.split_examples}")
    full_preds = _load_predictions(args.full_eval, args.split_name)
    retain_preds = _load_predictions(args.retain_eval, args.split_name)
    if not (len(split_examples) == len(full_preds) == len(retain_preds)):
        raise ValueError(
            "Length mismatch across split examples and eval predictions: "
            f"{len(split_examples)} vs {len(full_preds)} vs {len(retain_preds)}"
        )

    denotations = _read_denotations(args.denotations)
    removed_entities = _load_removed_entities(args.removed_entities)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows: List[Dict[str, Any]] = []
    for idx, (example, full_row, retain_row) in enumerate(
        zip(split_examples, full_preds, retain_preds)
    ):
        if not isinstance(example, dict):
            continue
        metadata = example.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        query_index = metadata.get("query_index")
        if not isinstance(query_index, int):
            continue

        denotation = denotations.get(query_index)
        if denotation is None:
            continue

        aligned_values = _extract_values(denotation.get("result_aligned"))
        null_values = _extract_values(denotation.get("result_null"))
        aligned_only = set(aligned_values) - set(null_values)
        null_only = set(null_values) - set(aligned_values)
        removed_aligned_only = sorted(token for token in aligned_only if token in removed_entities)

        if not removed_aligned_only or not null_only:
            continue

        reference = str(example.get("answer", "")).strip()
        full_prediction = str(full_row.get("prediction", "")).strip()
        retain_prediction = str(retain_row.get("prediction", "")).strip()
        full_score = scorer.score(reference, full_prediction)["rougeL"].fmeasure
        retain_score = scorer.score(reference, retain_prediction)["rougeL"].fmeasure
        delta = float(full_score - retain_score)

        if float(args.min_delta) > -1.0 and delta < float(args.min_delta):
            continue

        rows.append(
            {
                "example_index": idx,
                "query_index": query_index,
                "query_type": metadata.get("query_type"),
                "question": str(example.get("question", "")).strip(),
                "reference": reference,
                "full_prediction": full_prediction,
                "retain_prediction": retain_prediction,
                "full_rougeL_f1": float(full_score),
                "retain_rougeL_f1": float(retain_score),
                "delta_full_minus_retain": delta,
                "aligned_only_removed_values": [
                    aligned_values[token] for token in removed_aligned_only
                ],
                "null_only_values": [
                    null_values[token] for token in sorted(null_only)
                ],
                "sql": denotation.get("sql", ""),
            }
        )

    rows = sorted(rows, key=lambda row: float(row["delta_full_minus_retain"]), reverse=True)
    if int(args.top_k) > 0:
        rows = rows[: int(args.top_k)]

    payload = {
        "summary": {
            "count": len(rows),
            "mean_full_rougeL_f1": _mean([float(row["full_rougeL_f1"]) for row in rows]),
            "mean_retain_rougeL_f1": _mean([float(row["retain_rougeL_f1"]) for row in rows]),
            "mean_delta_full_minus_retain": _mean(
                [float(row["delta_full_minus_retain"]) for row in rows]
            ),
            "min_delta": float(args.min_delta),
            "top_k": int(args.top_k),
            "split_name": args.split_name,
        },
        "metadata": {
            "split_examples": args.split_examples,
            "full_eval": args.full_eval,
            "retain_eval": args.retain_eval,
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
