"""Diagnose whether RAQUEL contains an answerable affected subset.

This script compares two RAQUEL eval JSONs, typically:
  - full model (`M_orig`)
  - retain-only retrained model (`M_ret`)

It builds progressively filtered subsets to test whether any plausible
"answerable" region shows `M_orig > M_ret`, even when the full split does not.

Subset families:
  1) structural:
     Filters out obviously degenerate / overly long / list-heavy references.
  2) answerable_either:
     Structural subset where max(full_score, retain_score) >= threshold.
  3) answerable_full:
     Structural subset where full_score >= threshold.
  4) full_better_margin:
     Structural subset where full_score beats retain_score by margin.

Outputs:
  - summary JSON with aggregate statistics
  - JSON subsets for manual inspection / downstream evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from rouge_score import rouge_scorer


DEGENERATE_PATTERNS = (
    "no results found",
    "no data available",
    "not available in the data",
    "cannot determine",
    "can't determine",
    "no results",
    "not provided",
)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    var = sum((value - avg) ** 2 for value in values) / len(values)
    return float(math.sqrt(var))


def _is_degenerate(text: str) -> tuple[bool, str]:
    normalized = text.strip().lower()
    if not normalized:
        return True, "empty"
    for pattern in DEGENERATE_PATTERNS:
        if pattern in normalized:
            return True, pattern
    if normalized in {"0", "0.", "0.0", "0%"}:
        return True, "zero_only"
    return False, ""


def _delimiter_count(text: str) -> int:
    return text.count(",") + text.count(";") + text.count("\n")


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


def _load_examples(path: str) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _build_rows(
    *,
    full_eval_path: str,
    retain_eval_path: str,
    split_examples_path: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    full_rows = _load_predictions(full_eval_path, split_name)
    retain_rows = _load_predictions(retain_eval_path, split_name)
    examples = _load_examples(split_examples_path)

    if not (len(full_rows) == len(retain_rows) == len(examples)):
        raise ValueError(
            "Mismatched lengths across full eval, retain eval, and split examples: "
            f"{len(full_rows)} vs {len(retain_rows)} vs {len(examples)}"
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    merged: List[Dict[str, Any]] = []
    for idx, (frow, rrow, example) in enumerate(zip(full_rows, retain_rows, examples)):
        question = str(example.get("question", "")).strip()
        reference = str(frow.get("reference", example.get("answer", ""))).strip()
        full_prediction = str(frow.get("prediction", "")).strip()
        retain_prediction = str(rrow.get("prediction", "")).strip()

        # Sanity check that eval order still aligns with the split file.
        q_full = str(frow.get("question", "")).strip()
        q_retain = str(rrow.get("question", "")).strip()
        if question and q_full and question != q_full:
            raise ValueError(
                f"Question mismatch between split and full eval at index {idx}: "
                f"split={question!r} full={q_full!r}"
            )
        if question and q_retain and question != q_retain:
            raise ValueError(
                f"Question mismatch between split and retain eval at index {idx}: "
                f"split={question!r} retain={q_retain!r}"
            )

        full_score = scorer.score(reference, full_prediction)["rougeL"].fmeasure
        retain_score = scorer.score(reference, retain_prediction)["rougeL"].fmeasure
        degenerate, degenerate_reason = _is_degenerate(reference)

        merged.append(
            {
                "index": idx,
                "question": question,
                "reference": reference,
                "full_prediction": full_prediction,
                "retain_prediction": retain_prediction,
                "full_rougeL_f1": float(full_score),
                "retain_rougeL_f1": float(retain_score),
                "delta_full_minus_retain": float(full_score - retain_score),
                "reference_chars": len(reference),
                "reference_delimiters": _delimiter_count(reference),
                "degenerate_reference": bool(degenerate),
                "degenerate_reason": degenerate_reason,
                "metadata": example.get("metadata"),
            }
        )
    return merged


def _subset_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    full_scores = [float(row["full_rougeL_f1"]) for row in rows]
    retain_scores = [float(row["retain_rougeL_f1"]) for row in rows]
    deltas = [float(row["delta_full_minus_retain"]) for row in rows]
    full_wins = sum(1 for delta in deltas if delta > 0.0)
    retain_wins = sum(1 for delta in deltas if delta < 0.0)
    equal = len(rows) - full_wins - retain_wins
    return {
        "count": len(rows),
        "full_rougeL_f1": {"mean": _mean(full_scores), "std": _stdev(full_scores)},
        "retain_rougeL_f1": {"mean": _mean(retain_scores), "std": _stdev(retain_scores)},
        "delta_full_minus_retain": {"mean": _mean(deltas), "std": _stdev(deltas)},
        "win_counts": {
            "full": full_wins,
            "retain": retain_wins,
            "equal": equal,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "question",
        "reference",
        "full_prediction",
        "retain_prediction",
        "full_rougeL_f1",
        "retain_rougeL_f1",
        "delta_full_minus_retain",
        "reference_chars",
        "reference_delimiters",
        "degenerate_reference",
        "degenerate_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _top_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    reverse: bool,
) -> List[Mapping[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row["delta_full_minus_retain"]),
        reverse=reverse,
    )
    return ranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose answerable affected subsets for RAQUEL."
    )
    parser.add_argument("--full_eval", required=True, help="Full-model RAQUEL eval JSON")
    parser.add_argument(
        "--retain_eval", required=True, help="Retain-model RAQUEL eval JSON"
    )
    parser.add_argument(
        "--split_examples",
        required=True,
        help="Held-out split JSON used for the evaluation split",
    )
    parser.add_argument(
        "--split_name",
        default="affected",
        choices=["affected", "unaffected"],
        help="Which split to diagnose",
    )
    parser.add_argument(
        "--out_dir",
        default="reports/raquel/answerable_subset_diagnostic",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--max_reference_chars",
        type=int,
        default=256,
        help="Max reference length for the structural subset",
    )
    parser.add_argument(
        "--max_reference_delimiters",
        type=int,
        default=4,
        help="Max comma/semicolon/newline count for the structural subset",
    )
    parser.add_argument(
        "--answerable_threshold",
        type=float,
        default=0.15,
        help="ROUGE-L threshold used for answerable subset definitions",
    )
    parser.add_argument(
        "--full_better_margin",
        type=float,
        default=0.02,
        help="Margin used to define a strong full-model win subset",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="How many best/worst examples to export for inspection",
    )
    args = parser.parse_args()

    rows = _build_rows(
        full_eval_path=args.full_eval,
        retain_eval_path=args.retain_eval,
        split_examples_path=args.split_examples,
        split_name=args.split_name,
    )

    structural = [
        row
        for row in rows
        if not bool(row["degenerate_reference"])
        and int(row["reference_chars"]) <= int(args.max_reference_chars)
        and int(row["reference_delimiters"]) <= int(args.max_reference_delimiters)
    ]
    threshold = float(args.answerable_threshold)
    answerable_either = [
        row
        for row in structural
        if max(float(row["full_rougeL_f1"]), float(row["retain_rougeL_f1"])) >= threshold
    ]
    answerable_full = [
        row for row in structural if float(row["full_rougeL_f1"]) >= threshold
    ]
    full_better_margin = [
        row
        for row in structural
        if float(row["delta_full_minus_retain"]) >= float(args.full_better_margin)
    ]

    summary: Dict[str, Any] = {
        "inputs": {
            "full_eval": args.full_eval,
            "retain_eval": args.retain_eval,
            "split_examples": args.split_examples,
            "split_name": args.split_name,
        },
        "criteria": {
            "max_reference_chars": int(args.max_reference_chars),
            "max_reference_delimiters": int(args.max_reference_delimiters),
            "answerable_threshold": threshold,
            "full_better_margin": float(args.full_better_margin),
        },
        "subsets": {
            "all": _subset_summary(rows),
            "structural": _subset_summary(structural),
            "answerable_either": _subset_summary(answerable_either),
            "answerable_full": _subset_summary(answerable_full),
            "full_better_margin": _subset_summary(full_better_margin),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.split_name
    _write_json(out_dir / f"{stem}_diagnostic_summary.json", summary)
    _write_json(out_dir / f"{stem}_structural_subset.json", structural)
    _write_json(out_dir / f"{stem}_answerable_either_subset.json", answerable_either)
    _write_json(out_dir / f"{stem}_answerable_full_subset.json", answerable_full)
    _write_json(out_dir / f"{stem}_full_better_margin_subset.json", full_better_margin)

    top_full = _top_examples(structural, limit=int(args.top_k), reverse=True)
    top_retain = _top_examples(structural, limit=int(args.top_k), reverse=False)
    _write_csv(out_dir / f"{stem}_top_full_over_retain.csv", top_full)
    _write_csv(out_dir / f"{stem}_top_retain_over_full.csv", top_retain)


if __name__ == "__main__":
    main()
