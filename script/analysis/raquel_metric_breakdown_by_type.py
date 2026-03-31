"""Compute RAQUEL ROUGE breakdowns by heuristic SQL query type.

This script joins:
  - data/raquel/query_index_map.json (QA -> query_index)
  - data/raquel/query_type_tags_heuristic.json (query_index -> tags)
  - one or more RAQUEL eval JSONs produced by `script/evaluation/run_raquel_eval.py --save_predictions`

and reports ROUGE (esp. rougeL_fmeasure) by tag bucket for affected/unaffected splits.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.metrics import RougeMetric
from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_metric_breakdown_by_type", __file__)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class MappedExample:
    query_index: int
    question: str
    reference: str
    prediction: str
    tags: Tuple[str, ...]


def _load_query_index_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    by_split: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in payload:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip()
        if split not in {"affected", "unaffected"}:
            continue
        by_split[split].append(row)
    # Sort by example_index if present; otherwise preserve file order.
    for split in list(by_split.keys()):
        rows = by_split[split]
        if rows and "example_index" in rows[0]:
            rows = sorted(rows, key=lambda r: int(r.get("example_index", 0)))
        by_split[split] = rows
    return dict(by_split)


def _load_tags(path: str) -> Dict[int, Tuple[str, ...]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[int, Tuple[str, ...]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        idx = row.get("query_index")
        tags = row.get("tags")
        if not isinstance(idx, int) or not isinstance(tags, list):
            continue
        out[int(idx)] = tuple(str(t) for t in tags)
    return out


def _compute_rouge(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    metric = RougeMetric()
    metric.update(predictions=preds, references=refs)
    return metric.compute()


def _map_eval_split(
    *,
    split_name: str,
    eval_predictions: List[Dict[str, Any]],
    index_rows: List[Dict[str, Any]],
    tags_by_index: Mapping[int, Tuple[str, ...]],
) -> List[MappedExample]:
    """Map evaluation predictions to query indices using (stable) file order.

    We validate alignment by checking the question text at each position.
    """
    mapped: List[MappedExample] = []
    n = len(eval_predictions)
    if n > len(index_rows):
        raise ValueError(
            f"Eval split '{split_name}' has {n} predictions but index map has only {len(index_rows)} rows"
        )

    for i in range(n):
        pred_row = eval_predictions[i]
        map_row = index_rows[i]

        q_pred = str(pred_row.get("question", "")).strip()
        q_map = str(map_row.get("question", "")).strip()
        if q_pred and q_map and q_pred != q_map:
            # Fall back to permissive matching if order changed for some reason.
            # (This should be rare; we keep it simple and log.)
            logger.warning(
                "Question mismatch at %s[%d]: eval='%s' map='%s'",
                split_name,
                i,
                q_pred[:80],
                q_map[:80],
            )

        query_index = int(map_row["query_index"])
        tags = tags_by_index.get(query_index, ("other",))
        mapped.append(
            MappedExample(
                query_index=query_index,
                question=q_pred or q_map,
                reference=str(pred_row.get("reference", "")).strip(),
                prediction=str(pred_row.get("prediction", "")).strip(),
                tags=tuple(tags),
            )
        )
    return mapped


def _breakdown_by_tag(examples: Sequence[MappedExample]) -> Dict[str, Any]:
    """Compute ROUGE breakdown for each tag (overlapping buckets)."""
    all_tags: List[str] = sorted({tag for ex in examples for tag in ex.tags})
    by_tag: Dict[str, Any] = {}
    for tag in all_tags:
        bucket = [ex for ex in examples if tag in ex.tags]
        preds = [ex.prediction for ex in bucket]
        refs = [ex.reference for ex in bucket]
        by_tag[tag] = {
            "count": len(bucket),
            "rouge": _compute_rouge(preds, refs) if bucket else {},
        }
    return {"tags": all_tags, "by_tag": by_tag}


def _load_eval(path: str) -> Dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    if "affected" not in payload or "unaffected" not in payload:
        raise ValueError(f"Missing affected/unaffected in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="RAQUEL metric breakdown by query type.")
    parser.add_argument(
        "--query_index_map", default="data/raquel/query_index_map.json"
    )
    parser.add_argument(
        "--query_type_tags", default="data/raquel/query_type_tags_heuristic.json"
    )
    parser.add_argument(
        "--eval_glob",
        action="append",
        default=[],
        help="Glob(s) for RAQUEL eval JSONs with predictions (repeatable).",
    )
    parser.add_argument(
        "--eval_path",
        action="append",
        default=[],
        help="Explicit RAQUEL eval JSON path(s) (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_metric_breakdown_by_type.json",
    )
    args = parser.parse_args()

    index_map = _load_query_index_map(args.query_index_map)
    tags_by_index = _load_tags(args.query_type_tags)

    eval_paths: List[str] = []
    for pattern in args.eval_glob:
        eval_paths.extend(sorted(glob.glob(pattern)))
    eval_paths.extend(args.eval_path)
    eval_paths = sorted(dict.fromkeys(eval_paths))  # de-dup, stable order

    if not eval_paths:
        raise ValueError("No eval files provided. Use --eval_path or --eval_glob.")

    reports: List[Dict[str, Any]] = []
    for path in eval_paths:
        payload = _load_eval(path)
        affected_preds = payload["affected"].get("predictions")
        unaffected_preds = payload["unaffected"].get("predictions")
        if not isinstance(affected_preds, list) or not isinstance(unaffected_preds, list):
            raise ValueError(
                f"Eval file missing predictions lists (run with --save_predictions): {path}"
            )

        mapped_affected = _map_eval_split(
            split_name="affected",
            eval_predictions=affected_preds,
            index_rows=index_map["affected"],
            tags_by_index=tags_by_index,
        )
        mapped_unaffected = _map_eval_split(
            split_name="unaffected",
            eval_predictions=unaffected_preds,
            index_rows=index_map["unaffected"],
            tags_by_index=tags_by_index,
        )

        report = {
            "eval_path": path,
            "metadata": payload.get("metadata", {}),
            "affected": {
                "overall": payload["affected"].get("rouge", {}),
                "breakdown": _breakdown_by_tag(mapped_affected),
            },
            "unaffected": {
                "overall": payload["unaffected"].get("rouge", {}),
                "breakdown": _breakdown_by_tag(mapped_unaffected),
            },
        }
        reports.append(report)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"reports": reports}, handle, indent=2)

    logger.info("Wrote %s (%d eval files)", args.out, len(reports))


if __name__ == "__main__":
    main()

