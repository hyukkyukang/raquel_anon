"""Build a denotation-metric "gap table" for RAQUEL.

This script aggregates denotation-based metrics (value-level + optional row-level)
from `model/**/raquel_denotation_eval.json` into a compact JSON artifact suitable
for paper tables.

It is intentionally file-driven (no external APIs).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_denotation_gap_table", __file__)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = float(sum(values) / len(values))
    var = float(sum((x - mean) ** 2 for x in values) / len(values))  # population std
    return {"mean": mean, "std": float(var**0.5)}


def _resolve_repo_root() -> str:
    # Script runs from repo root in our workflows; keep it robust anyway.
    return os.path.abspath(os.getcwd())


def _abs_path(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _rel_path(repo_root: str, path: str) -> str:
    try:
        return os.path.relpath(path, repo_root)
    except Exception:
        return path


def _infer_model_dir_from_raquel_eval_path(repo_root: str, raquel_eval_path: str) -> str:
    """Infer the model directory given a RAQUEL eval JSON path."""
    abs_eval_path = _abs_path(repo_root, raquel_eval_path)
    base = os.path.basename(abs_eval_path)
    parent = os.path.dirname(abs_eval_path)
    if base == "raquel_eval_final.json":
        # .../<model_dir>/raquel_eval/raquel_eval_final.json
        return os.path.dirname(parent)
    # .../<model_dir>/raquel_eval.json (or other direct path)
    return parent


def _get_denotation_metrics(payload: Mapping[str, Any], split: str) -> Dict[str, float]:
    """Extract (row_f1, value_f1, parse_success) for a split."""
    split_payload = payload.get(split, {})
    if not isinstance(split_payload, dict):
        return {"parse_success_rate": 0.0, "row_f1_to_aligned": 0.0, "row_f1_to_null": 0.0,
                "value_f1_to_aligned": 0.0, "value_f1_to_null": 0.0}

    parse_success = float(split_payload.get("parse_success_rate", 0.0) or 0.0)

    # New schema (preferred).
    row_aligned = float(split_payload.get("denotation_row_f1_to_aligned", 0.0) or 0.0)
    row_null = float(split_payload.get("denotation_row_f1_to_null", 0.0) or 0.0)
    value_aligned = float(split_payload.get("denotation_value_f1_to_aligned", 0.0) or 0.0)
    value_null = float(split_payload.get("denotation_value_f1_to_null", 0.0) or 0.0)

    # Back-compat: older files used denotation_f1_to_* only (row-set based).
    if "denotation_value_f1_to_aligned" not in split_payload and "denotation_f1_to_aligned" in split_payload:
        value_aligned = float(split_payload.get("denotation_f1_to_aligned", 0.0) or 0.0)
        value_null = float(split_payload.get("denotation_f1_to_null", 0.0) or 0.0)

    return {
        "parse_success_rate": parse_success,
        "row_f1_to_aligned": row_aligned,
        "row_f1_to_null": row_null,
        "value_f1_to_aligned": value_aligned,
        "value_f1_to_null": value_null,
    }


@dataclass(frozen=True)
class DenotationRow:
    seed: int
    model_dir: str
    denotation_eval_path: str
    affected: Dict[str, float]
    unaffected: Dict[str, float]


def _load_rows_from_main_eval_inputs(
    repo_root: str, main_eval_inputs_path: str
) -> Dict[str, List[DenotationRow]]:
    """Load model rows (by group) from main_eval_1b_table_inputs.json."""
    payload = _read_json(main_eval_inputs_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {main_eval_inputs_path}")

    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, dict):
        raise ValueError("Missing 'per_seed' in main eval inputs JSON.")

    out: Dict[str, List[DenotationRow]] = {}
    for group_name, rows in per_seed.items():
        if not isinstance(rows, list):
            continue
        group_rows: List[DenotationRow] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            seed_raw = row.get("seed")
            if not isinstance(seed_raw, int):
                continue

            # Derive model_dir from the RAQUEL eval JSON path stored in the inputs.
            raquel_path: Optional[str] = None
            paths = row.get("paths")
            if isinstance(paths, dict) and isinstance(paths.get("raquel"), str):
                raquel_path = paths["raquel"]
            if raquel_path is None:
                continue

            model_dir_abs = _infer_model_dir_from_raquel_eval_path(repo_root, raquel_path)
            den_path_abs = os.path.join(model_dir_abs, "raquel_denotation_eval.json")

            if not os.path.exists(den_path_abs):
                logger.warning(
                    "Missing denotation eval for %s seed %d at %s",
                    group_name,
                    seed_raw,
                    _rel_path(repo_root, den_path_abs),
                )
                continue

            den_payload = _read_json(den_path_abs)
            if not isinstance(den_payload, dict):
                logger.warning("Invalid denotation eval JSON: %s", den_path_abs)
                continue

            group_rows.append(
                DenotationRow(
                    seed=seed_raw,
                    model_dir=_rel_path(repo_root, model_dir_abs),
                    denotation_eval_path=_rel_path(repo_root, den_path_abs),
                    affected=_get_denotation_metrics(den_payload, "affected"),
                    unaffected=_get_denotation_metrics(den_payload, "unaffected"),
                )
            )
        out[group_name] = sorted(group_rows, key=lambda r: r.seed)
    return out


def _aggregate_group(rows: Sequence[DenotationRow]) -> Dict[str, Any]:
    seeds = [r.seed for r in rows]
    aff_value_leak = [r.affected["value_f1_to_aligned"] for r in rows]
    aff_value_correct = [r.affected["value_f1_to_null"] for r in rows]
    unaff_value = [r.unaffected["value_f1_to_aligned"] for r in rows]

    return {
        "seeds": seeds,
        "affected_value_leakage_to_aligned": _mean_std(aff_value_leak),
        "affected_value_correctness_to_null": _mean_std(aff_value_correct),
        "unaffected_value_utility": _mean_std(unaff_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RAQUEL denotation evaluation into a gap-table JSON."
    )
    parser.add_argument(
        "--main_eval_inputs",
        default="reports/paper/main_eval_1b_table_inputs.json",
        help="Path to main 1B table inputs JSON (used to discover model paths).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_denotation_eval_1b_table_inputs.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    main_eval_inputs_abs = _abs_path(repo_root, str(args.main_eval_inputs))
    out_abs = _abs_path(repo_root, str(args.out))

    rows_by_group = _load_rows_from_main_eval_inputs(repo_root, main_eval_inputs_abs)

    per_seed: Dict[str, List[Dict[str, Any]]] = {}
    aggregate: Dict[str, Any] = {}
    for group_name, rows in rows_by_group.items():
        per_seed[group_name] = [
            {
                "seed": r.seed,
                "model_dir": r.model_dir,
                "denotation_eval_path": r.denotation_eval_path,
                "affected": r.affected,
                "unaffected": r.unaffected,
            }
            for r in rows
        ]
        aggregate[group_name] = _aggregate_group(rows)

    payload_out = {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "metadata": {
            "main_eval_inputs": _rel_path(repo_root, main_eval_inputs_abs),
        },
    }
    _write_json(out_abs, payload_out)
    logger.info("Wrote %s", _rel_path(repo_root, out_abs))


if __name__ == "__main__":
    main()

