"""Aggregate RAQUEL two-reference evaluation into a paper-ready JSON.

This script reads per-model outputs from:
  <model_dir>/raquel_two_reference_eval.json

and aggregates metrics by group/seed using `reports/paper/main_eval_1b_table_inputs.json`
to discover the model directories.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_two_reference_gap_table", __file__)


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
    var = float(sum((x - mean) ** 2 for x in values) / len(values))
    return {"mean": mean, "std": float(var**0.5)}


def _resolve_repo_root() -> str:
    return os.path.abspath(os.getcwd())


def _abs_path(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _rel_path(repo_root: str, path: str) -> str:
    try:
        return os.path.relpath(path, repo_root)
    except Exception:
        return path


def _infer_model_dir_from_raquel_eval_path(repo_root: str, raquel_eval_path: str) -> str:
    abs_eval_path = _abs_path(repo_root, raquel_eval_path)
    base = os.path.basename(abs_eval_path)
    parent = os.path.dirname(abs_eval_path)
    if base == "raquel_eval_final.json":
        return os.path.dirname(parent)
    return parent


@dataclass(frozen=True)
class TwoRefRow:
    seed: int
    model_dir: str
    eval_path: str
    count: int
    unmatched_questions: int
    missing_denotations: int
    aligned_r1: float
    aligned_r2: float
    aligned_rl: float
    null_r1: float
    null_r2: float
    null_rl: float
    delta_rl: float
    prefer_aligned_rate: float
    prefer_null_rate: float
    tie_rate: float


def _load_two_ref_row(repo_root: str, *, seed: int, model_dir_abs: str) -> Optional[TwoRefRow]:
    eval_path_abs = os.path.join(model_dir_abs, "raquel_two_reference_eval.json")
    if not os.path.exists(eval_path_abs):
        logger.warning("Missing two-reference eval: %s", _rel_path(repo_root, eval_path_abs))
        return None

    payload = _read_json(eval_path_abs)
    if not isinstance(payload, dict):
        return None
    affected = payload.get("affected", {})
    if not isinstance(affected, dict):
        return None
    aligned = affected.get("mean_similarity_to_aligned", {})
    null = affected.get("mean_similarity_to_null", {})
    delta = affected.get("delta", {})
    pref = affected.get("preference", {})
    if not all(isinstance(x, Mapping) for x in (aligned, null, delta, pref)):
        return None

    try:
        return TwoRefRow(
            seed=seed,
            model_dir=_rel_path(repo_root, model_dir_abs),
            eval_path=_rel_path(repo_root, eval_path_abs),
            count=int(affected.get("count", 0)),
            unmatched_questions=int(affected.get("unmatched_questions", 0)),
            missing_denotations=int(affected.get("missing_denotations", 0)),
            aligned_r1=float(aligned.get("rouge1_fmeasure", 0.0)),
            aligned_r2=float(aligned.get("rouge2_fmeasure", 0.0)),
            aligned_rl=float(aligned.get("rougeL_fmeasure", 0.0)),
            null_r1=float(null.get("rouge1_fmeasure", 0.0)),
            null_r2=float(null.get("rouge2_fmeasure", 0.0)),
            null_rl=float(null.get("rougeL_fmeasure", 0.0)),
            delta_rl=float(delta.get("rougeL_fmeasure", 0.0)),
            prefer_aligned_rate=float(pref.get("prefer_aligned_rate", 0.0)),
            prefer_null_rate=float(pref.get("prefer_null_rate", 0.0)),
            tie_rate=float(pref.get("tie_rate", 0.0)),
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RAQUEL two-reference eval into table inputs JSON."
    )
    parser.add_argument(
        "--main_eval_inputs",
        default="reports/paper/main_eval_1b_table_inputs.json",
        help="Path to main 1B table inputs JSON (used to discover model paths).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_two_reference_eval_1b_table_inputs.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    main_eval_inputs_abs = _abs_path(repo_root, str(args.main_eval_inputs))
    out_abs = _abs_path(repo_root, str(args.out))

    main_inputs = _read_json(main_eval_inputs_abs)
    if not isinstance(main_inputs, dict) or "per_seed" not in main_inputs:
        raise ValueError("Invalid main eval inputs JSON.")
    per_seed = main_inputs.get("per_seed", {})
    if not isinstance(per_seed, dict):
        raise ValueError("Invalid main eval inputs JSON: per_seed missing.")

    per_seed_out: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_out: Dict[str, Any] = {}

    for group_name, rows in per_seed.items():
        if not isinstance(rows, list):
            continue
        loaded_rows: List[TwoRefRow] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            seed = row.get("seed")
            paths = row.get("paths")
            if not isinstance(seed, int) or not isinstance(paths, dict):
                continue
            raquel_path = paths.get("raquel")
            if not isinstance(raquel_path, str):
                continue
            model_dir_abs = _infer_model_dir_from_raquel_eval_path(repo_root, raquel_path)
            loaded = _load_two_ref_row(repo_root, seed=seed, model_dir_abs=model_dir_abs)
            if loaded is not None:
                loaded_rows.append(loaded)

        loaded_rows = sorted(loaded_rows, key=lambda r: r.seed)
        per_seed_out[group_name] = [
            {
                "seed": r.seed,
                "model_dir": r.model_dir,
                "two_reference_eval_path": r.eval_path,
                "affected_count": r.count,
                "unmatched_questions": r.unmatched_questions,
                "missing_denotations": r.missing_denotations,
                "mean_similarity_to_aligned": {
                    "rouge1_fmeasure": r.aligned_r1,
                    "rouge2_fmeasure": r.aligned_r2,
                    "rougeL_fmeasure": r.aligned_rl,
                },
                "mean_similarity_to_null": {
                    "rouge1_fmeasure": r.null_r1,
                    "rouge2_fmeasure": r.null_r2,
                    "rougeL_fmeasure": r.null_rl,
                },
                "delta": {"rougeL_fmeasure": r.delta_rl},
                "preference": {
                    "prefer_aligned_rate": r.prefer_aligned_rate,
                    "prefer_null_rate": r.prefer_null_rate,
                    "tie_rate": r.tie_rate,
                },
            }
            for r in loaded_rows
        ]
        aggregate_out[group_name] = {
            "seeds": [r.seed for r in loaded_rows],
            "affected_count": _mean_std([float(r.count) for r in loaded_rows]),
            "unmatched_questions": _mean_std([float(r.unmatched_questions) for r in loaded_rows]),
            "missing_denotations": _mean_std([float(r.missing_denotations) for r in loaded_rows]),
            "mean_similarity_to_aligned": {
                "rouge1_fmeasure": _mean_std([r.aligned_r1 for r in loaded_rows]),
                "rouge2_fmeasure": _mean_std([r.aligned_r2 for r in loaded_rows]),
                "rougeL_fmeasure": _mean_std([r.aligned_rl for r in loaded_rows]),
            },
            "mean_similarity_to_null": {
                "rouge1_fmeasure": _mean_std([r.null_r1 for r in loaded_rows]),
                "rouge2_fmeasure": _mean_std([r.null_r2 for r in loaded_rows]),
                "rougeL_fmeasure": _mean_std([r.null_rl for r in loaded_rows]),
            },
            "delta": {
                "rougeL_fmeasure": _mean_std([r.delta_rl for r in loaded_rows]),
            },
            "preference": {
                "prefer_aligned_rate": _mean_std([r.prefer_aligned_rate for r in loaded_rows]),
                "prefer_null_rate": _mean_std([r.prefer_null_rate for r in loaded_rows]),
                "tie_rate": _mean_std([r.tie_rate for r in loaded_rows]),
            },
        }

    payload_out = {
        "per_seed": per_seed_out,
        "aggregate": aggregate_out,
        "metadata": {"main_eval_inputs": _rel_path(repo_root, main_eval_inputs_abs)},
    }
    _write_json(out_abs, payload_out)
    logger.info("Wrote %s", _rel_path(repo_root, out_abs))


if __name__ == "__main__":
    main()
