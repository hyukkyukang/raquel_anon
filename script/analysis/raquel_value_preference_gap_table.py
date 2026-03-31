"""Aggregate RAQUEL value-preference evaluation into a paper-ready JSON.

This script reads per-model outputs from:
  <model_dir>/raquel_value_preference_eval_removedA.json

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

logger = get_logger("script.analysis.raquel_value_preference_gap_table", __file__)


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
        # .../<model_dir>/raquel_eval/raquel_eval_final.json
        return os.path.dirname(parent)
    # .../<model_dir>/raquel_eval.json
    return parent


@dataclass(frozen=True)
class ValuePrefRow:
    seed: int
    model_dir: str
    eval_path: str
    count: int
    invalid_pairs: int
    prefer_aligned_rate: float
    mean_margin: float


def _load_value_pref_row(repo_root: str, *, seed: int, model_dir_abs: str) -> Optional[ValuePrefRow]:
    eval_path_abs = os.path.join(model_dir_abs, "raquel_value_preference_eval_removedA.json")
    if not os.path.exists(eval_path_abs):
        logger.warning("Missing value-preference eval: %s", _rel_path(repo_root, eval_path_abs))
        return None
    payload = _read_json(eval_path_abs)
    if not isinstance(payload, dict):
        return None
    affected = payload.get("affected", {})
    if not isinstance(affected, dict):
        return None
    pref = affected.get("preference", {})
    if not isinstance(pref, dict):
        return None
    try:
        count = int(affected.get("count", 0))
        invalid = int(affected.get("invalid_pairs", 0))
        prefer_aligned = float(pref.get("prefer_aligned_rate", 0.0))
        mean_margin = float(pref.get("mean_margin", 0.0))
    except Exception:
        return None
    return ValuePrefRow(
        seed=seed,
        model_dir=_rel_path(repo_root, model_dir_abs),
        eval_path=_rel_path(repo_root, eval_path_abs),
        count=count,
        invalid_pairs=invalid,
        prefer_aligned_rate=prefer_aligned,
        mean_margin=mean_margin,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RAQUEL value-preference eval into table inputs JSON."
    )
    parser.add_argument(
        "--main_eval_inputs",
        default="reports/paper/main_eval_1b_table_inputs.json",
        help="Path to main 1B table inputs JSON (used to discover model paths).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_value_preference_eval_1b_table_inputs.json",
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
        loaded_rows: List[ValuePrefRow] = []
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
            loaded = _load_value_pref_row(repo_root, seed=seed, model_dir_abs=model_dir_abs)
            if loaded is not None:
                loaded_rows.append(loaded)

        loaded_rows = sorted(loaded_rows, key=lambda r: r.seed)
        per_seed_out[group_name] = [
            {
                "seed": r.seed,
                "model_dir": r.model_dir,
                "value_preference_eval_path": r.eval_path,
                "affected_count": r.count,
                "affected_invalid_pairs": r.invalid_pairs,
                "prefer_aligned_rate": r.prefer_aligned_rate,
                "mean_margin": r.mean_margin,
            }
            for r in loaded_rows
        ]
        aggregate_out[group_name] = {
            "seeds": [r.seed for r in loaded_rows],
            "prefer_aligned_rate": _mean_std([r.prefer_aligned_rate for r in loaded_rows]),
            "mean_margin": _mean_std([r.mean_margin for r in loaded_rows]),
            "affected_count": _mean_std([float(r.count) for r in loaded_rows]),
            "affected_invalid_pairs": _mean_std([float(r.invalid_pairs) for r in loaded_rows]),
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

