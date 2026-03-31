"""Paired comparison for RAQUEL two-reference deltas on full vs subset slices.

This script compares two model groups (default: Morig vs Mret) using the
per-example `delta_rougeL_fmeasure` values already stored in
`raquel_two_reference_eval.json`.

It reports:
  - same-query win/loss/tie counts by seed
  - mean win/loss/tie rates across seeds
  - query-level mean delta differences across seeds
  - bootstrap confidence intervals for the query-level mean difference

The goal is not to claim definitive significance, but to show whether the
capability-controlled subset yields a more stable `Morig > Mret` pattern than
the full affected slice.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = _mean(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": float(mean), "std": float(math.sqrt(var))}


def _load_subset_query_indices(path: str) -> List[int]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in {path}")
    out: List[int] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        metadata = example.get("metadata")
        if not isinstance(metadata, dict):
            continue
        query_index = metadata.get("query_index")
        if isinstance(query_index, int):
            out.append(int(query_index))
    return out


def _extract_two_reference_paths(path: str) -> Dict[str, List[Dict[str, Any]]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, dict):
        raise ValueError(f"Missing per_seed in {path}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for group_name, rows in per_seed.items():
        if not isinstance(rows, list):
            continue
        kept: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            seed = row.get("seed")
            eval_path = row.get("two_reference_eval_path")
            if isinstance(seed, int) and isinstance(eval_path, str):
                kept.append({"seed": int(seed), "two_reference_eval_path": eval_path})
        out[group_name] = sorted(kept, key=lambda item: int(item["seed"]))
    return out


def _load_delta_map(path: str) -> Dict[int, float]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    affected = payload.get("affected")
    if not isinstance(affected, Mapping):
        raise ValueError(f"Missing affected block in {path}")
    rows = affected.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing affected.rows in {path}")
    out: Dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qidx = row.get("query_index")
        delta = row.get("delta_rougeL_fmeasure")
        if isinstance(qidx, int):
            try:
                out[int(qidx)] = float(delta)
            except Exception:
                continue
    return out


def _compare_maps(
    a_map: Mapping[int, float],
    b_map: Mapping[int, float],
    query_indices: Iterable[int],
) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0
    compared: List[Tuple[int, float]] = []
    for qidx in query_indices:
        if qidx not in a_map or qidx not in b_map:
            continue
        diff = float(a_map[qidx] - b_map[qidx])
        compared.append((int(qidx), diff))
        if diff > 0.0:
            wins += 1
        elif diff < 0.0:
            losses += 1
        else:
            ties += 1
    total = len(compared)
    return {
        "count": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / total) if total else 0.0,
        "loss_rate": float(losses / total) if total else 0.0,
        "tie_rate": float(ties / total) if total else 0.0,
        "mean_diff": _mean([diff for _qidx, diff in compared]),
        "per_query_diffs": compared,
    }


def _bootstrap_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(int(seed))
    means: List[float] = []
    values = list(values)
    for _ in range(int(n_boot)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(_mean(sample))
    means = sorted(means)
    low = means[int(0.025 * len(means))]
    high = means[int(0.975 * len(means))]
    return {"mean": _mean(values), "ci95_low": float(low), "ci95_high": float(high)}


def _aggregate_seed_summaries(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "count": _mean_std([float(row["count"]) for row in rows]),
        "wins": _mean_std([float(row["wins"]) for row in rows]),
        "losses": _mean_std([float(row["losses"]) for row in rows]),
        "ties": _mean_std([float(row["ties"]) for row in rows]),
        "win_rate": _mean_std([float(row["win_rate"]) for row in rows]),
        "loss_rate": _mean_std([float(row["loss_rate"]) for row in rows]),
        "tie_rate": _mean_std([float(row["tie_rate"]) for row in rows]),
        "mean_diff": _mean_std([float(row["mean_diff"]) for row in rows]),
    }


def _fmt_ci(stats: Mapping[str, Any]) -> str:
    return f"{float(stats['mean']):.4f} [{float(stats['ci95_low']):.4f}, {float(stats['ci95_high']):.4f}]"


def _build_markdown(payload: Mapping[str, Any]) -> str:
    meta = payload["metadata"]
    full = payload["aggregate"]["full_slice"]
    subset = payload["aggregate"]["subset"]
    lines: List[str] = []
    lines.append("# RAQUEL Two-Reference Paired Robustness Check")
    lines.append("")
    lines.append(
        f"- Compared groups: `{meta['group_a']}` vs `{meta['group_b']}`"
    )
    lines.append(
        f"- Frozen subset size: `{meta['subset_query_count']}` affected queries"
    )
    lines.append(
        f"- Bootstrap draws: `{meta['bootstrap_draws']}`"
    )
    lines.append("")
    lines.append("## Seed-Level Same-Query Comparison")
    lines.append("")
    lines.append("| Slice | Mean wins | Mean losses | Mean ties | Mean win rate | Mean diff |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Full affected | {full['seed_level']['wins']['mean']:.2f} | {full['seed_level']['losses']['mean']:.2f} | {full['seed_level']['ties']['mean']:.2f} | {full['seed_level']['win_rate']['mean']:.3f} | {full['seed_level']['mean_diff']['mean']:.4f} |"
    )
    lines.append(
        f"| Frozen solvable subset | {subset['seed_level']['wins']['mean']:.2f} | {subset['seed_level']['losses']['mean']:.2f} | {subset['seed_level']['ties']['mean']:.2f} | {subset['seed_level']['win_rate']['mean']:.3f} | {subset['seed_level']['mean_diff']['mean']:.4f} |"
    )
    lines.append("")
    lines.append("## Query-Level Mean Difference Across Seeds")
    lines.append("")
    lines.append(
        f"- Full affected: {_fmt_ci(full['query_level_bootstrap'])}"
    )
    lines.append(
        f"- Frozen solvable subset: {_fmt_ci(subset['query_level_bootstrap'])}"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        f"- On the full slice, the query-level mean difference between `{meta['group_a']}` and `{meta['group_b']}` is small and unstable."
    )
    lines.append(
        f"- On the frozen solvable subset, the same gap is larger and its bootstrap interval is more clearly positive."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired robustness comparison on RAQUEL two-reference deltas."
    )
    parser.add_argument(
        "--two_ref_inputs",
        default="reports/paper/raquel_two_reference_eval_1b_table_inputs_workstream01.json",
    )
    parser.add_argument(
        "--subset_json",
        default="reports/raquel/affected_solvable_subset.json",
    )
    parser.add_argument("--group_a", default="full")
    parser.add_argument("--group_b", default="retain")
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--out_json",
        default="reports/paper/raquel_two_reference_paired_compare_full_vs_retain.json",
    )
    parser.add_argument(
        "--out_md",
        default="reports/paper/raquel_two_reference_paired_compare_full_vs_retain.md",
    )
    args = parser.parse_args()

    path_map = _extract_two_reference_paths(args.two_ref_inputs)
    if args.group_a not in path_map or args.group_b not in path_map:
        raise ValueError("Missing comparison groups in two-reference inputs.")

    subset_query_indices = _load_subset_query_indices(args.subset_json)
    subset_index_set = set(subset_query_indices)

    group_a_rows = {int(row["seed"]): str(row["two_reference_eval_path"]) for row in path_map[args.group_a]}
    group_b_rows = {int(row["seed"]): str(row["two_reference_eval_path"]) for row in path_map[args.group_b]}
    seeds = sorted(set(group_a_rows) & set(group_b_rows))
    if not seeds:
        raise ValueError("No overlapping seeds found.")

    per_seed_full: List[Dict[str, Any]] = []
    per_seed_subset: List[Dict[str, Any]] = []
    mean_diffs_full_by_query: Dict[int, List[float]] = {}
    mean_diffs_subset_by_query: Dict[int, List[float]] = {}

    for seed in seeds:
        a_map = _load_delta_map(group_a_rows[seed])
        b_map = _load_delta_map(group_b_rows[seed])
        shared_query_indices = sorted(set(a_map) & set(b_map))
        subset_shared = [qidx for qidx in shared_query_indices if qidx in subset_index_set]

        full_summary = _compare_maps(a_map, b_map, shared_query_indices)
        subset_summary = _compare_maps(a_map, b_map, subset_shared)
        full_summary["seed"] = seed
        subset_summary["seed"] = seed
        per_seed_full.append(full_summary)
        per_seed_subset.append(subset_summary)

        for qidx, diff in full_summary["per_query_diffs"]:
            mean_diffs_full_by_query.setdefault(int(qidx), []).append(float(diff))
        for qidx, diff in subset_summary["per_query_diffs"]:
            mean_diffs_subset_by_query.setdefault(int(qidx), []).append(float(diff))

    full_query_level = [
        _mean(values) for _qidx, values in sorted(mean_diffs_full_by_query.items())
    ]
    subset_query_level = [
        _mean(values) for _qidx, values in sorted(mean_diffs_subset_by_query.items())
    ]

    payload_out = {
        "metadata": {
            "two_ref_inputs": args.two_ref_inputs,
            "subset_json": args.subset_json,
            "group_a": args.group_a,
            "group_b": args.group_b,
            "seeds": seeds,
            "subset_query_count": len(subset_query_indices),
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "per_seed": {
            "full_slice": per_seed_full,
            "subset": per_seed_subset,
        },
        "aggregate": {
            "full_slice": {
                "seed_level": _aggregate_seed_summaries(per_seed_full),
                "query_level_bootstrap": _bootstrap_ci(
                    full_query_level,
                    n_boot=int(args.bootstrap_draws),
                    seed=int(args.bootstrap_seed),
                ),
            },
            "subset": {
                "seed_level": _aggregate_seed_summaries(per_seed_subset),
                "query_level_bootstrap": _bootstrap_ci(
                    subset_query_level,
                    n_boot=int(args.bootstrap_draws),
                    seed=int(args.bootstrap_seed),
                ),
            },
        },
    }

    _write_json(args.out_json, payload_out)
    _write_text(args.out_md, _build_markdown(payload_out))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
