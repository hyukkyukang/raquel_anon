"""Compare full affected vs frozen solvable-subset two-reference evaluation.

This is a post-hoc analysis over existing `raquel_two_reference_eval.json`
artifacts. It filters the per-example rows by the frozen solvable subset's
`query_index` set, then reports per-seed and aggregate metrics for:

  - the full affected evaluation slice
  - the frozen solvable subset

The script is intentionally read-only with respect to model inference. The
existing per-example two-reference scores are already available, so the subset
pass can be computed exactly by filtering rows rather than regenerating model
outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


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
    query_indices: List[int] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        metadata = example.get("metadata")
        if not isinstance(metadata, dict):
            continue
        query_index = metadata.get("query_index")
        if isinstance(query_index, int):
            query_indices.append(int(query_index))
    return query_indices


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
        kept_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            seed = row.get("seed")
            eval_path = row.get("two_reference_eval_path")
            if not isinstance(seed, int) or not isinstance(eval_path, str):
                continue
            kept_rows.append({"seed": int(seed), "two_reference_eval_path": eval_path})
        out[group_name] = kept_rows
    return out


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aligned_r1 = [float(row["aligned_scores"]["rouge1_fmeasure"]) for row in rows]
    aligned_r2 = [float(row["aligned_scores"]["rouge2_fmeasure"]) for row in rows]
    aligned_rl = [float(row["aligned_scores"]["rougeL_fmeasure"]) for row in rows]
    null_r1 = [float(row["null_scores"]["rouge1_fmeasure"]) for row in rows]
    null_r2 = [float(row["null_scores"]["rouge2_fmeasure"]) for row in rows]
    null_rl = [float(row["null_scores"]["rougeL_fmeasure"]) for row in rows]
    delta_rl = [float(row["delta_rougeL_fmeasure"]) for row in rows]

    prefer_aligned = sum(1 for row in rows if str(row.get("preferred_side", "")) == "aligned")
    prefer_null = sum(1 for row in rows if str(row.get("preferred_side", "")) == "null")
    ties = sum(1 for row in rows if str(row.get("preferred_side", "")) == "tie")
    total = len(rows)

    return {
        "count": total,
        "mean_similarity_to_aligned": {
            "rouge1_fmeasure": _mean(aligned_r1),
            "rouge2_fmeasure": _mean(aligned_r2),
            "rougeL_fmeasure": _mean(aligned_rl),
        },
        "mean_similarity_to_null": {
            "rouge1_fmeasure": _mean(null_r1),
            "rouge2_fmeasure": _mean(null_r2),
            "rougeL_fmeasure": _mean(null_rl),
        },
        "delta": {"rougeL_fmeasure": _mean(delta_rl)},
        "preference": {
            "prefer_aligned_rate": float(prefer_aligned / total) if total else 0.0,
            "prefer_null_rate": float(prefer_null / total) if total else 0.0,
            "tie_rate": float(ties / total) if total else 0.0,
            "prefer_aligned_minus_null": float((prefer_aligned - prefer_null) / total) if total else 0.0,
        },
    }


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "count": _mean_std([float(row["count"]) for row in rows]),
        "mean_similarity_to_aligned": {
            "rouge1_fmeasure": _mean_std(
                [float(row["mean_similarity_to_aligned"]["rouge1_fmeasure"]) for row in rows]
            ),
            "rouge2_fmeasure": _mean_std(
                [float(row["mean_similarity_to_aligned"]["rouge2_fmeasure"]) for row in rows]
            ),
            "rougeL_fmeasure": _mean_std(
                [float(row["mean_similarity_to_aligned"]["rougeL_fmeasure"]) for row in rows]
            ),
        },
        "mean_similarity_to_null": {
            "rouge1_fmeasure": _mean_std(
                [float(row["mean_similarity_to_null"]["rouge1_fmeasure"]) for row in rows]
            ),
            "rouge2_fmeasure": _mean_std(
                [float(row["mean_similarity_to_null"]["rouge2_fmeasure"]) for row in rows]
            ),
            "rougeL_fmeasure": _mean_std(
                [float(row["mean_similarity_to_null"]["rougeL_fmeasure"]) for row in rows]
            ),
        },
        "delta": {
            "rougeL_fmeasure": _mean_std(
                [float(row["delta"]["rougeL_fmeasure"]) for row in rows]
            )
        },
        "preference": {
            "prefer_aligned_rate": _mean_std(
                [float(row["preference"]["prefer_aligned_rate"]) for row in rows]
            ),
            "prefer_null_rate": _mean_std(
                [float(row["preference"]["prefer_null_rate"]) for row in rows]
            ),
            "tie_rate": _mean_std(
                [float(row["preference"]["tie_rate"]) for row in rows]
            ),
            "prefer_aligned_minus_null": _mean_std(
                [float(row["preference"]["prefer_aligned_minus_null"]) for row in rows]
            ),
        },
    }


def _fmt_stat(stat: Mapping[str, Any], key: str) -> str:
    value = float(stat[key]["mean"])
    std = float(stat[key]["std"])
    return f"{value:.3f} ± {std:.3f}"


def _build_markdown(payload: Mapping[str, Any]) -> str:
    subset_count = int(payload["metadata"]["subset_query_count"])
    lines: List[str] = []
    lines.append("# RAQUEL Two-Reference Eval on Capability-Controlled Solvable Subset")
    lines.append("")
    lines.append(f"- Frozen solvable subset size: `{subset_count}` affected queries")
    lines.append("- Source full affected pool: existing `200`-example two-reference evaluation slice")
    lines.append("- Comparison metric: deterministic denotation-text two-reference ROUGE-L preference")
    lines.append("")
    lines.append("## Aggregate Comparison")
    lines.append("")
    lines.append("| Group | Full count | Subset count | Full prefer-aligned | Subset prefer-aligned | Full prefer-null | Subset prefer-null | Full delta RL | Subset delta RL |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    aggregate = payload["aggregate"]
    for group_name, group_payload in aggregate.items():
        full_stats = group_payload["full"]
        subset_stats = group_payload["subset"]
        lines.append(
            "| "
            + group_name
            + f" | {_fmt_stat(full_stats, 'count')}"
            + f" | {_fmt_stat(subset_stats, 'count')}"
            + f" | {_fmt_stat(full_stats['preference'], 'prefer_aligned_rate')}"
            + f" | {_fmt_stat(subset_stats['preference'], 'prefer_aligned_rate')}"
            + f" | {_fmt_stat(full_stats['preference'], 'prefer_null_rate')}"
            + f" | {_fmt_stat(subset_stats['preference'], 'prefer_null_rate')}"
            + f" | {_fmt_stat(full_stats['delta'], 'rougeL_fmeasure')}"
            + f" | {_fmt_stat(subset_stats['delta'], 'rougeL_fmeasure')}"
            + " |"
        )

    lines.append("")
    lines.append("## Separation Summary")
    lines.append("")
    full = aggregate["full"]["subset"]["preference"]["prefer_aligned_rate"]["mean"]
    retain = aggregate["retain"]["subset"]["preference"]["prefer_aligned_rate"]["mean"]
    full_delta = aggregate["full"]["subset"]["delta"]["rougeL_fmeasure"]["mean"]
    retain_delta = aggregate["retain"]["subset"]["delta"]["rougeL_fmeasure"]["mean"]
    lines.append(
        f"- On the solvable subset, `Morig` prefer-aligned rate is `{full:.3f}` vs `Mret` `{retain:.3f}`."
    )
    lines.append(
        f"- On the solvable subset, mean `aligned - null` ROUGE-L delta is `{full_delta:.3f}` for `Morig` vs `{retain_delta:.3f}` for `Mret`."
    )
    lines.append("")
    lines.append("Interpret this comparison conservatively: the subset only tests whether the counterfactual signal becomes cleaner after removing obviously hard or pathological affected probes.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full vs solvable-subset RAQUEL two-reference evaluation."
    )
    parser.add_argument(
        "--subset_json",
        default="reports/raquel/affected_solvable_subset.json",
    )
    parser.add_argument(
        "--two_ref_inputs",
        default="reports/paper/raquel_two_reference_eval_1b_table_inputs_workstream01.json",
        help="Per-seed table inputs listing the existing two-reference eval files.",
    )
    parser.add_argument(
        "--out_json",
        default="reports/paper/raquel_two_reference_eval_solvable_subset.json",
    )
    parser.add_argument(
        "--out_md",
        default="reports/paper/raquel_two_reference_eval_solvable_subset.md",
    )
    args = parser.parse_args()

    subset_query_indices = _load_subset_query_indices(args.subset_json)
    subset_index_set = set(subset_query_indices)
    path_map = _extract_two_reference_paths(args.two_ref_inputs)

    per_seed_out: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_out: Dict[str, Any] = {}

    for group_name, rows in path_map.items():
        group_seed_rows: List[Dict[str, Any]] = []
        for row in rows:
            eval_path = str(row["two_reference_eval_path"])
            payload = _read_json(eval_path)
            affected = payload.get("affected")
            if not isinstance(affected, Mapping):
                raise ValueError(f"Missing affected block in {eval_path}")
            full_rows = affected.get("rows")
            if not isinstance(full_rows, list):
                raise ValueError(f"Missing affected.rows in {eval_path}")

            subset_rows = [
                item
                for item in full_rows
                if isinstance(item, Mapping)
                and isinstance(item.get("query_index"), int)
                and int(item["query_index"]) in subset_index_set
            ]

            missing_subset = sorted(
                subset_index_set - {int(item["query_index"]) for item in subset_rows}
            )
            full_summary = _summarize_rows(full_rows)
            subset_summary = _summarize_rows(subset_rows)
            group_seed_rows.append(
                {
                    "seed": int(row["seed"]),
                    "two_reference_eval_path": eval_path,
                    "full": full_summary,
                    "subset": subset_summary,
                    "missing_subset_query_indices": missing_subset,
                }
            )

        group_seed_rows = sorted(group_seed_rows, key=lambda item: int(item["seed"]))
        per_seed_out[group_name] = group_seed_rows
        aggregate_out[group_name] = {
            "full": _aggregate_metric_rows([row["full"] for row in group_seed_rows]),
            "subset": _aggregate_metric_rows([row["subset"] for row in group_seed_rows]),
        }

    payload_out = {
        "per_seed": per_seed_out,
        "aggregate": aggregate_out,
        "metadata": {
            "subset_json": args.subset_json,
            "two_ref_inputs": args.two_ref_inputs,
            "subset_query_count": len(subset_query_indices),
            "subset_query_indices": subset_query_indices,
        },
    }

    _write_json(args.out_json, payload_out)
    _write_text(args.out_md, _build_markdown(payload_out))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
