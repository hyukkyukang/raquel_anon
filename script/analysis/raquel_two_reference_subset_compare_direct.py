"""Compare full vs frozen subset metrics for explicit two-reference eval files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _parse_eval_specs(values: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --eval spec {raw!r}; expected LABEL=PATH.")
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --eval spec {raw!r}; expected LABEL=PATH.")
        parsed.append((label, path))
    return parsed


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
        qidx = metadata.get("query_index")
        if isinstance(qidx, int):
            out.append(int(qidx))
    return out


def _load_rows(path: str) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    affected = payload.get("affected")
    if not isinstance(affected, dict):
        raise ValueError(f"Missing affected section in {path}")
    rows = affected.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing affected.rows in {path}")
    return [row for row in rows if isinstance(row, dict)]


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aligned_rl = [float(row["aligned_scores"]["rougeL_fmeasure"]) for row in rows]
    null_rl = [float(row["null_scores"]["rougeL_fmeasure"]) for row in rows]
    delta_rl = [float(row["delta_rougeL_fmeasure"]) for row in rows]

    prefer_aligned = sum(1 for row in rows if str(row.get("preferred_side", "")) == "aligned")
    prefer_null = sum(1 for row in rows if str(row.get("preferred_side", "")) == "null")
    ties = sum(1 for row in rows if str(row.get("preferred_side", "")) == "tie")
    total = len(rows)

    return {
        "count": total,
        "mean_similarity_to_aligned": {"rougeL_fmeasure": _mean(aligned_rl)},
        "mean_similarity_to_null": {"rougeL_fmeasure": _mean(null_rl)},
        "delta": {"rougeL_fmeasure": _mean(delta_rl)},
        "preference": {
            "prefer_aligned_rate": float(prefer_aligned / total) if total else 0.0,
            "prefer_null_rate": float(prefer_null / total) if total else 0.0,
            "tie_rate": float(ties / total) if total else 0.0,
            "prefer_aligned_minus_null": float((prefer_aligned - prefer_null) / total) if total else 0.0,
        },
    }


def _build_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Held-Out Join `s7` Two-Reference Eval on Frozen Solvable Subset")
    lines.append("")
    lines.append(f"- Frozen subset size: `{payload['metadata']['subset_query_count']}` affected queries")
    lines.append("- Source pool: held-out `join_holdout_s7` affected split")
    lines.append("- Metric: deterministic denotation-text two-reference ROUGE-L preference")
    lines.append("")
    lines.append("| Model | Full count | Subset count | Full prefer-aligned | Subset prefer-aligned | Full prefer-null | Subset prefer-null | Full delta RL | Subset delta RL |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for label, stats in payload["results"].items():
        full = stats["full"]
        subset = stats["subset"]
        lines.append(
            f"| {label} | {full['count']} | {subset['count']} | "
            f"{full['preference']['prefer_aligned_rate']:.3f} | {subset['preference']['prefer_aligned_rate']:.3f} | "
            f"{full['preference']['prefer_null_rate']:.3f} | {subset['preference']['prefer_null_rate']:.3f} | "
            f"{full['delta']['rougeL_fmeasure']:.3f} | {subset['delta']['rougeL_fmeasure']:.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full vs subset two-reference metrics for explicit eval files.")
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    subset_query_indices = set(_load_subset_query_indices(args.subset_json))
    eval_specs = _parse_eval_specs(args.eval)

    results: Dict[str, Any] = {}
    for label, path in eval_specs:
        rows = _load_rows(path)
        subset_rows = [row for row in rows if int(row.get("query_index", -1)) in subset_query_indices]
        results[label] = {
            "path": path,
            "full": _summarize(rows),
            "subset": _summarize(subset_rows),
        }

    payload = {
        "metadata": {
            "subset_json": args.subset_json,
            "subset_query_count": len(subset_query_indices),
            "eval_specs": [{"label": label, "path": path} for label, path in eval_specs],
        },
        "results": results,
    }
    _write_json(args.out_json, payload)
    _write_text(args.out_md, _build_markdown(payload))


if __name__ == "__main__":
    main()
