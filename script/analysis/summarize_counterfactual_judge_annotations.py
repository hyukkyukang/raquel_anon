"""Summarize manual counterfactual-preference judge annotations.

This aggregates the row-level semantic judgments exported by
`export_counterfactual_judge_review.py` into model-level counts and rates.
The main use is rebuttal analysis on a frozen solvable affected subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PREFERENCE_ORDER = ["aligned", "nullified", "both_shared", "neither", ""]
SUPPORT_ORDER = ["yes", "partial", "no", ""]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _pref_key(value: str) -> str:
    v = str(value or "").strip()
    return v if v in PREFERENCE_ORDER else "other"


def _support_key(value: str) -> str:
    v = str(value or "").strip()
    return v if v in SUPPORT_ORDER else "other"


def _model_summary(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    pref_counts = Counter(_pref_key(row.get("judge_counterfactual_preference", "")) for row in rows)
    support_counts = Counter(_support_key(row.get("judge_output_supported", "")) for row in rows)
    return {
        "total_rows": total,
        "preference_counts": dict(pref_counts),
        "preference_rates": {k: _rate(pref_counts.get(k, 0), total) for k in PREFERENCE_ORDER if k},
        "support_counts": dict(support_counts),
        "support_rates": {k: _rate(support_counts.get(k, 0), total) for k in SUPPORT_ORDER if k},
        "explicit_counterfactual_rate": _rate(
            pref_counts.get("aligned", 0) + pref_counts.get("nullified", 0), total
        ),
        "supported_non_neither_rate": _rate(
            pref_counts.get("aligned", 0)
            + pref_counts.get("nullified", 0)
            + pref_counts.get("both_shared", 0),
            total,
        ),
    }


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _render_markdown(
    summary: Mapping[str, Any],
    sample_json: str,
    annotations_jsonl: str,
) -> str:
    lines: List[str] = []
    lines.append("# Counterfactual Judge Summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Sample: `{sample_json}`")
    lines.append(f"- Annotations: `{annotations_jsonl}`")
    lines.append(f"- Total rows: `{summary['total_rows']}`")
    lines.append(f"- Unique queries: `{summary['unique_queries']}`")
    lines.append("")
    lines.append("## Preference Rates by Model")
    lines.append("")
    lines.append("| Model | Aligned | Nullified | Both-shared | Neither | Explicit cf. | Supported non-neither |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_label, payload in summary["by_model"].items():
        pref = payload["preference_rates"]
        lines.append(
            f"| {model_label}"
            + f" | {_fmt_pct(pref.get('aligned', 0.0))}"
            + f" | {_fmt_pct(pref.get('nullified', 0.0))}"
            + f" | {_fmt_pct(pref.get('both_shared', 0.0))}"
            + f" | {_fmt_pct(pref.get('neither', 0.0))}"
            + f" | {_fmt_pct(payload['explicit_counterfactual_rate'])}"
            + f" | {_fmt_pct(payload['supported_non_neither_rate'])} |"
        )
    lines.append("")
    lines.append("## Support Rates by Model")
    lines.append("")
    lines.append("| Model | Yes | Partial | No |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model_label, payload in summary["by_model"].items():
        support = payload["support_rates"]
        lines.append(
            f"| {model_label}"
            + f" | {_fmt_pct(support.get('yes', 0.0))}"
            + f" | {_fmt_pct(support.get('partial', 0.0))}"
            + f" | {_fmt_pct(support.get('no', 0.0))} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "- `aligned` / `nullified` count rows where the model output materially points toward one counterfactual world."
    )
    lines.append(
        "- `both_shared` counts rows that use shared denotation content without resolving the aligned-vs-nullified difference."
    )
    lines.append(
        "- `neither` counts unsupported or contradictory outputs."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize counterfactual judge annotations.")
    parser.add_argument(
        "--sample_json",
        default="reports/raquel/solvable_subset_counterfactual_judge_v120_sample.json",
    )
    parser.add_argument(
        "--annotations_jsonl",
        default="reports/raquel/solvable_subset_counterfactual_judge_v120_annotations.jsonl",
    )
    parser.add_argument(
        "--out_json",
        default="reports/paper/solvable_subset_counterfactual_judge_v120_summary.json",
    )
    parser.add_argument(
        "--out_md",
        default="reports/paper/solvable_subset_counterfactual_judge_v120_summary.md",
    )
    args = parser.parse_args()

    sample_payload = _read_json(args.sample_json)
    sample_rows = sample_payload.get("rows", []) if isinstance(sample_payload, dict) else []
    annotations = _read_jsonl(args.annotations_jsonl)
    if len(sample_rows) != len(annotations):
        raise ValueError(
            f"Row count mismatch: sample has {len(sample_rows)} rows, annotations have {len(annotations)} rows."
        )

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unique_queries = set()
    for sample_row, ann_row in zip(sample_rows, annotations):
        if int(sample_row.get("row_id")) != int(ann_row.get("row_id")):
            raise ValueError("Row alignment mismatch between sample and annotations.")
        merged = dict(sample_row)
        merged.update(ann_row)
        model_label = str(merged.get("model_label", ""))
        by_model[model_label].append(merged)
        unique_queries.add(int(merged.get("query_index")))

    summary = {
        "sample_json": args.sample_json,
        "annotations_jsonl": args.annotations_jsonl,
        "total_rows": len(sample_rows),
        "unique_queries": len(unique_queries),
        "by_model": {label: _model_summary(rows) for label, rows in sorted(by_model.items())},
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(
        _render_markdown(summary, args.sample_json, args.annotations_jsonl),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
