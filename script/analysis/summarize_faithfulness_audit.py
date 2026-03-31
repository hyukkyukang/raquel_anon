"""Summarize the RAQUEL benchmark-faithfulness audit annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


CRITERIA = (
    "sql_to_text_faithful",
    "answer_preserved",
    "label_correct",
)


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
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def _parse_issue_flags(raw: str) -> List[str]:
    return [flag.strip() for flag in str(raw or "").split(";") if flag.strip()]


def _merge_rows(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ann_by_key: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for row in annotation_rows:
        sample_order = row.get("sample_order")
        query_index = row.get("query_index")
        if not isinstance(sample_order, int) or not isinstance(query_index, int):
            continue
        ann_by_key[(sample_order, query_index)] = row

    merged: List[Dict[str, Any]] = []
    missing: List[Tuple[int, int]] = []
    for sample_row in sample_rows:
        sample_order = sample_row.get("sample_order")
        query_index = sample_row.get("query_index")
        if not isinstance(sample_order, int) or not isinstance(query_index, int):
            continue
        annotation = ann_by_key.get((sample_order, query_index))
        if annotation is None:
            missing.append((sample_order, query_index))
            continue
        merged.append({**sample_row, "annotation": dict(annotation)})

    if missing:
        raise ValueError(
            "Missing annotations for sample rows: "
            + ", ".join(f"(sample_order={order}, query_index={qidx})" for order, qidx in missing)
        )
    return merged


def _criterion_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for criterion in CRITERIA:
        yes_count = 0
        no_count = 0
        blank_count = 0
        for row in rows:
            annotation = row.get("annotation", {})
            value = str(annotation.get(criterion, "")).strip().lower()
            if value == "yes":
                yes_count += 1
            elif value == "no":
                no_count += 1
            else:
                blank_count += 1
        out[criterion] = {
            "yes": yes_count,
            "no": no_count,
            "blank": blank_count,
            "total": len(rows),
        }
    return out


def _all_pass_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        annotation = row.get("annotation", {})
        if all(str(annotation.get(criterion, "")).strip().lower() == "yes" for criterion in CRITERIA):
            count += 1
    return count


def _issue_flag_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        annotation = row.get("annotation", {})
        for flag in _parse_issue_flags(annotation.get("issue_flags", "")):
            counter[flag] += 1
    return dict(sorted(counter.items()))


def _failure_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for row in rows:
        annotation = row.get("annotation", {})
        failed = [
            criterion
            for criterion in CRITERIA
            if str(annotation.get(criterion, "")).strip().lower() == "no"
        ]
        if failed:
            failures.append(
                {
                    "sample_order": row.get("sample_order"),
                    "split": row.get("split"),
                    "query_index": row.get("query_index"),
                    "question": row.get("question"),
                    "failed_criteria": failed,
                    "issue_flags": _parse_issue_flags(annotation.get("issue_flags", "")),
                    "notes": str(annotation.get("notes", "")).strip(),
                }
            )
    return failures


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _render_markdown(
    *,
    merged_rows: Sequence[Mapping[str, Any]],
    per_split: Mapping[str, Sequence[Mapping[str, Any]]],
    sample_sizes: Mapping[str, int],
    title: str,
    protocol_note: str,
    interpretation_note: str,
) -> str:
    overall = _criterion_summary(merged_rows)
    affected = _criterion_summary(per_split.get("affected", []))
    unaffected = _criterion_summary(per_split.get("unaffected", []))
    failures = _failure_rows(merged_rows)
    issue_flags = _issue_flag_counts(merged_rows)

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"We audited `{sample_sizes.get('affected', 0)}` affected and "
        f"`{sample_sizes.get('unaffected', 0)}` unaffected RAQUEL examples "
        "sampled with per-split family stratification."
    )
    lines.append(protocol_note)
    lines.append(
        "The final annotation fields were `sql_to_text_faithful`, "
        "`answer_preserved`, and `label_correct`."
    )
    lines.append("")
    lines.append("## Overall Pass Rates")
    lines.append("")
    lines.append("| Criterion | Yes | No | Pass rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for criterion in CRITERIA:
        summary = overall[criterion]
        lines.append(
            f"| `{criterion}` | {summary['yes']} | {summary['no']} | "
            f"{_pct(summary['yes'], summary['total']):.1f}% |"
        )
    lines.append(
        f"| `all_three_pass` | {_all_pass_count(merged_rows)} | "
        f"{len(merged_rows) - _all_pass_count(merged_rows)} | "
        f"{_pct(_all_pass_count(merged_rows), len(merged_rows)):.1f}% |"
    )
    lines.append("")
    lines.append("## By Split")
    lines.append("")
    lines.append("| Split | Criterion | Yes | No | Pass rate |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for split_name, summary_map in (("affected", affected), ("unaffected", unaffected)):
        total = len(per_split.get(split_name, []))
        for criterion in CRITERIA:
            summary = summary_map[criterion]
            lines.append(
                f"| `{split_name}` | `{criterion}` | {summary['yes']} | {summary['no']} | "
                f"{_pct(summary['yes'], total):.1f}% |"
            )
        lines.append(
            f"| `{split_name}` | `all_three_pass` | {_all_pass_count(per_split.get(split_name, []))} | "
            f"{total - _all_pass_count(per_split.get(split_name, []))} | "
            f"{_pct(_all_pass_count(per_split.get(split_name, [])), total):.1f}% |"
        )
    lines.append("")
    lines.append("## Failure Pattern Notes")
    lines.append("")
    if issue_flags:
        for flag, count in issue_flags.items():
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- No issue flags recorded.")
    lines.append("")
    lines.append("## Representative Failures")
    lines.append("")
    if failures:
        for row in failures[:5]:
            failed = ", ".join(f"`{item}`" for item in row["failed_criteria"])
            flags = ", ".join(f"`{item}`" for item in row["issue_flags"]) if row["issue_flags"] else "none"
            lines.append(
                f"- Sample `{row['sample_order']}` (`{row['split']}`, query `{row['query_index']}`): "
                f"failed {failed}; flags: {flags}. {row['notes']}"
            )
    else:
        lines.append("- No failures recorded.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(interpretation_note)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the RAQUEL faithfulness audit.")
    parser.add_argument("--sample_json", default="reports/raquel/faithfulness_audit_sample.json")
    parser.add_argument("--annotations_jsonl", default="reports/raquel/faithfulness_audit_annotations.jsonl")
    parser.add_argument("--out_md", default="reports/paper/faithfulness_audit_summary.md")
    parser.add_argument("--out_json", default="reports/paper/faithfulness_audit_summary.json")
    parser.add_argument("--title", default="Faithfulness Audit Summary")
    parser.add_argument(
        "--protocol_note",
        default=(
            "This was a hybrid script-assisted pass: `label_correct` was checked "
            "against the stored aligned-vs-nullified execution results, flagged "
            "pathology-heavy rows were manually reviewed, and clean rows were "
            "accepted after spot-checking."
        ),
    )
    parser.add_argument(
        "--interpretation_note",
        default=(
            "The sampled audit suggests that most benchmark examples are faithful "
            "to their backing SQL and denotations, with the main observed "
            "failures coming from null-heavy verbalization and time-sensitive "
            "questions rather than from widespread label errors."
        ),
    )
    args = parser.parse_args()

    sample_payload = _read_json(args.sample_json)
    if not isinstance(sample_payload, dict) or not isinstance(sample_payload.get("examples"), list):
        raise ValueError(f"Expected {{'examples': [...]}} payload in {args.sample_json}")
    sample_rows = [row for row in sample_payload["examples"] if isinstance(row, dict)]
    annotation_rows = _read_jsonl(args.annotations_jsonl)

    merged_rows = _merge_rows(sample_rows=sample_rows, annotation_rows=annotation_rows)
    per_split: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        per_split[str(row.get("split", ""))].append(row)

    sample_sizes = {
        "affected": len(per_split.get("affected", [])),
        "unaffected": len(per_split.get("unaffected", [])),
        "combined": len(merged_rows),
    }

    summary_payload = {
        "sample_sizes": sample_sizes,
        "overall": _criterion_summary(merged_rows),
        "by_split": {
            split: _criterion_summary(rows)
            for split, rows in per_split.items()
        },
        "all_three_pass": {
            "combined": _all_pass_count(merged_rows),
            "affected": _all_pass_count(per_split.get("affected", [])),
            "unaffected": _all_pass_count(per_split.get("unaffected", [])),
        },
        "title": args.title,
        "protocol_note": args.protocol_note,
        "interpretation_note": args.interpretation_note,
        "issue_flags": _issue_flag_counts(merged_rows),
        "failures": _failure_rows(merged_rows),
    }

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _render_markdown(
            merged_rows=merged_rows,
            per_split=per_split,
            sample_sizes=sample_sizes,
            title=args.title,
            protocol_note=args.protocol_note,
            interpretation_note=args.interpretation_note,
        ),
        encoding="utf-8",
    )
    _write_json(Path(args.out_json), summary_payload)

    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
