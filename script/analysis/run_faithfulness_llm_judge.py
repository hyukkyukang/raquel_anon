"""Run an LLM-assisted benchmark-faithfulness audit over the frozen RAQUEL sample.

This script judges the benchmark artifacts themselves, not model outputs.
Each row asks the model to inspect the SQL, question, textualized answer,
and aligned/nullified denotations, then decide:
  1) whether the question faithfully renders the SQL intent
  2) whether the textualized answer preserves the aligned denotation
  3) whether the affected/unaffected label is justified by the aligned-vs-nullified difference
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hkkang_utils.misc as misc_utils
from openai import OpenAI

misc_utils.load_dotenv()


VALID_YES_NO = {"yes", "no"}
VALID_FLAGS = {
    "sql_text_mismatch",
    "answer_denotation_mismatch",
    "label_mismatch",
    "order_only_difference",
    "null_heavy_verbalization",
    "placeholder_like",
    "time_sensitive",
    "other",
}

SYSTEM_PROMPT = """You are auditing the quality of benchmark examples generated from SQL and execution denotations.

You will receive one benchmark row with:
- the dataset split label (`affected` or `unaffected`)
- a natural-language question
- a textualized answer
- the backing SQL query
- the aligned denotation (the answer source for the benchmark row)
- the nullified denotation (the counterfactual world used only to assess whether the row should be marked affected or unaffected)

Decide three binary properties:
1. `sql_to_text_faithful`
   - "yes" if the question semantically matches the SQL intent.
   - Check filters, grouping, aggregation, ordering, distinctness, limits, and requested attributes.
   - Mark "no" if the question adds or drops a decisive condition, misstates the aggregation, or asks for information the SQL does not return.
2. `answer_preserved`
   - "yes" if the textualized answer accurately verbalizes the aligned denotation.
   - Mark "no" if it drops decisive rows or values, adds unsupported content, collapses a structured denotation into a materially weaker statement, mishandles nulls, or otherwise changes the meaning.
   - Be strict about requested row identities: if the question/SQL asks for specific rows, entities, or a top-k list, the answer must name those rows or entities rather than replace them with a generic summary.
   - If the aligned denotation has multiple rows and the answer compresses them into an aggregate or high-level statement that no longer tells the reader which rows are in the result, mark "no" unless the SQL/question itself only asks for such a summary.
3. `label_correct`
   - "yes" if the provided split label is justified by comparing the aligned and nullified denotations with respect to the question semantics.
   - For `affected`, there must be a material question-relevant difference between aligned and nullified.
   - For `unaffected`, there must not be a material question-relevant difference.
   - Ignore superficial formatting differences.
   - Ignore order-only differences unless order is required by the SQL/question (for example top-k, oldest/newest, ORDER BY with LIMIT).

You may optionally emit issue flags from this set:
- `sql_text_mismatch`
- `answer_denotation_mismatch`
- `label_mismatch`
- `order_only_difference`
- `null_heavy_verbalization`
- `placeholder_like`
- `time_sensitive`
- `other`

When unsure, prefer "no" over "yes". This is a benchmark audit, so preserved structure matters.

Examples:
- If the question asks "What are the 10 series with the oldest average publication year, and what is each series' average publication year and its era?" and the aligned denotation lists 10 concrete series names, then an answer like "There is no available average publication year for any of the ten series, and all ten are in the modern era." should be `answer_preserved = "no"` because it omits the requested series identities.
- If the question asks for the five most recent works and the answer lists the five titles while noting that all publication years are null or missing, that can still be `answer_preserved = "yes"` if the titles match the aligned denotation.

Return JSON only:
{
  "sql_to_text_faithful": "yes|no",
  "answer_preserved": "yes|no",
  "label_correct": "yes|no",
  "issue_flags": ["flag", "..."],
  "notes": "brief explanation"
}
"""


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _row_key(row: Mapping[str, Any]) -> Tuple[int, int]:
    sample_order = row.get("sample_order")
    query_index = row.get("query_index")
    if not isinstance(sample_order, int) or not isinstance(query_index, int):
        raise ValueError(f"Missing key fields in row: {row}")
    return (sample_order, query_index)


def _build_user_prompt(row: Mapping[str, Any]) -> str:
    payload = {
        "sample_order": row["sample_order"],
        "split": row["split"],
        "label": row["label"],
        "query_index": row["query_index"],
        "example_index": row["example_index"],
        "primary_family": row.get("primary_family"),
        "tags": row.get("tags", []),
        "question": row["question"],
        "textualized_answer": row["textualized_answer"],
        "sql": row["sql"],
        "aligned_row_count": row.get("aligned_row_count"),
        "null_row_count": row.get("null_row_count"),
        "aligned_only_values": row.get("aligned_only_values", []),
        "null_only_values": row.get("null_only_values", []),
        "aligned_denotation": row["aligned_denotation"],
        "nullified_denotation": row["nullified_denotation"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _single_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _parse_response(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(raw_text)

    def _parse_yes_no(field: str) -> str:
        value = _single_line(str(payload.get(field, ""))).lower()
        if value not in VALID_YES_NO:
            raise ValueError(f"Invalid {field}: {value}")
        return value

    raw_flags = payload.get("issue_flags", [])
    issue_flags: List[str] = []
    if isinstance(raw_flags, list):
        for item in raw_flags:
            flag = _single_line(str(item))
            if not flag:
                continue
            if flag not in VALID_FLAGS:
                flag = "other"
            if flag not in issue_flags:
                issue_flags.append(flag)

    return {
        "sql_to_text_faithful": _parse_yes_no("sql_to_text_faithful"),
        "answer_preserved": _parse_yes_no("answer_preserved"),
        "label_correct": _parse_yes_no("label_correct"),
        "issue_flags": issue_flags,
        "notes": _single_line(str(payload.get("notes", ""))),
    }


def _call_openai_judge(
    *,
    client: OpenAI,
    model_name: str,
    user_prompt: str,
    seed: int,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        seed=seed,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Missing text content in judge response.")
    return content.strip()


def _annotation_row(
    *,
    sample_row: Mapping[str, Any],
    parsed: Mapping[str, Any],
    model_name: str,
    raw_response: str,
) -> Dict[str, Any]:
    return {
        "sample_order": sample_row["sample_order"],
        "split": sample_row["split"],
        "label": sample_row["label"],
        "query_index": sample_row["query_index"],
        "example_index": sample_row["example_index"],
        "primary_family": sample_row.get("primary_family"),
        "sql_to_text_faithful": parsed["sql_to_text_faithful"],
        "answer_preserved": parsed["answer_preserved"],
        "label_correct": parsed["label_correct"],
        "issue_flags": ";".join(parsed["issue_flags"]),
        "notes": parsed["notes"],
        "judge_model": model_name,
        "judge_raw_response": raw_response,
    }


def _load_sample_rows(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected {{'examples': [...]}} payload in {path}")
    rows = [row for row in examples if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: (int(row.get("sample_order", 0)), int(row.get("query_index", 0))))
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM-assisted faithfulness audit over the frozen RAQUEL sample.")
    parser.add_argument("--sample_json", default="reports/raquel/faithfulness_audit_sample.json")
    parser.add_argument("--judge_model", default="gpt-5-mini")
    parser.add_argument("--api_seed", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay_sec", type=float, default=2.0)
    parser.add_argument(
        "--out_prefix",
        default="reports/raquel/faithfulness_audit_gpt5mini",
        help="Prefix for annotation and summary outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing annotation file.")
    args = parser.parse_args()

    sample_rows = _load_sample_rows(args.sample_json, args.limit)
    if not sample_rows:
        raise ValueError("No sample rows found to judge.")

    out_prefix = Path(args.out_prefix)
    annotations_jsonl = out_prefix.with_name(out_prefix.name + "_annotations.jsonl")
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")
    summary_md = out_prefix.with_name(out_prefix.name + "_summary.md")

    if args.overwrite and annotations_jsonl.exists():
        annotations_jsonl.unlink()

    existing_rows = _read_jsonl(annotations_jsonl)
    existing_by_key = {_row_key(row): row for row in existing_rows}

    client = OpenAI()
    total = len(sample_rows)

    for idx, row in enumerate(sample_rows, start=1):
        key = _row_key(row)
        if key in existing_by_key:
            print(f"[{idx}/{total}] sample={row['sample_order']} qidx={row['query_index']} skipped (existing)")
            continue

        user_prompt = _build_user_prompt(row)
        parsed: Optional[Dict[str, Any]] = None
        raw_response: Optional[str] = None
        last_error: Optional[str] = None

        for _attempt in range(args.max_retries):
            try:
                raw_response = _call_openai_judge(
                    client=client,
                    model_name=args.judge_model,
                    user_prompt=user_prompt,
                    seed=args.api_seed,
                )
                parsed = _parse_response(raw_response)
                break
            except Exception as exc:  # pragma: no cover - runtime retry path
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(args.retry_delay_sec)

        if parsed is None or raw_response is None:
            raise RuntimeError(
                f"Judge failed for sample_order={row['sample_order']} query_index={row['query_index']}: {last_error}"
            )

        annotation = _annotation_row(
            sample_row=row,
            parsed=parsed,
            model_name=args.judge_model,
            raw_response=raw_response,
        )
        _append_jsonl(annotations_jsonl, annotation)
        existing_by_key[key] = annotation
        print(
            f"[{idx}/{total}] sample={row['sample_order']} qidx={row['query_index']} "
            f"sql={parsed['sql_to_text_faithful']} answer={parsed['answer_preserved']} label={parsed['label_correct']}"
        )

    ordered_annotations = [
        existing_by_key[_row_key(row)]
        for row in sample_rows
        if _row_key(row) in existing_by_key
    ]
    summary_payload = {
        "judge_model": args.judge_model,
        "sample_json": args.sample_json,
        "annotation_count": len(ordered_annotations),
        "annotations_jsonl": str(annotations_jsonl),
        "summary_md": str(summary_md),
    }
    _write_json(summary_json, summary_payload)

    print(f"Wrote {annotations_jsonl}")
    print(f"Wrote {summary_json}")
    print(f"Next: summarize with {summary_md}")


if __name__ == "__main__":
    main()
