"""Run an LLM-based counterfactual semantic judge over a frozen RAQUEL subset.

This script evaluates saved model outputs against two counterfactual worlds:
the aligned world (pre-forget) and the nullified world (post-forget).

Each judged row asks the model to decide:
  1) whether the output is materially supported by either world
  2) if supported, whether it is closer to world A, world B, both shared, or neither

To reduce position bias, the mapping between A/B and aligned/nullified is
randomized per row with a fixed seed and recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import hkkang_utils.misc as misc_utils
from openai import OpenAI


misc_utils.load_dotenv()


_WS_RE = re.compile(r"\s+")
_ANSWER_TAG_RE = re.compile(r"(?i)\banswer\s*:\s*")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

VALID_SUPPORT = {"yes", "partial", "no"}
VALID_PREFERENCE = {"A", "B", "both_shared", "neither"}


SYSTEM_PROMPT = """You are judging whether a QA model output reflects one of two counterfactual worlds.

You will receive:
- a question
- a model output
- World A and World B reference summaries
- values unique to World A
- values unique to World B

Your task:
1. Decide whether the model output is materially supported by either world.
2. If it is supported fully or partially, decide whether it is closer to World A, closer to World B, only reflects shared content in both, or reflects neither.

Judging rules:
- Focus on factual meaning, not wording.
- Ignore order differences unless order is required by the question.
- If the output contains some grounded correct content plus hallucinated additions, use support="partial".
- Do not give credit for merely naming the right attribute or question topic. If the answer gives the wrong entity, wrong number, or wrong value for the decisive quantity, treat it as unsupported unless it also contains at least one materially correct value from one world.
- For numeric questions, an incorrect number should usually be support="no", not "partial", unless another part of the answer contains a correct world-specific value.
- Use preference="A" if the grounded content clearly matches A-only information.
- Use preference="B" if the grounded content clearly matches B-only information.
- Use preference="both_shared" if the answer only states content shared by both worlds.
- Use preference="neither" if the answer is unsupported, contradictory, too vague to map, or entirely hallucinated.
- If support="no", preference must be "neither".

Return JSON only with this schema:
{
  "support": "yes|partial|no",
  "preference": "A|B|both_shared|neither",
  "matched_A_only_values": ["..."],
  "matched_B_only_values": ["..."],
  "reason": "brief explanation"
}
"""


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _single_line(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip())


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip())


def _clean_prediction_text(prediction: str, question: str) -> str:
    cleaned = str(prediction or "").strip()
    matches = list(_ANSWER_TAG_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()

    q_norm = _normalize_text(question)
    c_norm = _normalize_text(cleaned)
    if q_norm and c_norm.lower().startswith(q_norm.lower()):
        cleaned = cleaned[len(q_norm) :].strip(" \n\r\t:.-")
    return cleaned.strip()


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = _single_line(value)
        return text if text else "null"
    return str(value)


def _row_sort_key(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _textualize_denotation(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "No results found."

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append({str(k): v for k, v in row.items()})
        else:
            normalized_rows.append({"value": row})

    normalized_rows = sorted(normalized_rows, key=_row_sort_key)
    row_texts: List[str] = []
    for row in normalized_rows:
        parts = [f"{key}={_format_value(row[key])}" for key in sorted(row.keys())]
        row_texts.append("; ".join(parts))
    return " | ".join(row_texts)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(_WS_RE.sub(" ", str(text).strip()).lower())


def _safe_fmeasure(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap <= 0 or pred_total <= 0 or ref_total <= 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_fmeasure(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    overlap = _lcs_length(pred_tokens, ref_tokens)
    return _safe_fmeasure(overlap, len(pred_tokens), len(ref_tokens))


def _compute_rouge_preference(prediction: str, aligned_ref: str, null_ref: str) -> Dict[str, Any]:
    aligned = _rouge_l_fmeasure(prediction, aligned_ref)
    nullified = _rouge_l_fmeasure(prediction, null_ref)
    if aligned > nullified:
        pref = "aligned"
    elif nullified > aligned:
        pref = "nullified"
    else:
        pref = "tie"
    return {
        "aligned_rougeL": aligned,
        "nullified_rougeL": nullified,
        "preference": pref,
    }


def _build_user_prompt(row: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "question": row["question"],
            "model_output": row["model_output"],
            "world_A_reference": row["world_A_reference"],
            "world_B_reference": row["world_B_reference"],
            "world_A_only_values": row["world_A_only_values"],
            "world_B_only_values": row["world_B_only_values"],
        },
        ensure_ascii=False,
        indent=2,
    )


def _parse_judge_response(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(raw_text)
    support = str(payload.get("support", "")).strip()
    preference = str(payload.get("preference", "")).strip()
    if support not in VALID_SUPPORT:
        raise ValueError(f"Invalid support label: {support}")
    if preference not in VALID_PREFERENCE:
        raise ValueError(f"Invalid preference label: {preference}")
    if support == "no" and preference != "neither":
        raise ValueError("support=no must imply preference=neither")

    def _clean_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            text = _single_line(str(item))
            if text:
                out.append(text)
        return out

    return {
        "support": support,
        "preference": preference,
        "matched_A_only_values": _clean_list(payload.get("matched_A_only_values", [])),
        "matched_B_only_values": _clean_list(payload.get("matched_B_only_values", [])),
        "reason": _single_line(str(payload.get("reason", ""))),
    }


def _call_openai_judge(
    *,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=seed,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Missing text content in judge response.")
    return content.strip()


def _build_rows(
    subset_payload: Mapping[str, Any],
    *,
    model_labels: Sequence[str],
    order_seed: int,
) -> List[Dict[str, Any]]:
    examples = subset_payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Expected examples list in subset payload.")

    rows: List[Dict[str, Any]] = []
    row_id = 0

    for example in examples:
        if not isinstance(example, dict):
            continue
        metadata = example.get("metadata")
        review_context = example.get("review_context")
        if not isinstance(metadata, dict) or not isinstance(review_context, dict):
            continue
        predictions = review_context.get("predictions")
        if not isinstance(predictions, Mapping):
            continue

        aligned_text = _textualize_denotation(review_context.get("aligned_denotation", []))
        null_text = _textualize_denotation(review_context.get("null_denotation", []))

        for model_label in model_labels:
            if model_label not in predictions:
                continue
            row_id += 1
            local_rng = random.Random(
                f"{order_seed}|{metadata.get('query_index')}|{model_label}"
            )
            a_is_aligned = bool(local_rng.randint(0, 1))
            world_a_reference = aligned_text if a_is_aligned else null_text
            world_b_reference = null_text if a_is_aligned else aligned_text
            world_a_only_values = (
                list(metadata.get("aligned_only_values", []))
                if a_is_aligned
                else list(metadata.get("null_only_values", []))
            )
            world_b_only_values = (
                list(metadata.get("null_only_values", []))
                if a_is_aligned
                else list(metadata.get("aligned_only_values", []))
            )
            prediction_text = _clean_prediction_text(
                str(predictions[model_label]),
                str(example.get("question", "")),
            )
            rouge = _compute_rouge_preference(prediction_text, aligned_text, null_text)
            rows.append(
                {
                    "row_id": row_id,
                    "query_index": metadata.get("query_index"),
                    "example_index": metadata.get("example_index"),
                    "query_type": metadata.get("query_type"),
                    "model_label": model_label,
                    "question": str(example.get("question", "")),
                    "aligned_reference": aligned_text,
                    "nullified_reference": null_text,
                    "model_output": prediction_text,
                    "a_is_aligned": a_is_aligned,
                    "world_A_reference": world_a_reference,
                    "world_B_reference": world_b_reference,
                    "world_A_only_values": world_a_only_values,
                    "world_B_only_values": world_b_only_values,
                    "rouge_preference": rouge,
                }
            )

    return rows


def _mapped_preference(raw_preference: str, *, a_is_aligned: bool) -> str:
    raw_preference = str(raw_preference)
    if raw_preference == "A":
        return "aligned" if a_is_aligned else "nullified"
    if raw_preference == "B":
        return "nullified" if a_is_aligned else "aligned"
    return raw_preference


def _aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_model: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_label"])].append(row)

    summary_by_model: Dict[str, Any] = {}
    for model_label, model_rows in sorted(by_model.items()):
        total = len(model_rows)
        support_counts = Counter(str(row["judge"]["support"]) for row in model_rows)
        pref_counts = Counter(str(row["judge_mapped_preference"]) for row in model_rows)

        supported_rows = [
            row for row in model_rows if str(row["judge"]["support"]) in {"yes", "partial"}
        ]
        decisive_supported_rows = [
            row
            for row in supported_rows
            if str(row["judge_mapped_preference"]) in {"aligned", "nullified"}
        ]

        rouge_agreement = Counter()
        for row in decisive_supported_rows:
            judge_pref = str(row["judge_mapped_preference"])
            rouge_pref = str(row["rouge_preference"]["preference"])
            if rouge_pref == judge_pref:
                rouge_agreement["same_side"] += 1
            elif rouge_pref == "tie":
                rouge_agreement["tie"] += 1
            else:
                rouge_agreement["opposite_side"] += 1

        def _rate(count: int, denom: int) -> float:
            return float(count / denom) if denom else 0.0

        summary_by_model[model_label] = {
            "total_rows": total,
            "support_counts": dict(support_counts),
            "support_rates": {
                label: _rate(support_counts.get(label, 0), total)
                for label in ["yes", "partial", "no"]
            },
            "preference_counts": dict(pref_counts),
            "preference_rates": {
                label: _rate(pref_counts.get(label, 0), total)
                for label in ["aligned", "nullified", "both_shared", "neither"]
            },
            "supported_rows": len(supported_rows),
            "supported_rate": _rate(len(supported_rows), total),
            "decisive_supported_rows": len(decisive_supported_rows),
            "decisive_supported_rate": _rate(len(decisive_supported_rows), total),
            "rouge_agreement_on_decisive_supported": {
                "counts": dict(rouge_agreement),
                "rates": {
                    key: _rate(rouge_agreement.get(key, 0), len(decisive_supported_rows))
                    for key in ["same_side", "tie", "opposite_side"]
                },
            },
        }

    return {
        "row_count": len(rows),
        "query_count": len({int(row["query_index"]) for row in rows}),
        "by_model": summary_by_model,
    }


def _render_markdown(
    *,
    summary: Mapping[str, Any],
    model_name: str,
    subset_json: str,
    annotations_jsonl: str,
) -> str:
    lines: List[str] = []
    lines.append("# LLM Counterfactual Judge Summary")
    lines.append("")
    lines.append(f"- Judge model: `{model_name}`")
    lines.append(f"- Subset: `{subset_json}`")
    lines.append(f"- Row annotations: `{annotations_jsonl}`")
    lines.append(f"- Total judged rows: `{summary['row_count']}`")
    lines.append(f"- Unique queries: `{summary['query_count']}`")
    lines.append("")
    lines.append("## Support Rates")
    lines.append("")
    lines.append("| Model | Yes | Partial | No | Supported |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model_label, payload in summary["by_model"].items():
        support = payload["support_rates"]
        lines.append(
            f"| {model_label}"
            f" | {100.0 * support.get('yes', 0.0):.1f}%"
            f" | {100.0 * support.get('partial', 0.0):.1f}%"
            f" | {100.0 * support.get('no', 0.0):.1f}%"
            f" | {100.0 * payload.get('supported_rate', 0.0):.1f}% |"
        )
    lines.append("")
    lines.append("## Preference Rates")
    lines.append("")
    lines.append("| Model | Aligned | Nullified | Both-shared | Neither | Decisive supported |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model_label, payload in summary["by_model"].items():
        pref = payload["preference_rates"]
        lines.append(
            f"| {model_label}"
            f" | {100.0 * pref.get('aligned', 0.0):.1f}%"
            f" | {100.0 * pref.get('nullified', 0.0):.1f}%"
            f" | {100.0 * pref.get('both_shared', 0.0):.1f}%"
            f" | {100.0 * pref.get('neither', 0.0):.1f}%"
            f" | {100.0 * payload.get('decisive_supported_rate', 0.0):.1f}% |"
        )
    lines.append("")
    lines.append("## Judge vs Two-Reference ROUGE")
    lines.append("")
    lines.append("| Model | Same side | ROUGE tie | Opposite side |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model_label, payload in summary["by_model"].items():
        agree = payload["rouge_agreement_on_decisive_supported"]["rates"]
        lines.append(
            f"| {model_label}"
            f" | {100.0 * agree.get('same_side', 0.0):.1f}%"
            f" | {100.0 * agree.get('tie', 0.0):.1f}%"
            f" | {100.0 * agree.get('opposite_side', 0.0):.1f}% |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- `supported` means `yes` or `partial` support from at least one counterfactual world.")
    lines.append("- `aligned` / `nullified` are mapped back from randomized `A/B` labels.")
    lines.append("- Judge-vs-ROUGE agreement is computed only on supported rows with a decisive judge-side label.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM-based counterfactual judge on a frozen RAQUEL subset.")
    parser.add_argument(
        "--subset_json",
        default="reports/raquel/affected_solvable_subset_v120.json",
    )
    parser.add_argument(
        "--model_label",
        action="append",
        default=[],
        help="Model label(s) embedded in the subset predictions. Defaults to Morig and Mret.",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-5.4-nano-2026-03-17",
    )
    parser.add_argument(
        "--order_seed",
        type=int,
        default=20260330,
        help="Random seed for A/B world assignment.",
    )
    parser.add_argument(
        "--api_seed",
        type=int,
        default=0,
        help="Seed passed to the OpenAI chat completion API.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--retry_delay_sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--out_prefix",
        default="reports/raquel/llm_counterfactual_judge_morig_mret_v120",
    )
    args = parser.parse_args()

    subset_payload = _read_json(args.subset_json)
    model_labels = args.model_label or ["Morig", "Mret"]
    rows = _build_rows(
        subset_payload,
        model_labels=model_labels,
        order_seed=args.order_seed,
    )
    if not rows:
        raise ValueError("No rows found to judge.")

    client = OpenAI()
    judged_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        user_prompt = _build_user_prompt(row)
        raw_response: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = _call_openai_judge(
                    client=client,
                    model_name=args.judge_model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    seed=args.api_seed,
                )
                parsed = _parse_judge_response(raw_response)
                break
            except Exception as exc:  # pragma: no cover - runtime retry path
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(args.retry_delay_sec)
        if parsed is None:
            raise RuntimeError(
                f"Judge failed for row_id={row['row_id']} after {args.max_retries} attempts: {last_error}"
            )

        mapped_pref = _mapped_preference(parsed["preference"], a_is_aligned=bool(row["a_is_aligned"]))
        judged_row = dict(row)
        judged_row["judge"] = parsed
        judged_row["judge_mapped_preference"] = mapped_pref
        judged_row["judge_raw_response"] = raw_response
        judged_rows.append(judged_row)
        print(
            f"[{idx}/{len(rows)}] qidx={row['query_index']} model={row['model_label']} "
            f"support={parsed['support']} pref={mapped_pref}"
        )

    summary = _aggregate_summary(judged_rows)
    summary["judge_model"] = args.judge_model
    summary["subset_json"] = args.subset_json
    summary["model_labels"] = model_labels
    summary["order_seed"] = args.order_seed
    summary["api_seed"] = args.api_seed

    out_prefix = Path(args.out_prefix)
    annotations_jsonl = out_prefix.with_suffix(".jsonl")
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")
    summary_md = out_prefix.with_name(out_prefix.name + "_summary.md")

    _write_jsonl(annotations_jsonl, judged_rows)
    _write_json(summary_json, summary)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(
        _render_markdown(
            summary=summary,
            model_name=args.judge_model,
            subset_json=args.subset_json,
            annotations_jsonl=str(annotations_jsonl),
        ),
        encoding="utf-8",
    )

    print(f"Wrote {annotations_jsonl}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
