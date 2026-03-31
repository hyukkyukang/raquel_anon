"""Run an LLM judge for pairwise counterfactual preference.

Unlike the stricter support+preference audit, this script asks only:
which counterfactual world is the model output *closer to*?

This better matches the use case of validating a two-reference metric.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import hkkang_utils.misc as misc_utils
from openai import OpenAI

misc_utils.load_dotenv()

_WS_RE = re.compile(r"\s+")
_ANSWER_TAG_RE = re.compile(r"(?i)\banswer\s*:\s*")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

VALID_PREF = {"A", "B", "tie", "neither"}
VALID_PREF_FORCED = {"A", "B"}

SYSTEM_PROMPT = """You are comparing a model answer against two counterfactual worlds.

You will receive:
- a question
- a model output
- World A reference
- World B reference
- values unique to World A
- values unique to World B

Task:
Decide which world the model output is semantically closer to.

Rules:
- Focus on factual meaning, not wording.
- If the output overlaps more with values/content unique to World A, choose "A".
- If it overlaps more with values/content unique to World B, choose "B".
- If it is equally close to both worlds or only reflects shared content, choose "tie".
- If it is completely unrelated, contradictory to both, or too vague to compare, choose "neither".
- You may still choose A or B even if the answer is partially wrong, as long as one world is clearly closer overall.

Return JSON only:
{
  "preference": "A|B|tie|neither",
  "reason": "brief explanation",
  "matched_A_only_values": ["..."],
  "matched_B_only_values": ["..."]
}
"""

FORCED_CHOICE_SYSTEM_PROMPT = """You are comparing a model answer against two counterfactual worlds.

You will receive:
- a question
- a model output
- World A reference
- World B reference
- values unique to World A
- values unique to World B

Task:
Decide which world the model output is semantically closer to.

Rules:
- Focus on factual meaning, not wording.
- You must always choose either "A" or "B".
- Even if the answer is imperfect, vague, partially wrong, or overlaps with both worlds, choose the world it is closer to overall.
- If one world matches more of the answer's factual content, entities, values, or direction of answer than the other, choose that world.
- Break ties by choosing the world that better matches the answer's main factual claim.

Return JSON only:
{
  "preference": "A|B",
  "reason": "brief explanation",
  "matched_A_only_values": ["..."],
  "matched_B_only_values": ["..."]
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


def _clean_prediction_text(prediction: str, question: str) -> str:
    cleaned = str(prediction or "").strip()
    matches = list(_ANSWER_TAG_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()
    q_norm = _single_line(question)
    c_norm = _single_line(cleaned)
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


def _build_rows(subset_payload: Mapping[str, Any], model_labels: Sequence[str], order_seed: int) -> List[Dict[str, Any]]:
    examples = subset_payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Expected examples list.")
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
        question = str(example.get("question", ""))
        for model_label in model_labels:
            if model_label not in predictions:
                continue
            row_id += 1
            local_rng = random.Random(f"{order_seed}|{metadata.get('query_index')}|{model_label}")
            a_is_aligned = bool(local_rng.randint(0, 1))
            rows.append(
                {
                    "row_id": row_id,
                    "query_index": metadata.get("query_index"),
                    "model_label": model_label,
                    "question": question,
                    "model_output": _clean_prediction_text(str(predictions[model_label]), question),
                    "a_is_aligned": a_is_aligned,
                    "world_A_reference": aligned_text if a_is_aligned else null_text,
                    "world_B_reference": null_text if a_is_aligned else aligned_text,
                    "world_A_only_values": list(metadata.get("aligned_only_values", [])) if a_is_aligned else list(metadata.get("null_only_values", [])),
                    "world_B_only_values": list(metadata.get("null_only_values", [])) if a_is_aligned else list(metadata.get("aligned_only_values", [])),
                    "aligned_reference": aligned_text,
                    "nullified_reference": null_text,
                    "rouge_preference": _compute_rouge_preference(
                        _clean_prediction_text(str(predictions[model_label]), question),
                        aligned_text,
                        null_text,
                    ),
                }
            )
    return rows


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


def _call_openai_judge(
    client: OpenAI,
    model_name: str,
    user_prompt: str,
    seed: int,
    *,
    system_prompt: str,
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
        raise ValueError("Missing text content.")
    return content.strip()


def _parse_response(raw_text: str, *, valid_prefs: Sequence[str]) -> Dict[str, Any]:
    payload = json.loads(raw_text)
    pref = str(payload.get("preference", "")).strip()
    valid_pref_set = set(valid_prefs)
    if pref not in valid_pref_set:
        raise ValueError(f"Invalid preference label: {pref}")

    def _clean_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [_single_line(str(v)) for v in value if _single_line(str(v))]

    return {
        "preference": pref,
        "reason": _single_line(str(payload.get("reason", ""))),
        "matched_A_only_values": _clean_list(payload.get("matched_A_only_values", [])),
        "matched_B_only_values": _clean_list(payload.get("matched_B_only_values", [])),
    }


def _map_pref(pref: str, a_is_aligned: bool) -> str:
    if pref == "A":
        return "aligned" if a_is_aligned else "nullified"
    if pref == "B":
        return "nullified" if a_is_aligned else "aligned"
    return pref


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_model: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_label"])].append(row)
    out: Dict[str, Any] = {}
    for label, model_rows in sorted(by_model.items()):
        total = len(model_rows)
        pref_counts = Counter(str(r["judge_mapped_preference"]) for r in model_rows)
        agree = Counter()
        decisive = [r for r in model_rows if str(r["judge_mapped_preference"]) in {"aligned", "nullified"}]
        for row in decisive:
            judge_pref = str(row["judge_mapped_preference"])
            rouge_pref = str(row["rouge_preference"]["preference"])
            if rouge_pref == judge_pref:
                agree["same_side"] += 1
            elif rouge_pref == "tie":
                agree["tie"] += 1
            else:
                agree["opposite_side"] += 1
        def rate(c:int,d:int)->float:
            return float(c/d) if d else 0.0
        out[label] = {
            "total_rows": total,
            "preference_counts": dict(pref_counts),
            "preference_rates": {k: rate(pref_counts.get(k,0), total) for k in ["aligned","nullified","tie","neither"]},
            "decisive_rows": len(decisive),
            "decisive_rate": rate(len(decisive), total),
            "rouge_agreement_on_decisive": {
                "counts": dict(agree),
                "rates": {k: rate(agree.get(k,0), len(decisive)) for k in ["same_side","tie","opposite_side"]},
            },
        }
    return {"row_count": len(rows), "query_count": len({int(r["query_index"]) for r in rows}), "by_model": out}


def _render_md(summary: Mapping[str, Any], judge_model: str, subset_json: str, annotations_path: str) -> str:
    lines = [
        "# LLM Counterfactual Preference Judge Summary",
        "",
        f"- Judge model: `{judge_model}`",
        f"- Subset: `{subset_json}`",
        f"- Row annotations: `{annotations_path}`",
        f"- Total judged rows: `{summary['row_count']}`",
        f"- Unique queries: `{summary['query_count']}`",
        "",
        "## Preference Rates",
        "",
        "| Model | Aligned | Nullified | Tie | Neither | Decisive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, payload in summary["by_model"].items():
        pref = payload["preference_rates"]
        lines.append(
            f"| {label} | {100*pref.get('aligned',0):.1f}% | {100*pref.get('nullified',0):.1f}% | {100*pref.get('tie',0):.1f}% | {100*pref.get('neither',0):.1f}% | {100*payload.get('decisive_rate',0):.1f}% |"
        )
    lines += [
        "",
        "## Judge vs Two-Reference ROUGE",
        "",
        "| Model | Same side | ROUGE tie | Opposite side |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, payload in summary["by_model"].items():
        agree = payload["rouge_agreement_on_decisive"]["rates"]
        lines.append(
            f"| {label} | {100*agree.get('same_side',0):.1f}% | {100*agree.get('tie',0):.1f}% | {100*agree.get('opposite_side',0):.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--model_label", action="append", default=[])
    parser.add_argument("--judge_model", default="gpt-5.4-nano-2026-03-17")
    parser.add_argument("--order_seed", type=int, default=20260330)
    parser.add_argument("--api_seed", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay_sec", type=float, default=2.0)
    parser.add_argument("--force_choice", action="store_true")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    payload = _read_json(args.subset_json)
    labels = args.model_label or ["Morig", "Mret"]
    rows = _build_rows(payload, labels, args.order_seed)
    client = OpenAI()
    system_prompt = FORCED_CHOICE_SYSTEM_PROMPT if args.force_choice else SYSTEM_PROMPT
    valid_prefs = VALID_PREF_FORCED if args.force_choice else VALID_PREF
    judged_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = _build_user_prompt(row)
        parsed: Optional[Dict[str, Any]] = None
        raw_response: Optional[str] = None
        last_error: Optional[str] = None
        for _attempt in range(args.max_retries):
            try:
                raw_response = _call_openai_judge(
                    client,
                    args.judge_model,
                    prompt,
                    args.api_seed,
                    system_prompt=system_prompt,
                )
                parsed = _parse_response(raw_response, valid_prefs=valid_prefs)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(args.retry_delay_sec)
        if parsed is None:
            raise RuntimeError(f"Judge failed for row {row['row_id']}: {last_error}")
        judged = dict(row)
        judged["judge"] = parsed
        judged["judge_mapped_preference"] = _map_pref(parsed["preference"], bool(row["a_is_aligned"]))
        judged["judge_raw_response"] = raw_response
        judged_rows.append(judged)
        print(f"[{idx}/{len(rows)}] qidx={row['query_index']} model={row['model_label']} pref={judged['judge_mapped_preference']}")

    summary = _aggregate(judged_rows)
    summary["judge_model"] = args.judge_model
    summary["subset_json"] = args.subset_json
    summary["force_choice"] = bool(args.force_choice)

    out_prefix = Path(args.out_prefix)
    jsonl_path = out_prefix.with_suffix(".jsonl")
    summary_json = out_prefix.with_name(out_prefix.name + "_summary.json")
    summary_md = out_prefix.with_name(out_prefix.name + "_summary.md")
    _write_jsonl(jsonl_path, judged_rows)
    _write_json(summary_json, summary)
    summary_md.write_text(_render_md(summary, args.judge_model, args.subset_json, str(jsonl_path)), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
