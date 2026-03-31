"""Post-hoc two-reference evaluation for RAQUEL affected predictions.

Why this evaluation?
--------------------
Affected RAQUEL queries have two relevant counterfactual answers:
  - aligned answer    (pre-forget world)
  - nullified answer  (post-forget counterfactual world)

The original paper reports only similarity to the aligned reference. This script
reuses saved model generations from `run_raquel_eval.py --save_predictions` and
compares each prediction against *both* references.

Important design choice:
  Both references are generated from denotations using the same deterministic
  textualizer, so the comparison is symmetric. We intentionally do not mix the
  original aligned natural-language answer with a newly serialized null answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from torchmetrics.functional.text.rouge import rouge_score
except ImportError:
    rouge_score = None

from src.utils.logging import get_logger

logger = get_logger("script.evaluation.run_raquel_two_reference_eval", __file__)

_WS_RE = re.compile(r"\s+")
_ANSWER_TAG_RE = re.compile(r"(?i)\banswer\s*:\s*")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(_WS_RE.sub(" ", str(text).strip()).lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _overlap_count(
    pred_counts: Mapping[Tuple[str, ...], int],
    ref_counts: Mapping[Tuple[str, ...], int],
) -> int:
    total = 0
    for gram, pred_count in pred_counts.items():
        ref_count = ref_counts.get(gram, 0)
        total += min(pred_count, ref_count)
    return total


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


def _fallback_rouge_score(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    rouge_keys: Sequence[str],
) -> Dict[str, float]:
    if len(predictions) != 1 or len(references) != 1:
        raise ValueError("Fallback ROUGE scorer expects exactly one prediction and one reference.")

    pred_tokens = _tokenize(predictions[0])
    ref_tokens = _tokenize(references[0])

    scores: Dict[str, float] = {}
    if "rouge1" in rouge_keys:
        pred_counts = _ngram_counts(pred_tokens, 1)
        ref_counts = _ngram_counts(ref_tokens, 1)
        overlap = _overlap_count(pred_counts, ref_counts)
        scores["rouge1_fmeasure"] = _safe_fmeasure(overlap, sum(pred_counts.values()), sum(ref_counts.values()))
    if "rouge2" in rouge_keys:
        pred_counts = _ngram_counts(pred_tokens, 2)
        ref_counts = _ngram_counts(ref_tokens, 2)
        overlap = _overlap_count(pred_counts, ref_counts)
        scores["rouge2_fmeasure"] = _safe_fmeasure(overlap, sum(pred_counts.values()), sum(ref_counts.values()))
    if "rougeL" in rouge_keys:
        overlap = _lcs_length(pred_tokens, ref_tokens)
        scores["rougeL_fmeasure"] = _safe_fmeasure(overlap, len(pred_tokens), len(ref_tokens))
    return scores


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            qidx = record.get("query_index")
            if not isinstance(qidx, int):
                continue
            out[int(qidx)] = record
    return out


def _normalize_question(text: str) -> str:
    return _WS_RE.sub(" ", str(text).strip())


def _clean_prediction_text(prediction: str, question: str) -> str:
    """Remove obvious prompt-echo scaffolding from saved generations."""
    cleaned = str(prediction).strip()

    matches = list(_ANSWER_TAG_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()

    question_norm = _normalize_question(question)
    cleaned_norm = _normalize_question(cleaned)
    if question_norm and cleaned_norm.lower().startswith(question_norm.lower()):
        cleaned = cleaned[len(question_norm) :].strip(" \n\r\t:.-")
        cleaned_norm = _normalize_question(cleaned)

    if question_norm and cleaned_norm:
        prefix_chars = max(12, min(len(question_norm), 48))
        question_prefix = question_norm[:prefix_chars].lower()
        if question_prefix and cleaned_norm.lower().startswith(question_prefix):
            qmark_idx = cleaned.find("?")
            if qmark_idx >= 0 and qmark_idx + 1 < len(cleaned):
                candidate = cleaned[qmark_idx + 1 :].strip(" \n\r\t:.-")
                if candidate:
                    cleaned = candidate

    return cleaned.strip()


def _load_question_index_map(
    path: str,
    *,
    split: str,
) -> Dict[str, Deque[Dict[str, Any]]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")

    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("split", "")).strip() != split:
            continue
        question = _normalize_question(str(row.get("question", "")))
        if not question:
            continue
        grouped[question].append(row)

    out: Dict[str, Deque[Dict[str, Any]]] = {}
    for question, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item.get("example_index", 0)))
        out[question] = deque(rows)
    return out


def _resolve_paths(
    *,
    model_dir: Optional[str],
    predictions_path: Optional[str],
    output_path: Optional[str],
) -> tuple[str, str]:
    if predictions_path:
        pred_path = predictions_path
    elif model_dir:
        pred_path = os.path.join(model_dir, "raquel_eval_with_predictions.json")
    else:
        raise ValueError("Either model_dir or predictions_path must be provided.")

    if output_path:
        out_path = output_path
    else:
        out_path = os.path.join(os.path.dirname(pred_path), "raquel_two_reference_eval.json")
    return pred_path, out_path


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = _WS_RE.sub(" ", value).strip()
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


def _to_float_dict(scores: Mapping[str, Any]) -> Dict[str, float]:
    return {name: float(value) for name, value in scores.items()}


def _score_text(prediction: str, reference: str) -> Dict[str, float]:
    if rouge_score is not None:
        return _to_float_dict(
            rouge_score(
                [prediction],
                [reference],
                rouge_keys=("rouge1", "rouge2", "rougeL"),
                use_stemmer=True,
            )
        )
    return _fallback_rouge_score(
        [prediction],
        [reference],
        rouge_keys=("rouge1", "rouge2", "rougeL"),
    )


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run symmetric two-reference evaluation on saved RAQUEL predictions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir")
    group.add_argument("--predictions_path")
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument("--split_name", default="affected")
    parser.add_argument(
        "--output_path",
        help="Output path (default: alongside predictions as raquel_two_reference_eval.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictions_path, output_path = _resolve_paths(
        model_dir=args.model_dir,
        predictions_path=args.predictions_path,
        output_path=args.output_path,
    )

    payload = _read_json(predictions_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {predictions_path}")
    split_payload = payload.get(args.split_name)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split {args.split_name!r} in {predictions_path}")
    predictions = split_payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Missing predictions for split {args.split_name!r} in {predictions_path}")

    question_map = _load_question_index_map(args.query_index_map, split=args.split_name)
    denotations = _read_jsonl_denotations(args.denotations)

    rows: List[Dict[str, Any]] = []
    unmatched_questions: List[str] = []
    missing_denotations: List[int] = []

    for pred_row in predictions:
        if not isinstance(pred_row, dict):
            continue
        question = _normalize_question(str(pred_row.get("question", "")))
        match_queue = question_map.get(question)
        if not match_queue:
            unmatched_questions.append(question)
            continue
        matched = match_queue.popleft()
        qidx = int(matched["query_index"])
        denotation = denotations.get(qidx)
        if denotation is None:
            missing_denotations.append(qidx)
            continue

        raw_prediction_text = str(pred_row.get("prediction", "")).strip()
        prediction_text = _clean_prediction_text(raw_prediction_text, str(pred_row.get("question", "")))
        aligned_ref = _textualize_denotation(denotation.get("result_aligned"))
        null_ref = _textualize_denotation(denotation.get("result_null"))

        aligned_scores = _score_text(prediction_text, aligned_ref)
        null_scores = _score_text(prediction_text, null_ref)

        aligned_rl = float(aligned_scores.get("rougeL_fmeasure", 0.0))
        null_rl = float(null_scores.get("rougeL_fmeasure", 0.0))
        delta = float(aligned_rl - null_rl)
        if delta > 0.0:
            preferred = "aligned"
        elif delta < 0.0:
            preferred = "null"
        else:
            preferred = "tie"

        rows.append(
            {
                "query_index": qidx,
                "question": str(pred_row.get("question", "")),
                "raw_prediction": raw_prediction_text,
                "prediction": prediction_text,
                "aligned_reference": aligned_ref,
                "null_reference": null_ref,
                "aligned_scores": aligned_scores,
                "null_scores": null_scores,
                "delta_rougeL_fmeasure": delta,
                "preferred_side": preferred,
                "original_aligned_reference": str(pred_row.get("reference", "")),
            }
        )

    prefer_aligned = sum(1 for row in rows if row["preferred_side"] == "aligned")
    prefer_null = sum(1 for row in rows if row["preferred_side"] == "null")
    ties = sum(1 for row in rows if row["preferred_side"] == "tie")

    result: Dict[str, Any] = {
        args.split_name: {
            "count": len(rows),
            "mean_similarity_to_aligned": {
                key: _mean(row["aligned_scores"][key] for row in rows)
                for key in (
                    "rouge1_fmeasure",
                    "rouge2_fmeasure",
                    "rougeL_fmeasure",
                )
            },
            "mean_similarity_to_null": {
                key: _mean(row["null_scores"][key] for row in rows)
                for key in (
                    "rouge1_fmeasure",
                    "rouge2_fmeasure",
                    "rougeL_fmeasure",
                )
            },
            "delta": {
                "rougeL_fmeasure": _mean(row["delta_rougeL_fmeasure"] for row in rows),
            },
            "preference": {
                "prefer_aligned_rate": float(prefer_aligned / len(rows)) if rows else 0.0,
                "prefer_null_rate": float(prefer_null / len(rows)) if rows else 0.0,
                "tie_rate": float(ties / len(rows)) if rows else 0.0,
            },
            "unmatched_questions": len(unmatched_questions),
            "missing_denotations": len(missing_denotations),
            "rows": rows,
        },
        "metadata": {
            "predictions_path": predictions_path,
            "query_index_map": args.query_index_map,
            "denotations": args.denotations,
            "split_name": args.split_name,
            "reference_mode": "deterministic_denotation_text",
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    logger.info("Wrote %s", output_path)

    if unmatched_questions:
        logger.warning("Unmatched questions: %d", len(unmatched_questions))
    if missing_denotations:
        logger.warning("Missing denotations: %d", len(missing_denotations))


if __name__ == "__main__":
    main()
