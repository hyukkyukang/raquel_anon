"""Denotation-based RAQUEL evaluation (aligned vs null oracles).

This evaluator addresses the core critique that Affected queries have two denotations:
  - aligned execution result: E(q, D)          (pre-forget)
  - nullified execution result: E(q, D_null)  (counterfactual post-forget)

Given:
  - RAQUEL question lists (affected/unaffected JSON),
  - query_index_map.json (question -> query_index),
  - denotations/by_index.jsonl (query_index -> result_aligned/result_null),
the script prompts the model to output a JSON table and computes row-set F1 against
both denotations (for affected) and the shared denotation (for unaffected).

No external APIs are used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from script.evaluation.utils import load_fine_tuned_model
from src.utils.logging import get_logger

logger = get_logger("script.evaluation.run_raquel_denotation_eval", __file__)


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in payload if isinstance(item, dict)]


@dataclass(frozen=True)
class QueryIndexRow:
    split: str
    query_index: int
    example_index: int
    question: str


def _load_query_index_map(path: str) -> Dict[str, List[QueryIndexRow]]:
    rows_raw = _load_json_list(path)
    by_split: Dict[str, List[QueryIndexRow]] = {"affected": [], "unaffected": []}
    for row in rows_raw:
        split = str(row.get("split", "")).strip()
        if split not in by_split:
            continue
        try:
            qidx = int(row["query_index"])
            ex_idx = int(row.get("example_index", 0))
        except Exception:
            continue
        question = str(row.get("question", "")).strip()
        by_split[split].append(
            QueryIndexRow(
                split=split, query_index=qidx, example_index=ex_idx, question=question
            )
        )
    for split in by_split:
        by_split[split] = sorted(by_split[split], key=lambda r: r.example_index)
    return by_split


def _load_denotations_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record: Any = json.loads(line)
            if not isinstance(record, dict):
                continue
            qidx = record.get("query_index")
            if not isinstance(qidx, int):
                continue
            by_index[int(qidx)] = record
    return by_index


def _resolve_device(
    model: PreTrainedModel, requested_device: Optional[str]
) -> torch.device:
    try:
        model_device: torch.device = next(model.parameters()).device
    except StopIteration:
        fallback: str = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(fallback)
    if requested_device and str(model_device) != requested_device:
        logger.warning(
            "Requested device %s but model is on %s; using model device.",
            requested_device,
            model_device,
        )
    return model_device


def _build_prompt(question: str, columns: Sequence[str]) -> str:
    """Build a strict JSON-only prompt.

    Important: we intentionally do NOT use the training `ANSWER_PREFIX` ("Answer:")
    here, because it encourages models to emit "Answer: ..." and break JSON.
    """
    cols = ", ".join(f'"{c}"' for c in columns)
    example_row = ", ".join(f'"{c}": null' for c in columns)
    return (
        "Return ONLY valid JSON.\n"
        "Do NOT include any extra text (no 'Answer:', no markdown, no explanations).\n"
        f"Output format: a JSON array (list) of row objects with keys: [{cols}]\n"
        "- Use null for missing values.\n"
        "- If there are no results, return: []\n"
        f"Example: [{{{example_row}}}]\n\n"
        f"Question: {question}\n"
        "JSON:\n"
    )


def _generate(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: Optional[int],
    device: torch.device,
) -> List[str]:
    predictions: List[str] = []
    was_training: bool = model.training
    model.eval()

    original_padding_side: Optional[str] = getattr(tokenizer, "padding_side", None)
    if original_padding_side is not None:
        tokenizer.padding_side = "left"

    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                # With left padding, generated sequences include the *full padded input*.
                # Slice using the padded input length, not the unpadded prompt length.
                input_seq_len = int(input_ids.size(1))

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                for row_idx in range(generated.size(0)):
                    gen_ids = generated[row_idx][input_seq_len:]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    predictions.append(text)
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    return predictions


_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_NUMBER_TOKEN_RE = re.compile(
    r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
)
_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")


def _normalize_text_token(text: str) -> str:
    """Normalize a text token for approximate matching.

    This intentionally keeps whitespace (collapsed) and lowercases; we avoid heavy
    NLP to keep this evaluation lightweight and deterministic.
    """
    normalized = text.strip().strip('"').strip("'").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" \t\r\n,.;:()[]{}")
    return normalized.lower()


def _try_parse_float(value: Any) -> Optional[float]:
    """Parse numeric-like values into float (best effort)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none"}:
            return None
        # Remove thousands separators.
        text = text.replace(",", "")
        # Handle percent-encoded values.
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            parsed = float(text)
        except Exception:
            return None
        return parsed / 100.0 if is_percent else parsed
    return None


def _extract_values_from_rows(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
    """Extract normalized (string, float) atoms from a denotation table."""
    str_atoms: List[str] = []
    num_atoms: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            num = _try_parse_float(value)
            if num is not None:
                num_atoms.append(num)
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                token = _normalize_text_token(value)
                if token and token not in {"null", "none"}:
                    str_atoms.append(token)
                continue
            token = _normalize_text_token(str(value))
            if token and token not in {"null", "none"}:
                str_atoms.append(token)

    # De-duplicate while keeping deterministic ordering.
    str_unique = sorted(set(str_atoms))
    num_unique = sorted(set(num_atoms))
    return str_unique, num_unique


def _extract_values_from_text(text: str) -> Tuple[List[str], List[float]]:
    """Extract approximate (string, float) atoms from free-form model output.

    We intentionally use simple regex-based extraction; this is a fallback for
    models that do not reliably emit strict JSON tables.
    """
    stripped = text.strip()
    if not stripped:
        return [], []

    lower = stripped.lower()

    # 1) Numbers (including commas, scientific notation, and percents).
    num_atoms: List[float] = []
    for match in _NUMBER_TOKEN_RE.findall(lower):
        parsed = _try_parse_float(match)
        if parsed is not None:
            num_atoms.append(parsed)

    # 2) Strings: quoted spans + word n-grams (1..4).
    str_atoms: List[str] = []
    for quoted in re.findall(r"\"([^\"]+)\"", stripped):
        token = _normalize_text_token(quoted)
        if token:
            str_atoms.append(token)
    for quoted in re.findall(r"'([^']+)'", stripped):
        token = _normalize_text_token(quoted)
        if token:
            str_atoms.append(token)

    words: List[str] = [_normalize_text_token(w) for w in _WORD_TOKEN_RE.findall(stripped)]
    words = [w for w in words if w and w not in {"answer", "question"}]
    max_ngram = 4
    for n in range(1, max_ngram + 1):
        for i in range(0, max(0, len(words) - n + 1)):
            ngram = " ".join(words[i : i + n]).strip()
            if ngram:
                str_atoms.append(ngram)

    str_unique = sorted(set(str_atoms))
    num_unique = sorted(set(num_atoms))
    return str_unique, num_unique


def _match_float_values(
    *, gold: Sequence[float], predicted: Sequence[float], atol: float = 1e-6, rtol: float = 1e-4
) -> int:
    """Greedy match of float values under tolerance (counts matches)."""
    if not gold or not predicted:
        return 0
    used: List[bool] = [False for _ in range(len(predicted))]
    matches = 0
    for g in gold:
        tol = max(atol, abs(g) * rtol)
        for j, p in enumerate(predicted):
            if used[j]:
                continue
            if abs(p - g) <= tol:
                used[j] = True
                matches += 1
                break
    return matches


def _value_set_f1_from_text(
    *, prediction_text: str, gold_rows: List[Dict[str, Any]]
) -> float:
    """Compute a lightweight value-set F1 between model output and gold denotation."""
    gold_str, gold_num = _extract_values_from_rows(gold_rows)
    pred_str, pred_num = _extract_values_from_text(prediction_text)

    gold_count = len(gold_str) + len(gold_num)
    pred_count = len(pred_str) + len(pred_num)

    if gold_count == 0 and pred_count == 0:
        return 1.0
    if gold_count == 0 or pred_count == 0:
        return 0.0

    matched_str = len(set(gold_str) & set(pred_str))
    matched_num = _match_float_values(gold=gold_num, predicted=pred_num)
    matched = matched_str + matched_num

    precision = matched / pred_count if pred_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _canonical_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Normalize floats that are effectively integers.
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in {"null", "none"}:
            return "null"
        if _INT_RE.match(text):
            return str(int(text))
        if _FLOAT_RE.match(text):
            return str(float(text))
        return text
    return str(value).strip()


def _row_signature(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    normalized = {col: _canonical_value(row.get(col)) for col in columns}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _normalize_rows(payload: Any, columns: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    """Normalize a decoded JSON payload into a list of row dicts."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "rows" in payload and isinstance(payload["rows"], list):
            payload = payload["rows"]
        else:
            # A single row object.
            payload = [payload]

    if not isinstance(payload, list):
        return None

    rows: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append({str(k): v for k, v in item.items()})
            continue
        if isinstance(item, (list, tuple)):
            # Interpret as positional columns.
            as_row: Dict[str, Any] = {}
            for idx, col in enumerate(columns):
                if idx < len(item):
                    as_row[col] = item[idx]
            rows.append(as_row)
            continue
        # Scalar -> put into first column if available.
        if columns:
            rows.append({columns[0]: item})
    return rows


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort extraction of JSON from model output."""
    stripped = text.strip()
    if not stripped:
        return None
    # First try: whole string.
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # Try to extract a JSON array.
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try to extract a JSON object.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def _row_set_f1(
    *, predicted_rows: List[Dict[str, Any]], gold_rows: List[Dict[str, Any]], columns: Sequence[str]
) -> float:
    pred_set = {_row_signature(row, columns) for row in predicted_rows}
    gold_set = {_row_signature(row, columns) for row in gold_rows}

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    inter = len(pred_set & gold_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(gold_set) if gold_set else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _columns_for_denotation(aligned_rows: List[Dict[str, Any]], null_rows: List[Dict[str, Any]]) -> List[str]:
    cols: set[str] = set()
    for row in aligned_rows:
        cols.update(str(k) for k in row.keys())
    for row in null_rows:
        cols.update(str(k) for k in row.keys())
    return sorted(cols)


def _evaluate_split(
    *,
    split_name: str,
    rows: List[QueryIndexRow],
    denotations: Dict[int, Dict[str, Any]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: Optional[int],
    max_examples: int,
    shuffle: bool,
    sample_seed: int,
    max_denotation_rows: Optional[int],
    max_denotation_cols: Optional[int],
    save_failures: int,
) -> Dict[str, Any]:
    """Evaluate a split with dual-oracle denotation scoring.

    Notes:
    - We optionally filter by denotation size (rows/cols) because large result tables
      are hard for LLMs to emit as strict JSON.
    - We optionally shuffle before truncating to `max_examples` to avoid order bias.
    """
    prompts: List[str] = []
    columns_by_example: List[List[str]] = []
    aligned_gold: List[List[Dict[str, Any]]] = []
    null_gold: List[List[Dict[str, Any]]] = []
    query_indices: List[int] = []

    eligible_total = len(rows)
    kept: List[Tuple[QueryIndexRow, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]] = []
    skipped_rows = 0
    skipped_cols = 0

    for r in rows:
        record = denotations.get(r.query_index)
        if record is None:
            raise ValueError(f"Missing denotation for query_index={r.query_index}")
        a_rows = record.get("result_aligned") or []
        n_rows = record.get("result_null") or []
        if not isinstance(a_rows, list) or not isinstance(n_rows, list):
            raise ValueError(f"Invalid denotation rows for query_index={r.query_index}")
        cols = _columns_for_denotation(a_rows, n_rows)
        if max_denotation_rows is not None:
            if len(a_rows) > max_denotation_rows or len(n_rows) > max_denotation_rows:
                skipped_rows += 1
                continue
        if max_denotation_cols is not None and len(cols) > max_denotation_cols:
            skipped_cols += 1
            continue
        kept.append((r, a_rows, n_rows, cols))

    if shuffle:
        rng = random.Random(int(sample_seed))
        rng.shuffle(kept)

    if max_examples > 0:
        kept = kept[:max_examples]

    for r, a_rows, n_rows, cols in kept:
        prompts.append(_build_prompt(r.question, cols))
        columns_by_example.append(cols)
        aligned_gold.append(a_rows)
        null_gold.append(n_rows)
        query_indices.append(r.query_index)

    outputs: List[str] = _generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_prompt_length,
        device=device,
    )
    if len(outputs) != len(kept):
        raise RuntimeError("Generation output size mismatch.")

    row_f1_aligned: List[float] = []
    row_f1_null: List[float] = []
    value_f1_aligned: List[float] = []
    value_f1_null: List[float] = []
    parse_ok = 0
    failures: List[Dict[str, Any]] = []

    for i, text in enumerate(outputs):
        cols = columns_by_example[i]
        decoded = _extract_json(text)
        parsed_rows = _normalize_rows(decoded, cols)
        if parsed_rows is None:
            failures.append(
                {
                    "query_index": query_indices[i],
                    "question": kept[i][0].question,
                    "raw_output": text,
                }
            )
            parsed_rows = []
        else:
            parse_ok += 1

        row_f1_aligned.append(
            _row_set_f1(
                predicted_rows=parsed_rows,
                gold_rows=aligned_gold[i],
                columns=cols,
            )
        )
        row_f1_null.append(
            _row_set_f1(
                predicted_rows=parsed_rows,
                gold_rows=null_gold[i],
                columns=cols,
            )
        )
        # Value-level fallback is computed from the raw text, even if JSON parsing succeeded.
        value_f1_aligned.append(
            _value_set_f1_from_text(prediction_text=text, gold_rows=aligned_gold[i])
        )
        value_f1_null.append(
            _value_set_f1_from_text(prediction_text=text, gold_rows=null_gold[i])
        )

    mean_row_aligned = (
        float(sum(row_f1_aligned) / len(row_f1_aligned)) if row_f1_aligned else 0.0
    )
    mean_row_null = float(sum(row_f1_null) / len(row_f1_null)) if row_f1_null else 0.0
    mean_value_aligned = (
        float(sum(value_f1_aligned) / len(value_f1_aligned)) if value_f1_aligned else 0.0
    )
    mean_value_null = (
        float(sum(value_f1_null) / len(value_f1_null)) if value_f1_null else 0.0
    )

    result: Dict[str, Any] = {
        "split": split_name,
        "count": len(kept),
        "eligible_total": int(eligible_total),
        "filtered_total": int(len(kept)),
        "skipped_by_max_rows": int(skipped_rows),
        "skipped_by_max_cols": int(skipped_cols),
        "parse_success_rate": float(parse_ok / len(kept)) if kept else 0.0,
        "denotation_row_f1_to_aligned": mean_row_aligned,
        "denotation_row_f1_to_null": mean_row_null,
        "denotation_value_f1_to_aligned": mean_value_aligned,
        "denotation_value_f1_to_null": mean_value_null,
    }
    if save_failures > 0 and failures:
        result["parse_failures"] = failures[:save_failures]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run denotation-based RAQUEL evaluation.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle eligible examples before truncating to --max_examples.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="RNG seed used when --shuffle is set.",
    )
    parser.add_argument(
        "--max_denotation_rows",
        type=int,
        default=0,
        help="If > 0, skip examples with aligned/null denotation rows exceeding this.",
    )
    parser.add_argument(
        "--max_denotation_cols",
        type=int,
        default=0,
        help="If > 0, skip examples whose (aligned ∪ null) column count exceeds this.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument(
        "--output_path",
        help="Output path (default: model_path/raquel_denotation_eval.json).",
    )
    parser.add_argument("--save_failures", type=int, default=20)
    args = parser.parse_args()

    index_map = _load_query_index_map(args.query_index_map)
    denotations = _load_denotations_jsonl(args.denotations)

    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        base_model_name=args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    device = _resolve_device(model, args.device)

    affected_result = _evaluate_split(
        split_name="affected",
        rows=index_map["affected"],
        denotations=denotations,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        max_prompt_length=args.max_prompt_length,
        max_examples=int(args.max_examples),
        shuffle=bool(args.shuffle),
        sample_seed=int(args.sample_seed),
        max_denotation_rows=int(args.max_denotation_rows) if int(args.max_denotation_rows) > 0 else None,
        max_denotation_cols=int(args.max_denotation_cols) if int(args.max_denotation_cols) > 0 else None,
        save_failures=int(args.save_failures),
    )
    unaffected_result = _evaluate_split(
        split_name="unaffected",
        rows=index_map["unaffected"],
        denotations=denotations,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        max_prompt_length=args.max_prompt_length,
        max_examples=int(args.max_examples),
        shuffle=bool(args.shuffle),
        sample_seed=int(args.sample_seed),
        max_denotation_rows=int(args.max_denotation_rows) if int(args.max_denotation_rows) > 0 else None,
        max_denotation_cols=int(args.max_denotation_cols) if int(args.max_denotation_cols) > 0 else None,
        save_failures=int(args.save_failures),
    )

    output_path = (
        args.output_path
        if args.output_path
        else os.path.join(args.model_path, "raquel_denotation_eval.json")
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "affected": affected_result,
        "unaffected": unaffected_result,
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "device": str(device),
            "query_index_map": args.query_index_map,
            "denotations": args.denotations,
            "max_examples": int(args.max_examples),
            "batch_size": int(args.batch_size),
            "max_new_tokens": int(args.max_new_tokens),
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Denotation eval saved to %s", output_path)


if __name__ == "__main__":
    main()

