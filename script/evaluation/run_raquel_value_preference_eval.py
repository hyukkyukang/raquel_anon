"""Value-level preference evaluation for RAQUEL affected queries.

This script avoids free-form generation by scoring *candidate values* under
teacher forcing.

For each affected RAQUEL query q, we compute:
  - aligned-only value set: V_a = values(result_aligned) \\ values(result_null)
  - null-only value set:    V_n = values(result_null)   \\ values(result_aligned)

We then score short candidate answers consisting of individual values v in V_a
or V_n using average log-probability per answer token:
  score(v) = (1/|v_tokens|) * log p(v | prompt(q))

For each query we aggregate scores over up to K sampled values per side:
  score_aligned(q) = mean_{v in sample(V_a)} score(v)
  score_null(q)    = mean_{v in sample(V_n)} score(v)

Finally we report preference statistics on queries where both sides are non-empty:
  P(score_aligned > score_null), mean margin, etc.

Optionally, we can restrict candidate values to those that were removed during
nullification (high-precision forget-attributable entities).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from script.evaluation.utils import load_fine_tuned_model
from src.training.data.transforms import ANSWER_PREFIX
from src.utils.logging import get_logger

logger = get_logger("script.evaluation.run_raquel_value_preference_eval", __file__)

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+")  # hyphen variants
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'`.,;:()\\[\\]{}<>]+|[\s\"'`.,;:()\\[\\]{}<>]+$")


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
            rec: Any = json.loads(line)
            if not isinstance(rec, dict):
                continue
            qidx = rec.get("query_index")
            if not isinstance(qidx, int):
                continue
            out[int(qidx)] = rec
    return out


@dataclass(frozen=True)
class QueryIndexRow:
    query_index: int
    example_index: int
    question: str


def _load_query_index_rows(path: str, *, split: str) -> List[QueryIndexRow]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    rows: List[QueryIndexRow] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("split", "")).strip() != split:
            continue
        try:
            qidx = int(item["query_index"])
            ex_idx = int(item.get("example_index", 0))
        except Exception:
            continue
        question = str(item.get("question", "")).strip()
        rows.append(QueryIndexRow(query_index=qidx, example_index=ex_idx, question=question))
    return sorted(rows, key=lambda r: r.example_index)


def _normalize_value_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("_", " ")
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_EDGE_RE.sub("", normalized)
    return normalized.strip()


def _is_salient_string(value_norm: str) -> bool:
    if not value_norm or value_norm in {"null", "none"}:
        return False
    if len(value_norm) < 3:
        return False
    # Avoid obvious generic placeholders.
    stop = {
        "books",
        "works",
        "mother",
        "father",
        "authors",
        "author",
        "publishers",
        "archives",
        "libraries",
    }
    if value_norm in stop:
        return False
    return True


def _extract_string_values(rows: Any) -> Dict[str, str]:
    """Extract string cell values from a denotation table.

    Returns:
      Mapping from normalized value -> a representative *original* surface string.
    """
    if not isinstance(rows, list):
        return {}
    out: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for v in row.values():
            if not isinstance(v, str):
                continue
            original = v.strip()
            if not original:
                continue
            token = _normalize_value_text(original)
            if not _is_salient_string(token):
                continue
            # Keep the first-seen surface form for determinism.
            out.setdefault(token, original)
    return out


def _load_removed_entity_values(path: str) -> Set[str]:
    """Load removed (forget-attributable) entity values as normalized strings."""
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        table = str(row.get("table", "")).strip()
        column = str(row.get("column", "")).strip()
        value = row.get("value")
        if not isinstance(value, str):
            continue
        if not ((table == "person" and column == "name") or (table == "work" and column == "title")):
            continue
        token = _normalize_value_text(value)
        if _is_salient_string(token) and (" " in token):  # prefer names/titles
            out.add(token)
    return out


def _build_prompt(question: str) -> str:
    return f"Question: {question.strip()}{ANSWER_PREFIX}"


def _resolve_device(model: PreTrainedModel, requested_device: Optional[str]) -> torch.device:
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


def _avg_logprob_per_answer_token(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    answers: Sequence[str],
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> List[float]:
    if len(prompts) != len(answers):
        raise ValueError("prompts/answers length mismatch.")

    was_training = bool(model.training)
    model.eval()

    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = list(prompts[start : start + batch_size])
            batch_answers = list(answers[start : start + batch_size])
            batch_texts = [p + a for p, a in zip(batch_prompts, batch_answers)]

            original_padding_side: Optional[str] = getattr(tokenizer, "padding_side", None)
            if original_padding_side is not None:
                tokenizer.padding_side = "right"
            try:
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(max_seq_length),
                    return_tensors="pt",
                    add_special_tokens=False,
                )
            finally:
                if original_padding_side is not None:
                    tokenizer.padding_side = original_padding_side

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            labels = input_ids.clone()
            prompt_lens: List[int] = []
            for p in batch_prompts:
                p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]  # type: ignore[index]
                prompt_lens.append(int(len(p_ids)))
            for i, prompt_len in enumerate(prompt_lens):
                prompt_len = min(prompt_len, int(labels.size(1)))
                labels[i, :prompt_len] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            gather_labels = shift_labels.clone()
            gather_labels[gather_labels < 0] = 0
            token_logp = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)

            valid = (shift_labels != -100) & (shift_mask != 0)
            for i in range(token_logp.size(0)):
                denom = int(valid[i].sum().item())
                if denom <= 0:
                    scores.append(float("nan"))
                    continue
                scores.append(float(token_logp[i][valid[i]].sum().item() / denom))

    if was_training:
        model.train()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAQUEL value-preference evaluation.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--max_denotation_rows",
        type=int,
        default=0,
        help="If > 0, skip affected examples whose aligned/null row counts exceed this.",
    )
    parser.add_argument(
        "--restrict_to_removed_entities",
        action="store_true",
        help="If set, restrict BOTH aligned-only and null-only candidate values to removed_entities.json (very strict).",
    )
    parser.add_argument(
        "--restrict_aligned_to_removed_entities",
        action="store_true",
        help="If set, restrict ONLY aligned-only values to removed_entities.json (high precision leakage probe).",
    )
    parser.add_argument(
        "--removed_entities",
        default="data/aligned_db/log/nullify/removed_entities.json",
        help="Removed entities log (used when --restrict_to_removed_entities).",
    )
    parser.add_argument("--values_per_side", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument(
        "--output_path",
        help="Output path (default: model_path/raquel_value_preference_eval.json).",
    )
    args = parser.parse_args()

    rows = _load_query_index_rows(args.query_index_map, split="affected")
    den = _read_jsonl_denotations(args.denotations)

    removed_values: Optional[Set[str]] = None
    if bool(args.restrict_to_removed_entities):
        removed_values = _load_removed_entity_values(str(args.removed_entities))
        logger.info("Loaded %d removed entity values", len(removed_values))
    if bool(args.restrict_aligned_to_removed_entities) and removed_values is None:
        removed_values = _load_removed_entity_values(str(args.removed_entities))
        logger.info("Loaded %d removed entity values (aligned-only restriction)", len(removed_values))

    max_rows = int(args.max_denotation_rows)
    candidates: List[Tuple[QueryIndexRow, Set[str], Set[str]]] = []
    skipped_by_rows = 0
    skipped_by_missing_side = 0

    for r in rows:
        rec = den.get(r.query_index)
        if rec is None:
            continue
        a_rows = rec.get("result_aligned") or []
        n_rows = rec.get("result_null") or []
        if not isinstance(a_rows, list) or not isinstance(n_rows, list):
            continue
        if max_rows > 0 and (len(a_rows) > max_rows or len(n_rows) > max_rows):
            skipped_by_rows += 1
            continue
        a_map = _extract_string_values(a_rows)
        n_map = _extract_string_values(n_rows)
        a_norms = set(a_map.keys())
        n_norms = set(n_map.keys())
        a_only_norms = a_norms - n_norms
        n_only_norms = n_norms - a_norms
        if removed_values is not None:
            if bool(args.restrict_to_removed_entities) or bool(args.restrict_aligned_to_removed_entities):
                a_only_norms = set(v for v in a_only_norms if v in removed_values)
            if bool(args.restrict_to_removed_entities):
                n_only_norms = set(v for v in n_only_norms if v in removed_values)

        if not a_only_norms or not n_only_norms:
            skipped_by_missing_side += 1
            continue
        a_only_surface = set(a_map[n] for n in a_only_norms if n in a_map)
        n_only_surface = set(n_map[n] for n in n_only_norms if n in n_map)
        if not a_only_surface or not n_only_surface:
            skipped_by_missing_side += 1
            continue
        candidates.append((r, a_only_surface, n_only_surface))

    if bool(args.shuffle):
        rng = random.Random(int(args.sample_seed))
        rng.shuffle(candidates)

    if int(args.max_examples) > 0:
        candidates = candidates[: int(args.max_examples)]

    # Build scoring items.
    prompts: List[str] = []
    answers: List[str] = []
    pair_slices: List[Tuple[int, int]] = []  # (start_idx, end_idx) in flat arrays per query
    query_indices: List[int] = []

    k = max(1, int(args.values_per_side))
    for r, a_only, n_only in candidates:
        query_indices.append(int(r.query_index))
        prompt = _build_prompt(r.question)
        a_list = sorted(a_only)[:k]
        n_list = sorted(n_only)[:k]
        start = len(prompts)
        for v in a_list:
            prompts.append(prompt)
            answers.append(v)
        for v in n_list:
            prompts.append(prompt)
            answers.append(v)
        end = len(prompts)
        pair_slices.append((start, end))

    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        base_model_name=args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    device = _resolve_device(model, args.device)

    flat_scores = _avg_logprob_per_answer_token(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        answers=answers,
        batch_size=int(args.batch_size),
        max_seq_length=int(args.max_seq_length),
        device=device,
    )

    margins: List[float] = []
    invalid = 0
    for (start, end) in pair_slices:
        # first k are aligned-only values; next k are null-only values
        a_scores = [s for s in flat_scores[start : start + k] if math.isfinite(s)]
        n_scores = [s for s in flat_scores[start + k : end] if math.isfinite(s)]
        if not a_scores or not n_scores:
            invalid += 1
            continue
        margins.append(float(sum(a_scores) / len(a_scores) - sum(n_scores) / len(n_scores)))

    prefer_aligned = sum(1 for m in margins if m > 0.0)
    prefer_null = sum(1 for m in margins if m < 0.0)
    ties = len(margins) - prefer_aligned - prefer_null

    out: Dict[str, Any] = {
        "affected": {
            "count": len(candidates),
            "eligible_total": len(rows),
            "skipped_by_max_rows": int(skipped_by_rows),
            "skipped_by_missing_side": int(skipped_by_missing_side),
            "invalid_pairs": int(invalid),
            "preference": {
                "prefer_aligned_rate": float(prefer_aligned / len(margins)) if margins else 0.0,
                "prefer_null_rate": float(prefer_null / len(margins)) if margins else 0.0,
                "tie_rate": float(ties / len(margins)) if margins else 0.0,
                "mean_margin": float(sum(margins) / len(margins)) if margins else 0.0,
            },
        },
        "metadata": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "device": str(device),
            "query_index_map": args.query_index_map,
            "denotations": args.denotations,
            "max_examples": int(args.max_examples),
            "shuffle": bool(args.shuffle),
            "sample_seed": int(args.sample_seed),
            "max_denotation_rows": int(args.max_denotation_rows),
            "values_per_side": int(args.values_per_side),
            "batch_size": int(args.batch_size),
            "max_seq_length": int(args.max_seq_length),
            "restrict_to_removed_entities": bool(args.restrict_to_removed_entities),
            "restrict_aligned_to_removed_entities": bool(args.restrict_aligned_to_removed_entities),
            "removed_entities": str(args.removed_entities),
        },
    }

    out_path = (
        str(args.output_path)
        if args.output_path
        else os.path.join(str(args.model_path), "raquel_value_preference_eval.json")
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

