"""Pairwise denotation-preference evaluation for RAQUEL (aligned vs null).

Why this evaluation?
--------------------
Affected RAQUEL queries have *two* executable denotations:
  - aligned (pre-forget) denotation: E(q, D)
  - nullified (post-forget) denotation: E(q, D_null)

Free-form generation + ROUGE can be dominated by scaffolding and may not separate
`M_orig` from `M_ret`. This script avoids that failure mode by using *forced
scoring* (teacher-forced log-likelihood) of two candidate denotation strings.

For each affected query q, we:
  1) Serialize result_aligned and result_null into canonical JSON strings.
  2) Compute the model's average log-probability per answer token for each
     candidate string given the same prompt "Question: ...\\nAnswer: ".
  3) Report preference statistics, e.g., P(score_aligned > score_null).

This is fully offline: no external APIs, no judges.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from script.evaluation.utils import load_fine_tuned_model
from src.training.data.transforms import ANSWER_PREFIX
from src.utils.logging import get_logger

logger = get_logger("script.evaluation.run_raquel_denotation_preference_eval", __file__)


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
    rows = sorted(rows, key=lambda r: r.example_index)
    return rows


def _canonicalize_denotation(rows: Any) -> str:
    """Canonical JSON for a denotation result (row ordering + key ordering)."""
    if not isinstance(rows, list):
        rows = []
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append({str(k): v for k, v in row.items()})
    # Sort rows by their JSON signature for determinism.
    normalized_rows = sorted(
        normalized_rows,
        key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    )
    return json.dumps(normalized_rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _build_prompts(questions: Sequence[str]) -> List[str]:
    return [f"Question: {q.strip()}{ANSWER_PREFIX}" for q in questions]


def _compute_avg_logprob_per_answer_token(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    answers: Sequence[str],
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> List[float]:
    """Compute avg log P(answer | prompt) per answer token.

    We teacher-force on the concatenated string and mask prompt tokens out.
    """
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

            # Right padding keeps prompt prefix aligned at position 0.
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

            # Build labels and mask prompt tokens.
            labels = input_ids.clone()
            prompt_lens: List[int] = []
            for p in batch_prompts:
                p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]  # type: ignore[index]
                prompt_lens.append(int(len(p_ids)))

            for i, prompt_len in enumerate(prompt_lens):
                # Truncation may shorten; guard bounds.
                prompt_len = min(prompt_len, int(labels.size(1)))
                labels[i, :prompt_len] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Shift for causal LM loss: token t is predicted from logits at t-1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            # Gather log-prob at gold tokens where label != -100.
            gather_labels = shift_labels.clone()
            gather_labels[gather_labels < 0] = 0
            token_logp = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)

            valid = (shift_labels != -100) & (shift_mask != 0)
            for i in range(token_logp.size(0)):
                valid_i = valid[i]
                denom = int(valid_i.sum().item())
                if denom <= 0:
                    scores.append(float("-inf"))
                    continue
                score = float(token_logp[i][valid_i].sum().item() / denom)
                scores.append(score)

    if was_training:
        model.train()
    return scores


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run denotation-preference RAQUEL evaluation (aligned vs null)."
    )
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
        "--require_nonempty_null",
        action="store_true",
        help="If set, keep only affected queries with a non-empty null denotation (reduces empty-list prior effects).",
    )
    parser.add_argument(
        "--require_nonempty_aligned",
        action="store_true",
        help="If set, keep only affected queries with a non-empty aligned denotation.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max token length for prompt+answer scoring (truncates).",
    )
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument(
        "--output_path",
        help="Output path (default: model_path/raquel_denotation_preference_eval.json).",
    )
    args = parser.parse_args()

    index_rows = _load_query_index_rows(args.query_index_map, split="affected")
    denotations = _read_jsonl_denotations(args.denotations)

    # Filter by denotation size (rows) to keep answers reasonably short/stable.
    kept: List[QueryIndexRow] = []
    skipped_by_rows = 0
    skipped_by_empty_null = 0
    skipped_by_empty_aligned = 0
    max_rows = int(args.max_denotation_rows)
    for r in index_rows:
        rec = denotations.get(r.query_index)
        if rec is None:
            continue
        a_rows = rec.get("result_aligned") or []
        n_rows = rec.get("result_null") or []
        if not isinstance(a_rows, list) or not isinstance(n_rows, list):
            continue
        if max_rows > 0 and (len(a_rows) > max_rows or len(n_rows) > max_rows):
            skipped_by_rows += 1
            continue
        if bool(args.require_nonempty_aligned) and len(a_rows) == 0:
            skipped_by_empty_aligned += 1
            continue
        if bool(args.require_nonempty_null) and len(n_rows) == 0:
            skipped_by_empty_null += 1
            continue
        kept.append(r)

    if bool(args.shuffle):
        rng = random.Random(int(args.sample_seed))
        rng.shuffle(kept)

    if int(args.max_examples) > 0:
        kept = kept[: int(args.max_examples)]

    questions: List[str] = [r.question for r in kept]
    aligned_answers: List[str] = []
    null_answers: List[str] = []
    query_indices: List[int] = []
    for r in kept:
        rec = denotations.get(r.query_index)
        assert rec is not None
        aligned_answers.append(_canonicalize_denotation(rec.get("result_aligned")))
        null_answers.append(_canonicalize_denotation(rec.get("result_null")))
        query_indices.append(int(r.query_index))

    model, tokenizer = load_fine_tuned_model(
        args.model_path,
        base_model_name=args.base_model,
        device_map_auto=True,
        quantize_4bit=False,
        as_trainable=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    device = _resolve_device(model, args.device)

    prompts = _build_prompts(questions)
    score_aligned = _compute_avg_logprob_per_answer_token(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        answers=aligned_answers,
        batch_size=int(args.batch_size),
        max_seq_length=int(args.max_seq_length),
        device=device,
    )
    score_null = _compute_avg_logprob_per_answer_token(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        answers=null_answers,
        batch_size=int(args.batch_size),
        max_seq_length=int(args.max_seq_length),
        device=device,
    )

    if len(score_aligned) != len(score_null):
        raise RuntimeError("Score length mismatch.")

    valid_pairs = [
        (a, n) for a, n in zip(score_aligned, score_null) if math.isfinite(a) and math.isfinite(n)
    ]
    invalid = len(score_aligned) - len(valid_pairs)
    margins: List[float] = [float(a - n) for a, n in valid_pairs]
    prefer_aligned = sum(1 for m in margins if m > 0.0)
    prefer_null = sum(1 for m in margins if m < 0.0)
    ties = len(margins) - prefer_aligned - prefer_null

    payload: Dict[str, Any] = {
        "affected": {
            "count": len(kept),
            "eligible_total": len(index_rows),
            "skipped_by_max_rows": int(skipped_by_rows),
            "skipped_by_empty_aligned": int(skipped_by_empty_aligned),
            "skipped_by_empty_null": int(skipped_by_empty_null),
            "invalid_scoring_pairs": int(invalid),
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
            "batch_size": int(args.batch_size),
            "max_seq_length": int(args.max_seq_length),
        },
    }

    out_path = (
        str(args.output_path)
        if args.output_path
        else os.path.join(str(args.model_path), "raquel_denotation_preference_eval.json")
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

