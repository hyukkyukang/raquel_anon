"""Build local TOFU-style supervision files from regenerated aligned-DB QA pairs.

This script converts an aligned-db `qa_pairs.jsonl` artifact into standard
`{"question", "answer"}` JSON files and aligns it back to the original TOFU
retain/forget splits. The intended use is oracle retraining with regenerated QA
supervision while keeping the original full-vs-retain split semantics.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


Example = Dict[str, str]
PairKey = Tuple[str, str]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_regenerated_pairs(path: Path) -> List[Example]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")

    examples: List[Example] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                f"Regenerated row {idx} in {path} must be a two-item [question, answer] list."
            )
        question, answer = row
        examples.append({"question": str(question), "answer": str(answer)})
    return examples


def _load_examples(path: Path) -> List[Example]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")

    examples: List[Example] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Split row {idx} in {path} must be an object.")
        if "question" not in row or "answer" not in row:
            raise ValueError(f"Split row {idx} in {path} is missing question/answer.")
        examples.append({"question": str(row["question"]), "answer": str(row["answer"])})
    return examples


def _pair_key(example: Example) -> PairKey:
    return (_normalize(example["question"]), _normalize(example["answer"]))


def _score_question_similarity(left: str, right: str) -> float:
    return float(
        difflib.SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()
    )


def _select_fallback_match(
    target: Example,
    regenerated: Sequence[Example],
    available_indices: set[int],
) -> Tuple[int, str] | None:
    target_answer = _normalize(target["answer"])
    exact_answer_indices = [
        idx
        for idx in available_indices
        if _normalize(regenerated[idx]["answer"]) == target_answer
    ]
    if len(exact_answer_indices) == 1:
        return exact_answer_indices[0], "answer_only_unique"

    if len(exact_answer_indices) > 1:
        scored = sorted(
            (
                (
                    _score_question_similarity(
                        target["question"], regenerated[idx]["question"]
                    ),
                    idx,
                )
                for idx in exact_answer_indices
            ),
            reverse=True,
        )
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1], "answer_only_tiebreak"

    scored_all = sorted(
        (
            (
                _score_question_similarity(target["question"], regenerated[idx]["question"]),
                idx,
            )
            for idx in available_indices
        ),
        reverse=True,
    )
    if scored_all and scored_all[0][0] >= 0.98:
        return scored_all[0][1], "question_similarity"
    return None


def _align_subset(
    subset_name: str,
    subset_examples: Sequence[Example],
    regenerated: Sequence[Example],
) -> Tuple[List[Example], List[Dict[str, Any]]]:
    index_by_pair: Dict[PairKey, List[int]] = {}
    for idx, row in enumerate(regenerated):
        index_by_pair.setdefault(_pair_key(row), []).append(idx)

    available_indices = set(range(len(regenerated)))
    aligned: List[Example] = []
    audit: List[Dict[str, Any]] = []

    for target in subset_examples:
        exact_key = _pair_key(target)
        exact_candidates = [
            idx for idx in index_by_pair.get(exact_key, []) if idx in available_indices
        ]

        chosen_idx: int | None = None
        match_mode = "exact"
        if exact_candidates:
            chosen_idx = exact_candidates[0]
        else:
            fallback = _select_fallback_match(target, regenerated, available_indices)
            if fallback is not None:
                chosen_idx, match_mode = fallback

        if chosen_idx is None:
            audit.append(
                {
                    "subset": subset_name,
                    "match_mode": "unmatched",
                    "target_question": target["question"],
                    "target_answer": target["answer"],
                }
            )
            continue

        available_indices.remove(chosen_idx)
        chosen = regenerated[chosen_idx]
        aligned.append(chosen)
        audit.append(
            {
                "subset": subset_name,
                "match_mode": match_mode,
                "target_question": target["question"],
                "target_answer": target["answer"],
                "selected_question": chosen["question"],
                "selected_answer": chosen["answer"],
            }
        )

    return aligned, audit


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local regenerated TOFU-style supervision files."
    )
    parser.add_argument(
        "--regenerated_pairs",
        default="data/aligned_db_full_gpt54_20260326_tuned/qa_pairs.jsonl",
        help="Aligned-db QA pairs artifact encoded as a JSON list of [question, answer].",
    )
    parser.add_argument(
        "--retain_file",
        default="data/tofu/retain90.json",
        help="Original TOFU retain split used as the alignment target.",
    )
    parser.add_argument(
        "--forget_file",
        default="data/tofu/forget10.json",
        help="Original TOFU forget split used as the alignment target.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/tofu_regenerated/gpt54_tuned_20260326",
        help="Output directory for regenerated supervision files.",
    )
    args = parser.parse_args()

    regenerated = _load_regenerated_pairs(Path(args.regenerated_pairs))
    retain = _load_examples(Path(args.retain_file))
    forget = _load_examples(Path(args.forget_file))

    full_path = Path(args.out_dir) / "full.json"
    retain_path = Path(args.out_dir) / "retain90.json"
    forget_path = Path(args.out_dir) / "forget10.json"
    audit_path = Path(args.out_dir) / "alignment_audit.json"
    summary_path = Path(args.out_dir) / "alignment_summary.json"

    retain_aligned, retain_audit = _align_subset("retain90", retain, regenerated)
    forget_aligned, forget_audit = _align_subset("forget10", forget, regenerated)

    combined_audit = retain_audit + forget_audit
    exact_counts = Counter(item["match_mode"] for item in combined_audit)
    unmatched = [item for item in combined_audit if item["match_mode"] == "unmatched"]

    summary = {
        "regenerated_pairs": args.regenerated_pairs,
        "retain_file": args.retain_file,
        "forget_file": args.forget_file,
        "counts": {
            "full": len(regenerated),
            "retain90": len(retain_aligned),
            "forget10": len(forget_aligned),
        },
        "target_counts": {
            "retain90": len(retain),
            "forget10": len(forget),
        },
        "match_mode_counts": dict(exact_counts),
        "unmatched_count": len(unmatched),
    }

    _write_json(full_path, regenerated)
    _write_json(retain_path, retain_aligned)
    _write_json(forget_path, forget_aligned)
    _write_json(audit_path, combined_audit)
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if unmatched:
        raise SystemExit(
            f"Alignment left {len(unmatched)} unmatched examples. See {audit_path}."
        )


if __name__ == "__main__":
    main()
