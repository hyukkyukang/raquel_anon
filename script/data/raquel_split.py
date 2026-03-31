"""Split RAQUEL synthesized QA data into train/val/test with holdout strategies."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_examples(path: str) -> List[Dict[str, Any]]:
    """Load JSON examples while preserving metadata fields."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of examples in {path}")
    examples: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "question" not in item or "answer" not in item:
            continue
        examples.append(item)
    return examples


def _get_meta_value(example: Dict[str, Any], key: str) -> Optional[str]:
    """Retrieve a metadata field from top-level or nested metadata dict."""
    value = example.get(key)
    if value is not None:
        return str(value)
    meta = example.get("metadata") or example.get("meta")
    if isinstance(meta, dict) and key in meta:
        return str(meta[key])
    return None


def _group_key(example: Dict[str, Any], strategy: str) -> str:
    if strategy == "template_type":
        return (
            _get_meta_value(example, "query_type")
            or _get_meta_value(example, "template_type")
            or "unknown"
        )
    if strategy == "template_id":
        return (
            _get_meta_value(example, "template_id")
            or _get_meta_value(example, "query_index")
            or "unknown"
        )
    return "random"


def _load_query_index_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")

    by_split: Dict[str, List[Dict[str, Any]]] = {"affected": [], "unaffected": []}
    for row in payload:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip()
        if split in by_split:
            by_split[split].append(row)

    for split, rows in by_split.items():
        by_split[split] = sorted(rows, key=lambda r: int(r.get("example_index", 0)))
    return by_split


def _load_query_type_tags(path: str) -> Dict[int, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")

    out: Dict[int, List[str]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        query_index = row.get("query_index")
        tags = row.get("tags")
        if not isinstance(query_index, int) or not isinstance(tags, list):
            continue
        out[int(query_index)] = [str(tag) for tag in tags if str(tag).strip()]
    return out


def _load_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            query_index = row.get("query_index")
            if isinstance(query_index, int):
                out[int(query_index)] = row
    return out


def _merge_metadata(
    example: Dict[str, Any],
    *,
    split_name: str,
    query_index: int,
    query_tags: List[str],
) -> Dict[str, Any]:
    enriched = dict(example)
    meta = enriched.get("metadata") or enriched.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta["split"] = split_name
    meta["query_index"] = query_index
    meta["query_tags"] = list(query_tags)
    meta["query_type"] = query_tags[0] if query_tags else "other"
    enriched["metadata"] = meta
    return enriched


def _enrich_examples_with_query_metadata(
    examples: List[Dict[str, Any]],
    *,
    split_name: str,
    query_index_rows: List[Dict[str, Any]],
    tags_by_index: Dict[int, List[str]],
) -> List[Dict[str, Any]]:
    if not query_index_rows:
        return examples

    # Fast path: ordered alignment via example_index ordering.
    if len(examples) == len(query_index_rows):
        ordered_match = True
        for example, row in zip(examples, query_index_rows):
            if example.get("question") != row.get("question"):
                ordered_match = False
                break
        if ordered_match:
            enriched: List[Dict[str, Any]] = []
            for example, row in zip(examples, query_index_rows):
                query_index = int(row["query_index"])
                enriched.append(
                    _merge_metadata(
                        example,
                        split_name=split_name,
                        query_index=query_index,
                        query_tags=tags_by_index.get(query_index, ["other"]),
                    )
                )
            return enriched

    # Fallback: match by (question, answer), then by question only.
    qa_to_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    q_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in query_index_rows:
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        qa_to_rows.setdefault((question, answer), []).append(row)
        q_to_rows.setdefault(question, []).append(row)

    enriched = []
    for example in examples:
        question = str(example.get("question", ""))
        answer = str(example.get("answer", ""))
        row = None
        qa_key = (question, answer)
        if qa_to_rows.get(qa_key):
            row = qa_to_rows[qa_key].pop(0)
        elif q_to_rows.get(question):
            row = q_to_rows[question].pop(0)

        if row is None:
            enriched.append(dict(example))
            continue

        query_index = int(row["query_index"])
        enriched.append(
            _merge_metadata(
                example,
                split_name=split_name,
                query_index=query_index,
                query_tags=tags_by_index.get(query_index, ["other"]),
            )
        )
    return enriched


_ZEROISH_ANSWER_RE = re.compile(
    r"\b(?:zero|none|no\b|not provided|unknown|nil|0)\b", re.IGNORECASE
)
_POSSESSIVE_PLACEHOLDER_RE = re.compile(
    r"(?:'s (?:books|works|novels))$", re.IGNORECASE
)
_GENERIC_PLACEHOLDER_VALUES = {
    "archive",
    "archives",
    "author",
    "authors",
    "award",
    "awards",
    "book",
    "books",
    "genre",
    "genres",
    "libraries",
    "library",
    "novel",
    "novels",
    "organization",
    "organizations",
    "people",
    "person",
    "series",
    "work",
    "works",
}


def _is_placeholder_value(value: str) -> bool:
    surface = value.strip()
    if not surface:
        return False
    lowered = surface.lower().replace("_", " ").strip()
    if "_" in surface:
        return True
    if lowered in _GENERIC_PLACEHOLDER_VALUES:
        return True
    if _POSSESSIVE_PLACEHOLDER_RE.search(lowered):
        return True
    return False


def _placeholder_ratio(rows: Any) -> float:
    string_values: List[str] = []
    if not isinstance(rows, list):
        return 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, str):
                string_values.append(value)
    if not string_values:
        return 0.0
    placeholder_count = sum(_is_placeholder_value(value) for value in string_values)
    return float(placeholder_count / len(string_values))


def _numeric_denotation_stats(rows: Any) -> Optional[Dict[str, float]]:
    numeric_values: List[float] = []
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(float(value))
    if not numeric_values:
        return None
    return {
        "max": float(max(numeric_values)),
        "zero_ratio": float(
            sum(value == 0.0 for value in numeric_values) / len(numeric_values)
        ),
    }


def _filter_signal_pathologies(
    examples: List[Dict[str, Any]],
    *,
    denotations: Dict[int, Dict[str, Any]],
    split_name: str,
    placeholder_ratio_threshold: float,
    placeholder_ratio_low_threshold: float,
    max_numeric_value: float,
    zero_ratio_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    removed_by_reason: Counter[str] = Counter()
    removed_by_group: Counter[str] = Counter()

    for example in examples:
        metadata = example.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        query_index = metadata.get("query_index")
        denotation = (
            denotations.get(int(query_index))
            if isinstance(query_index, int)
            else None
        )
        if denotation is None:
            kept.append(example)
            continue

        aligned_rows = denotation.get("result_aligned") or []
        null_rows = denotation.get("result_null") or []
        placeholder_ratio = max(
            _placeholder_ratio(aligned_rows),
            _placeholder_ratio(null_rows),
        )

        aligned_numeric = _numeric_denotation_stats(aligned_rows)
        null_numeric = _numeric_denotation_stats(null_rows)
        has_numeric = aligned_numeric is not None or null_numeric is not None
        max_numeric = max(
            (aligned_numeric or {"max": -1.0})["max"],
            (null_numeric or {"max": -1.0})["max"],
        )
        zero_ratio = max(
            (aligned_numeric or {"zero_ratio": 0.0})["zero_ratio"],
            (null_numeric or {"zero_ratio": 0.0})["zero_ratio"],
        )
        zeroish_answer = bool(
            _ZEROISH_ANSWER_RE.search(str(example.get("answer", "")).strip())
        )

        reasons: List[str] = []
        if zeroish_answer and placeholder_ratio >= placeholder_ratio_threshold:
            reasons.append("zeroish_answer_and_placeholder")
        if has_numeric and placeholder_ratio >= placeholder_ratio_low_threshold:
            if max_numeric <= max_numeric_value:
                reasons.append("low_numeric_max_and_placeholder")
            if zero_ratio >= zero_ratio_threshold:
                reasons.append("zero_heavy_numeric_and_placeholder")

        if not reasons:
            kept.append(example)
            continue

        removed_example = dict(example)
        removed_metadata = dict(metadata)
        removed_metadata["filter_reasons"] = reasons
        removed_metadata["filter_signal_stats"] = {
            "placeholder_ratio": placeholder_ratio,
            "max_numeric_value": max_numeric,
            "zero_ratio": zero_ratio,
            "zeroish_answer": zeroish_answer,
        }
        removed_example["metadata"] = removed_metadata
        removed.append(removed_example)
        for reason in reasons:
            removed_by_reason[reason] += 1
        removed_by_group[_group_key(removed_example, "template_type")] += 1

    summary = {
        "split": split_name,
        "input_count": len(examples),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed_by_reason": dict(removed_by_reason),
        "removed_by_group": dict(removed_by_group),
    }
    return kept, removed, summary


def _split_random(
    examples: List[Dict[str, Any]],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_size = int(math.floor(total * test_ratio))
    val_size = int(math.floor(total * val_ratio))
    test_set = shuffled[:test_size]
    val_set = shuffled[test_size : test_size + val_size]
    train_set = shuffled[test_size + val_size :]
    return train_set, val_set, test_set


def _split_by_group(
    examples: List[Dict[str, Any]],
    *,
    strategy: str,
    holdout_ratio: float,
    val_ratio: float,
    rng: random.Random,
    holdout_groups: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        key = _group_key(ex, strategy)
        group_map.setdefault(key, []).append(ex)

    groups = list(group_map.keys())
    if holdout_groups:
        holdout_group_set = {group for group in holdout_groups if group in group_map}
        if not holdout_group_set:
            raise ValueError(
                "None of the requested holdout_groups were found in the examples."
            )
    else:
        rng.shuffle(groups)
        holdout_count = max(1, int(math.floor(len(groups) * holdout_ratio)))
        holdout_group_set = set(groups[:holdout_count])

    test_set = [ex for ex in examples if _group_key(ex, strategy) in holdout_group_set]
    remaining = [
        ex for ex in examples if _group_key(ex, strategy) not in holdout_group_set
    ]

    # Split remaining into train/val
    rng.shuffle(remaining)
    val_size = int(math.floor(len(remaining) * val_ratio))
    val_set = remaining[:val_size]
    train_set = remaining[val_size:]
    return train_set, val_set, test_set


def _apply_ratio(
    examples: List[Dict[str, Any]],
    *,
    ratio: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if ratio <= 0:
        return []
    if ratio == 1.0:
        return list(examples)
    target = int(math.floor(len(examples) * ratio))
    if ratio < 1.0:
        return rng.sample(examples, min(target, len(examples)))
    # Upsample with replacement
    return [rng.choice(examples) for _ in range(target)]


def _write_json(path: Path, examples: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(examples), f, indent=2)


def _write_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _summarize_split(
    name: str,
    train_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    strategy: str,
) -> Dict[str, Any]:
    def count_groups(data: List[Dict[str, Any]]) -> Dict[str, int]:
        return dict(Counter(_group_key(ex, strategy) for ex in data))

    return {
        "name": name,
        "counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "group_distribution": {
            "train": count_groups(train_set),
            "val": count_groups(val_set),
            "test": count_groups(test_set),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split RAQUEL QA data.")
    parser.add_argument("--affected", required=True, help="Path to affected JSON file")
    parser.add_argument("--unaffected", required=True, help="Path to unaffected JSON file")
    parser.add_argument("--out_dir", required=True, help="Output directory for splits")
    parser.add_argument(
        "--strategy",
        choices=["template_type", "template_id", "random"],
        default="template_type",
        help="Holdout strategy",
    )
    parser.add_argument(
        "--holdout_ratio",
        type=float,
        default=0.2,
        help="Fraction of groups held out for test (or random test ratio)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of remaining data used for validation",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Scaling ratio applied to train splits only",
    )
    parser.add_argument(
        "--query_index_map",
        default="data/raquel/query_index_map.json",
        help="Optional RAQUEL query-index map used to recover query metadata.",
    )
    parser.add_argument(
        "--query_type_tags",
        default="data/raquel/query_type_tags_heuristic.json",
        help="Optional RAQUEL query-type tags used for template-type holdout.",
    )
    parser.add_argument(
        "--holdout_groups",
        default="",
        help="Comma-separated explicit holdout groups for grouped strategies.",
    )
    parser.add_argument(
        "--filter_signal_quality",
        choices=["none", "affected", "both"],
        default="none",
        help=(
            "Optionally remove zero-count / placeholder-heavy examples before "
            "splitting. Default keeps the original dataset unchanged."
        ),
    )
    parser.add_argument(
        "--denotations",
        default="data/raquel/denotations/by_index.jsonl",
        help="Denotation JSONL used for signal-quality filtering.",
    )
    parser.add_argument(
        "--filter_placeholder_ratio_threshold",
        type=float,
        default=0.4,
        help="High placeholder ratio threshold used with zeroish textual answers.",
    )
    parser.add_argument(
        "--filter_placeholder_ratio_low_threshold",
        type=float,
        default=0.25,
        help="Lower placeholder ratio threshold used with numeric aggregate checks.",
    )
    parser.add_argument(
        "--filter_max_numeric_value",
        type=float,
        default=1.0,
        help="Maximum numeric aggregate value treated as low-signal for filtering.",
    )
    parser.add_argument(
        "--filter_zero_ratio_threshold",
        type=float,
        default=0.6,
        help="Numeric zero-ratio threshold used for low-signal filtering.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    affected = _load_examples(args.affected)
    unaffected = _load_examples(args.unaffected)

    if args.strategy != "random":
        by_split = _load_query_index_map(args.query_index_map)
        tags_by_index = _load_query_type_tags(args.query_type_tags)
        affected = _enrich_examples_with_query_metadata(
            affected,
            split_name="affected",
            query_index_rows=by_split.get("affected", []),
            tags_by_index=tags_by_index,
        )
        unaffected = _enrich_examples_with_query_metadata(
            unaffected,
            split_name="unaffected",
            query_index_rows=by_split.get("unaffected", []),
            tags_by_index=tags_by_index,
        )

    filter_summary: Dict[str, Any] = {"mode": args.filter_signal_quality}
    affected_removed: List[Dict[str, Any]] = []
    unaffected_removed: List[Dict[str, Any]] = []
    if args.filter_signal_quality != "none":
        denotations = _load_denotations(args.denotations)
        if args.filter_signal_quality in {"affected", "both"}:
            affected, affected_removed, affected_filter_summary = (
                _filter_signal_pathologies(
                    affected,
                    denotations=denotations,
                    split_name="affected",
                    placeholder_ratio_threshold=args.filter_placeholder_ratio_threshold,
                    placeholder_ratio_low_threshold=(
                        args.filter_placeholder_ratio_low_threshold
                    ),
                    max_numeric_value=args.filter_max_numeric_value,
                    zero_ratio_threshold=args.filter_zero_ratio_threshold,
                )
            )
            filter_summary["affected"] = affected_filter_summary
        if args.filter_signal_quality == "both":
            unaffected, unaffected_removed, unaffected_filter_summary = (
                _filter_signal_pathologies(
                    unaffected,
                    denotations=denotations,
                    split_name="unaffected",
                    placeholder_ratio_threshold=args.filter_placeholder_ratio_threshold,
                    placeholder_ratio_low_threshold=(
                        args.filter_placeholder_ratio_low_threshold
                    ),
                    max_numeric_value=args.filter_max_numeric_value,
                    zero_ratio_threshold=args.filter_zero_ratio_threshold,
                )
            )
            filter_summary["unaffected"] = unaffected_filter_summary

    requested_holdout_groups = [
        group.strip() for group in str(args.holdout_groups).split(",") if group.strip()
    ]

    if args.strategy == "random":
        split_fn = lambda data: _split_random(
            data, val_ratio=args.val_ratio, test_ratio=args.holdout_ratio, rng=rng
        )
    else:
        split_fn = lambda data: _split_by_group(
            data,
            strategy=args.strategy,
            holdout_ratio=args.holdout_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
            holdout_groups=requested_holdout_groups or None,
        )

    affected_train, affected_val, affected_test = split_fn(affected)
    unaffected_train, unaffected_val, unaffected_test = split_fn(unaffected)

    affected_train = _apply_ratio(affected_train, ratio=args.ratio, rng=rng)
    unaffected_train = _apply_ratio(unaffected_train, ratio=args.ratio, rng=rng)

    out_dir = Path(args.out_dir)
    _write_json(out_dir / "affected_train.json", affected_train)
    _write_json(out_dir / "affected_val.json", affected_val)
    _write_json(out_dir / "affected_test.json", affected_test)
    _write_json(out_dir / "unaffected_train.json", unaffected_train)
    _write_json(out_dir / "unaffected_val.json", unaffected_val)
    _write_json(out_dir / "unaffected_test.json", unaffected_test)
    if affected_removed:
        _write_json(out_dir / "affected_filtered_out.json", affected_removed)
    if unaffected_removed:
        _write_json(out_dir / "unaffected_filtered_out.json", unaffected_removed)

    summary = {
        "strategy": args.strategy,
        "holdout_ratio": args.holdout_ratio,
        "val_ratio": args.val_ratio,
        "train_ratio_scale": args.ratio,
        "holdout_groups": requested_holdout_groups,
        "filter_signal_quality": filter_summary,
        "affected": _summarize_split(
            "affected", affected_train, affected_val, affected_test, args.strategy
        ),
        "unaffected": _summarize_split(
            "unaffected", unaffected_train, unaffected_val, unaffected_test, args.strategy
        ),
    }
    _write_payload(out_dir / "split_summary.json", summary)


if __name__ == "__main__":
    main()
