"""Export a deterministic review pool for the capability-controlled affected subset.

This script builds a stratified sample from the RAQUEL affected split and exports
paper-reusable review artifacts:

  - JSON with full review context
  - CSV review sheet for manual annotation
  - JSONL annotation template
  - JSON summary with pool/sample composition

The subset-definition logic is intentionally independent of model win/loss
signals. Model outputs can be included for context, but the final solvability
decision should be based on question/denotation faithfulness and interpretability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_EVAL_SPECS = (
    "Morig=model/full_1b_s0/finetune/meta-llama/Llama-3.2-1B/raquel_eval_with_predictions.json",
    "Mret=model/retain_1b_s0/finetune/meta-llama/Llama-3.2-1B/raquel_eval_with_predictions.json",
    "GA+GD=model/unlearn_ga_gd_1b_s0/meta-llama/Llama-3.2-1B/raquel_eval_with_predictions.json",
    "NPO+GD=model/unlearn_npo_gd_1b_s0/meta-llama/Llama-3.2-1B/raquel_eval_with_predictions.json",
)

PRIMARY_FAMILY_PRIORITY = (
    "having",
    "groupby",
    "multi_join",
    "join",
    "orderby",
    "distinct",
    "between",
    "other",
)

DEGENERATE_PATTERNS = (
    "no results found",
    "no data available",
    "not available in the data",
    "cannot determine",
    "can't determine",
    "no results",
    "not provided",
)


@dataclass(frozen=True)
class PoolRow:
    query_index: int
    example_index: int
    question: str
    aligned_answer_text: str
    primary_family: str
    tags: Tuple[str, ...]
    sql: str
    result_aligned: Sequence[Mapping[str, Any]]
    result_null: Sequence[Mapping[str, Any]]


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


def _parse_eval_specs(values: Sequence[str]) -> List[Tuple[str, str]]:
    specs = list(values) if values else list(DEFAULT_EVAL_SPECS)
    parsed: List[Tuple[str, str]] = []
    for raw in specs:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --eval spec {raw!r}; expected LABEL=PATH."
            )
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(
                f"Invalid --eval spec {raw!r}; expected LABEL=PATH."
            )
        parsed.append((label, path))
    return parsed


def _load_query_index_rows(path: str, split: str) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    rows = [
        row
        for row in payload
        if isinstance(row, dict) and str(row.get("split", "")).strip() == split
    ]
    rows = sorted(rows, key=lambda row: int(row.get("example_index", 0)))
    return rows


def _load_tags(path: str) -> Dict[int, Tuple[str, ...]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[int, Tuple[str, ...]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        query_index = row.get("query_index")
        tags = row.get("tags")
        if not isinstance(query_index, int) or not isinstance(tags, list):
            continue
        out[int(query_index)] = tuple(str(tag).strip() for tag in tags if str(tag).strip())
    return out


def _load_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    rows = _read_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        query_index = row.get("query_index")
        if not isinstance(query_index, int):
            continue
        out[int(query_index)] = row
    return out


def _primary_family(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    for family in PRIMARY_FAMILY_PRIORITY:
        if family in tag_set:
            return family
    return "other"


def _load_eval_predictions(path: str, split: str) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split {split!r} in {path}")
    preds = split_payload.get("predictions")
    if not isinstance(preds, list):
        raise ValueError(
            f"Expected predictions list under split {split!r} in {path}; "
            "run evaluation with prediction saving enabled."
        )
    return [row for row in preds if isinstance(row, dict)]


def _map_eval_predictions_to_rows(
    *,
    eval_predictions: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
) -> Dict[int, str]:
    """Map prediction rows to query indices using stable file order first."""
    qmap_rows = list(index_rows)
    by_question: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in qmap_rows:
        question = str(row.get("question", "")).strip()
        if question:
            by_question[question].append(row)

    out: Dict[int, str] = {}
    n = len(eval_predictions)
    if n > len(qmap_rows):
        raise ValueError(
            f"Eval predictions length {n} exceeds query-map rows length {len(qmap_rows)}."
        )

    for idx, pred_row in enumerate(eval_predictions):
        pred_question = str(pred_row.get("question", "")).strip()
        pred_text = str(pred_row.get("prediction", "")).strip()

        map_row: Optional[Mapping[str, Any]] = None
        ordered_row = qmap_rows[idx]
        ordered_question = str(ordered_row.get("question", "")).strip()
        if pred_question and ordered_question and pred_question == ordered_question:
            map_row = ordered_row
        elif pred_question and by_question.get(pred_question):
            map_row = by_question[pred_question].pop(0)
        else:
            map_row = ordered_row

        query_index = map_row.get("query_index")
        if not isinstance(query_index, int):
            raise ValueError(f"Missing integer query_index at eval row {idx}.")
        out[int(query_index)] = pred_text
    return out


def _normalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def _extract_surface_values(rows: Any) -> Dict[str, str]:
    """Flatten denotation rows into a normalized->surface mapping."""
    out: Dict[str, str] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict):
            values = row.values()
        elif isinstance(row, list):
            values = row
        else:
            values = [row]
        for value in values:
            surface = _normalize_value(value)
            if not surface:
                continue
            norm = surface.casefold()
            out.setdefault(norm, surface)
    return out


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _delimiter_count(text: str) -> int:
    return text.count(",") + text.count(";") + text.count("\n")


def _is_degenerate_reference(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized in {"0", "0.", "0.0", "0%"}:
        return True
    return any(pattern in normalized for pattern in DEGENERATE_PATTERNS)


def _contains_placeholder_like(text: str) -> bool:
    normalized = text.strip().lower()
    if "_" in text:
        return True
    for phrase in (
        "unspecified",
        "unknown",
        "untitled",
        "placeholder",
        "not available",
        "not provided",
        "cannot determine",
    ):
        if phrase in normalized:
            return True
    return bool(re.search(r"\b[a-z]+_[a-z0-9_]+\b", text))


def _is_zero_heavy(text: str) -> bool:
    lowered = text.lower()
    zero_words = lowered.count("zero")
    zero_digits = len(re.findall(r"\b0\b", lowered))
    return zero_words >= 2 or zero_digits >= 3


def _auto_flags(
    *,
    question: str,
    aligned_answer_text: str,
    aligned_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    max_rows = max(len(aligned_rows), len(null_rows))
    return {
        "degenerate_reference": _is_degenerate_reference(aligned_answer_text),
        "placeholder_like": _contains_placeholder_like(question) or _contains_placeholder_like(aligned_answer_text),
        "list_heavy": len(aligned_answer_text) > 350 or _delimiter_count(aligned_answer_text) >= 6 or max_rows >= 6,
        "zero_heavy": _is_zero_heavy(aligned_answer_text),
        "long_reference": len(aligned_answer_text) > 500,
        "null_empty": len(null_rows) == 0,
        "aligned_empty": len(aligned_rows) == 0,
    }


def _allocate_sample_counts(
    family_counts: Mapping[str, int],
    sample_size: int,
    min_per_family: int,
) -> Dict[str, int]:
    families = sorted(family_counts)
    total = sum(family_counts.values())
    if sample_size >= total:
        return {family: family_counts[family] for family in families}
    if sample_size <= 0:
        return {family: 0 for family in families}

    allocation: Dict[str, int] = {family: 0 for family in families}
    remaining = sample_size

    if sample_size < len(families):
        for family, _count in sorted(
            family_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:sample_size]:
            allocation[family] = 1
        return allocation

    for family in families:
        base = min(int(family_counts[family]), int(min_per_family))
        allocation[family] = base
        remaining -= base

    if remaining <= 0:
        while remaining < 0:
            for family, _count in sorted(
                family_counts.items(),
                key=lambda item: (-allocation[item[0]], item[0]),
            ):
                if remaining == 0:
                    break
                if allocation[family] > 0:
                    allocation[family] -= 1
                    remaining += 1
        return allocation

    caps = {
        family: int(family_counts[family]) - int(allocation[family])
        for family in families
    }
    cap_total = sum(max(cap, 0) for cap in caps.values())
    if cap_total <= 0:
        return allocation

    extras: Dict[str, int] = {family: 0 for family in families}
    fractions: List[Tuple[float, str]] = []
    allocated_extra = 0
    for family in families:
        cap = max(caps[family], 0)
        if cap == 0:
            continue
        raw = remaining * (cap / cap_total)
        whole = min(cap, int(math.floor(raw)))
        extras[family] = whole
        allocated_extra += whole
        fractions.append((raw - whole, family))

    remainder = remaining - allocated_extra
    for _fraction, family in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remainder <= 0:
            break
        if extras[family] < caps[family]:
            extras[family] += 1
            remainder -= 1

    if remainder > 0:
        for family, _count in sorted(
            family_counts.items(),
            key=lambda item: (-caps[item[0]], item[0]),
        ):
            while remainder > 0 and extras[family] < caps[family]:
                extras[family] += 1
                remainder -= 1
            if remainder <= 0:
                break

    for family in families:
        allocation[family] += extras[family]
    return allocation


def _stratified_sample(
    rows: Sequence[PoolRow],
    *,
    sample_size: int,
    min_per_family: int,
    sample_seed: int,
) -> Tuple[List[PoolRow], Dict[str, Any]]:
    family_groups: DefaultDict[str, List[PoolRow]] = defaultdict(list)
    for row in rows:
        family_groups[row.primary_family].append(row)

    rng = random.Random(int(sample_seed))
    family_counts = {family: len(items) for family, items in family_groups.items()}
    allocation = _allocate_sample_counts(
        family_counts=family_counts,
        sample_size=int(sample_size),
        min_per_family=int(min_per_family),
    )

    sampled: List[PoolRow] = []
    for family in sorted(family_groups):
        items = list(family_groups[family])
        rng.shuffle(items)
        keep = int(allocation.get(family, 0))
        sampled.extend(items[:keep])

    rng.shuffle(sampled)
    sample_counts = Counter(row.primary_family for row in sampled)
    summary = {
        "pool_counts": dict(sorted(family_counts.items())),
        "sample_counts": dict(sorted(sample_counts.items())),
        "allocation": dict(sorted(allocation.items())),
    }
    return sampled, summary


def _prepare_pool_rows(
    *,
    index_rows: Sequence[Mapping[str, Any]],
    tags_by_index: Mapping[int, Tuple[str, ...]],
    denotations_by_index: Mapping[int, Mapping[str, Any]],
) -> List[PoolRow]:
    pool: List[PoolRow] = []
    for row in index_rows:
        query_index = row.get("query_index")
        example_index = row.get("example_index")
        if not isinstance(query_index, int) or not isinstance(example_index, int):
            continue
        den = denotations_by_index.get(query_index)
        if not isinstance(den, Mapping):
            continue
        tags = tags_by_index.get(query_index, ("other",))
        aligned_rows = den.get("result_aligned") or []
        null_rows = den.get("result_null") or []
        if not isinstance(aligned_rows, list) or not isinstance(null_rows, list):
            continue
        pool.append(
            PoolRow(
                query_index=int(query_index),
                example_index=int(example_index),
                question=str(row.get("question", "")).strip(),
                aligned_answer_text=str(row.get("answer", "")).strip(),
                primary_family=_primary_family(tags),
                tags=tuple(tags),
                sql=str(den.get("sql", "")).strip(),
                result_aligned=aligned_rows,
                result_null=null_rows,
            )
        )
    return pool


def _build_review_rows(
    *,
    sampled_rows: Sequence[PoolRow],
    predictions_by_label: Mapping[str, Mapping[int, str]],
) -> List[Dict[str, Any]]:
    review_rows: List[Dict[str, Any]] = []
    for sample_order, row in enumerate(sampled_rows, start=1):
        aligned_map = _extract_surface_values(row.result_aligned)
        null_map = _extract_surface_values(row.result_null)
        aligned_only = [aligned_map[key] for key in sorted(set(aligned_map) - set(null_map))]
        null_only = [null_map[key] for key in sorted(set(null_map) - set(aligned_map))]
        auto_flags = _auto_flags(
            question=row.question,
            aligned_answer_text=row.aligned_answer_text,
            aligned_rows=row.result_aligned,
            null_rows=row.result_null,
        )

        review_rows.append(
            {
                "sample_order": sample_order,
                "query_index": row.query_index,
                "example_index": row.example_index,
                "primary_family": row.primary_family,
                "tags": list(row.tags),
                "question": row.question,
                "aligned_answer_text": row.aligned_answer_text,
                "sql": row.sql,
                "aligned_row_count": len(row.result_aligned),
                "null_row_count": len(row.result_null),
                "aligned_only_values": aligned_only,
                "null_only_values": null_only,
                "aligned_denotation": row.result_aligned,
                "null_denotation": row.result_null,
                "auto_flags": auto_flags,
                "predictions": {
                    label: str(predictions.get(row.query_index, "")).strip()
                    for label, predictions in predictions_by_label.items()
                },
                "annotation": {
                    "judge_sql_to_question": "",
                    "judge_aligned_answer_supported": "",
                    "judge_null_shift_interpretable": "",
                    "judge_not_pathological": "",
                    "final_decision": "",
                    "pathology_flags": "",
                    "notes": "",
                },
            }
        )
    return review_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return slug or "model"


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    prediction_labels: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_order",
        "query_index",
        "example_index",
        "primary_family",
        "tags",
        "question",
        "aligned_answer_text",
        "aligned_row_count",
        "null_row_count",
        "aligned_only_values",
        "null_only_values",
        "sql",
        "aligned_denotation",
        "null_denotation",
        "auto_degenerate_reference",
        "auto_placeholder_like",
        "auto_list_heavy",
        "auto_zero_heavy",
        "auto_long_reference",
        "auto_null_empty",
        "auto_aligned_empty",
    ]
    prediction_fieldnames: Dict[str, str] = {}
    for label in prediction_labels:
        fieldname = f"prediction_{_slugify_label(label)}"
        prediction_fieldnames[label] = fieldname
        fieldnames.append(fieldname)
    fieldnames.extend(
        [
            "judge_sql_to_question",
            "judge_aligned_answer_supported",
            "judge_null_shift_interpretable",
            "judge_not_pathological",
            "final_decision",
            "pathology_flags",
            "notes",
        ]
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            auto_flags = row.get("auto_flags", {})
            annotation = row.get("annotation", {})
            predictions = row.get("predictions", {})
            flat: Dict[str, Any] = {
                "sample_order": row.get("sample_order"),
                "query_index": row.get("query_index"),
                "example_index": row.get("example_index"),
                "primary_family": row.get("primary_family"),
                "tags": ";".join(row.get("tags", [])),
                "question": _single_line(str(row.get("question", ""))),
                "aligned_answer_text": _single_line(str(row.get("aligned_answer_text", ""))),
                "aligned_row_count": row.get("aligned_row_count", 0),
                "null_row_count": row.get("null_row_count", 0),
                "aligned_only_values": "; ".join(row.get("aligned_only_values", [])),
                "null_only_values": "; ".join(row.get("null_only_values", [])),
                "sql": _single_line(str(row.get("sql", ""))),
                "aligned_denotation": _json_compact(row.get("aligned_denotation", [])),
                "null_denotation": _json_compact(row.get("null_denotation", [])),
                "auto_degenerate_reference": auto_flags.get("degenerate_reference", False),
                "auto_placeholder_like": auto_flags.get("placeholder_like", False),
                "auto_list_heavy": auto_flags.get("list_heavy", False),
                "auto_zero_heavy": auto_flags.get("zero_heavy", False),
                "auto_long_reference": auto_flags.get("long_reference", False),
                "auto_null_empty": auto_flags.get("null_empty", False),
                "auto_aligned_empty": auto_flags.get("aligned_empty", False),
                "judge_sql_to_question": annotation.get("judge_sql_to_question", ""),
                "judge_aligned_answer_supported": annotation.get("judge_aligned_answer_supported", ""),
                "judge_null_shift_interpretable": annotation.get("judge_null_shift_interpretable", ""),
                "judge_not_pathological": annotation.get("judge_not_pathological", ""),
                "final_decision": annotation.get("final_decision", ""),
                "pathology_flags": annotation.get("pathology_flags", ""),
                "notes": annotation.get("notes", ""),
            }
            for label in prediction_labels:
                flat[prediction_fieldnames[label]] = _single_line(predictions.get(label, ""))
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a stratified review pool for the RAQUEL capability-controlled subset."
    )
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--query_type_tags", default="data/raquel/query_type_tags_heuristic.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument(
        "--split",
        default="affected",
        choices=["affected", "unaffected"],
        help="Split to export. The capability-controlled subset uses affected.",
    )
    parser.add_argument(
        "--anchor_eval",
        default="model/full_1b_s0/finetune/meta-llama/Llama-3.2-1B/raquel_eval_with_predictions.json",
        help=(
            "Evaluation file with saved predictions used to define the candidate pool. "
            "If provided, only examples present in this eval slice are sampled."
        ),
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        help="Optional LABEL=PATH eval specs. If omitted, seed-0 Morig/Mret/GA+GD/NPO+GD are used.",
    )
    parser.add_argument("--sample_size", type=int, default=80)
    parser.add_argument("--min_per_family", type=int, default=4)
    parser.add_argument("--sample_seed", type=int, default=20260329)
    parser.add_argument(
        "--out_prefix",
        default="reports/raquel/affected_solvable_subset_review",
        help="Prefix for exported artifacts.",
    )
    args = parser.parse_args()

    eval_specs = _parse_eval_specs(args.eval)
    prediction_rows = _load_query_index_rows(args.query_index_map, split=str(args.split))
    tags_by_index = _load_tags(args.query_type_tags)
    denotations_by_index = _load_denotations(args.denotations)

    if args.anchor_eval:
        anchor_predictions = _load_eval_predictions(args.anchor_eval, split=str(args.split))
        anchor_map = _map_eval_predictions_to_rows(
            eval_predictions=anchor_predictions,
            index_rows=prediction_rows,
        )
        anchor_order = {
            query_index: order
            for order, query_index in enumerate(anchor_map.keys())
        }
        prediction_rows = [
            row
            for row in prediction_rows
            if isinstance(row.get("query_index"), int) and int(row["query_index"]) in anchor_order
        ]
        prediction_rows = sorted(
            prediction_rows,
            key=lambda row: anchor_order.get(int(row["query_index"]), int(row.get("example_index", 0)))
        )

    pool_rows = _prepare_pool_rows(
        index_rows=prediction_rows,
        tags_by_index=tags_by_index,
        denotations_by_index=denotations_by_index,
    )
    sampled_rows, sample_summary = _stratified_sample(
        pool_rows,
        sample_size=int(args.sample_size),
        min_per_family=int(args.min_per_family),
        sample_seed=int(args.sample_seed),
    )

    predictions_by_label: Dict[str, Dict[int, str]] = {}
    for label, path in eval_specs:
        eval_predictions = _load_eval_predictions(path, split=str(args.split))
        predictions_by_label[label] = _map_eval_predictions_to_rows(
            eval_predictions=eval_predictions,
            index_rows=prediction_rows,
        )

    review_rows = _build_review_rows(
        sampled_rows=sampled_rows,
        predictions_by_label=predictions_by_label,
    )

    auto_flag_counts = Counter()
    for row in review_rows:
        for key, value in row.get("auto_flags", {}).items():
            if value:
                auto_flag_counts[key] += 1

    out_prefix = Path(args.out_prefix)
    sample_path = out_prefix.with_name(out_prefix.name + "_sample.json")
    sheet_path = out_prefix.with_name(out_prefix.name + "_sheet.csv")
    annotations_path = out_prefix.with_name(out_prefix.name + "_annotations.jsonl")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")

    sample_payload = {
        "metadata": {
            "split": args.split,
            "query_index_map": args.query_index_map,
            "query_type_tags": args.query_type_tags,
            "denotations": args.denotations,
            "anchor_eval": args.anchor_eval,
            "eval_specs": [{"label": label, "path": path} for label, path in eval_specs],
            "sample_size": args.sample_size,
            "min_per_family": args.min_per_family,
            "sample_seed": args.sample_seed,
        },
        "examples": review_rows,
    }

    annotations_template = [
        {
            "query_index": row["query_index"],
            "sample_order": row["sample_order"],
            "primary_family": row["primary_family"],
            "judge_sql_to_question": "",
            "judge_aligned_answer_supported": "",
            "judge_null_shift_interpretable": "",
            "judge_not_pathological": "",
            "final_decision": "",
            "pathology_flags": "",
            "notes": "",
        }
        for row in review_rows
    ]

    summary_payload = {
        "pool_size": len(pool_rows),
        "sample_size": len(review_rows),
        "sample_summary": sample_summary,
        "auto_flag_counts": dict(sorted(auto_flag_counts.items())),
        "artifacts": {
            "sample_json": str(sample_path),
            "sheet_csv": str(sheet_path),
            "annotations_jsonl": str(annotations_path),
        },
    }

    _write_json(sample_path, sample_payload)
    _write_csv(sheet_path, review_rows, [label for label, _path in eval_specs])
    _write_jsonl(annotations_path, annotations_template)
    _write_json(summary_path, summary_payload)

    print(f"Wrote {sample_path}")
    print(f"Wrote {sheet_path}")
    print(f"Wrote {annotations_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
