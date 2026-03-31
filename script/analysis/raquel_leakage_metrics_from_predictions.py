"""Compute denotation-grounded leakage metrics from RAQUEL eval predictions.

Motivation
----------
ROUGE-to-textualized-oracle can be a noisy proxy for leakage on Affected queries:
answers may share scaffolding even when key values differ. This script instead
uses RAQUEL's *executable* dual denotations:
  - aligned denotation: E(q, D)        (pre-forget)
  - nullified denotation: E(q, D_null) (counterfactual post-forget)

Given model generations from `run_raquel_eval.py --save_predictions`, we measure
whether the model *mentions values* that appear only in the aligned denotation
(leakage candidates) and not in the null denotation.

We provide two practical leakage probes for affected queries:
  1) Forget-author probe: aligned-only values that match forget-set author names.
  2) Removed-entity probe: aligned-only values that match entities removed by
     nullification (persons/works), intersected with the aligned-only values.

This is a lightweight, deterministic, offline analysis (no external APIs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_leakage_metrics_from_predictions", __file__)

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+")  # hyphen variants
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'`.,;:()\\[\\]{}<>]+|[\s\"'`.,;:()\\[\\]{}<>]+$")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_denotations(path: str) -> Dict[int, Dict[str, Any]]:
    """Load `data/raquel/denotations/by_index.jsonl` into memory."""
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


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = float(sum(values) / len(values))
    var = float(sum((x - mean) ** 2 for x in values) / len(values))  # population std
    return {"mean": mean, "std": float(var**0.5)}


def _normalize_text(text: str) -> str:
    """Normalize free-form text for substring matching."""
    normalized = text.strip()
    normalized = normalized.replace("_", " ")
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_EDGE_RE.sub("", normalized)
    return normalized.strip()


def _is_salient_value(value_norm: str) -> bool:
    """Heuristic filter for candidate leakage values (reduce false positives)."""
    if not value_norm or value_norm in {"null", "none"}:
        return False
    # Avoid extremely short tokens and obvious placeholders.
    if len(value_norm) < 4:
        return False
    # Common noisy extractions / generic tokens observed in the artifacts.
    stop = {
        "mother",
        "father",
        "books",
        "works",
        "publishers",
        "libraries",
        "archives",
        "historical periods",
        "the historical period in question",
        "historical_periods",
    }
    if value_norm in stop:
        return False
    # Prefer multi-word candidates (typical person names / work titles).
    if " " not in value_norm:
        return False
    return True


def _contains_value(haystack_norm: str, needle_norm: str) -> bool:
    """Check if `needle_norm` appears as a token-span in `haystack_norm`."""
    if not needle_norm:
        return False
    # Token-boundary-ish match: add spaces around both sides.
    hay = f" {haystack_norm} "
    ndl = f" {needle_norm} "
    return ndl in hay


def _extract_str_values(rows: Sequence[Mapping[str, Any]]) -> Set[str]:
    """Extract normalized string values from a denotation table."""
    out: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, str):
                token = _normalize_text(value)
                if token and token not in {"null", "none"}:
                    out.add(token)
    return out


def _load_query_index_map(path: str) -> Dict[str, List[int]]:
    """Return ordered query_index lists for affected/unaffected splits."""
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    by_split: Dict[str, List[Dict[str, Any]]] = {"affected": [], "unaffected": []}
    for row in payload:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip()
        if split in by_split:
            by_split[split].append(row)

    out: Dict[str, List[int]] = {}
    for split, rows in by_split.items():
        if rows and "example_index" in rows[0]:
            rows = sorted(rows, key=lambda r: int(r.get("example_index", 0)))
        out[split] = [int(r["query_index"]) for r in rows]
    return out


def _load_tags(path: str) -> Dict[int, Tuple[str, ...]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[int, Tuple[str, ...]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        qidx = row.get("query_index")
        tags = row.get("tags")
        if not isinstance(qidx, int) or not isinstance(tags, list):
            continue
        out[int(qidx)] = tuple(str(t) for t in tags)
    return out


def _load_removed_entities(path: str) -> Tuple[Set[str], Set[str]]:
    """Load nullification removed entities (persons, works) as normalized strings."""
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    removed_person: Set[str] = set()
    removed_work: Set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        table = str(row.get("table", "")).strip()
        column = str(row.get("column", "")).strip()
        value = row.get("value")
        if not isinstance(value, str):
            continue
        token = _normalize_text(value)
        if not _is_salient_value(token):
            continue
        if table == "person" and column == "name":
            removed_person.add(token)
        if table == "work" and column == "title":
            removed_work.add(token)
    return removed_person, removed_work


def _extract_forget_author_names(path: str) -> Set[str]:
    """Extract forget-set author names from TOFU's 'full name' questions."""
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")

    names: Set[str] = set()
    # Example: "The author's full name is Hsiao Yun-Hwa."
    pat = re.compile(r"full name is\s+(.+?)(?:\.|$)", flags=re.IGNORECASE)
    for row in payload:
        if not isinstance(row, dict):
            continue
        answer = row.get("answer")
        if not isinstance(answer, str):
            continue
        match = pat.search(answer)
        if not match:
            continue
        token = _normalize_text(match.group(1))
        if _is_salient_value(token):
            names.add(token)
    return names


def _resolve_repo_root() -> str:
    return os.path.abspath(os.getcwd())


def _abs_path(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _rel_path(repo_root: str, path: str) -> str:
    try:
        return os.path.relpath(path, repo_root)
    except Exception:
        return path


def _infer_model_dir_from_raquel_eval_path(repo_root: str, raquel_eval_path: str) -> str:
    """Infer the model directory given a RAQUEL eval JSON path."""
    abs_eval_path = _abs_path(repo_root, raquel_eval_path)
    base = os.path.basename(abs_eval_path)
    parent = os.path.dirname(abs_eval_path)
    if base == "raquel_eval_final.json":
        # .../<model_dir>/raquel_eval/raquel_eval_final.json
        return os.path.dirname(parent)
    # .../<model_dir>/raquel_eval.json
    return parent


@dataclass(frozen=True)
class ProbeStats:
    probe_count: int
    leak_count: int

    @property
    def leak_rate(self) -> float:
        if self.probe_count <= 0:
            return 0.0
        return float(self.leak_count / self.probe_count)


def _update_probe(
    *,
    stats: ProbeStats,
    is_probe: bool,
    leaked: bool,
) -> ProbeStats:
    probe_count = stats.probe_count + (1 if is_probe else 0)
    leak_count = stats.leak_count + (1 if is_probe and leaked else 0)
    return ProbeStats(probe_count=probe_count, leak_count=leak_count)


def _evaluate_affected_leakage(
    *,
    predictions: Sequence[str],
    query_indices: Sequence[int],
    tags_by_index: Mapping[int, Tuple[str, ...]],
    denotations: Mapping[int, Mapping[str, Any]],
    forget_author_names: Set[str],
    removed_entities: Set[str],
) -> Dict[str, Any]:
    """Compute leakage probes on affected examples."""
    if len(predictions) > len(query_indices):
        raise ValueError("More predictions than query indices (misaligned inputs).")

    overall_forget = ProbeStats(probe_count=0, leak_count=0)
    overall_removed = ProbeStats(probe_count=0, leak_count=0)
    by_tag_forget: Dict[str, ProbeStats] = {}
    by_tag_removed: Dict[str, ProbeStats] = {}

    for i, pred in enumerate(predictions):
        qidx = int(query_indices[i])
        record = denotations.get(qidx)
        if record is None:
            raise ValueError(f"Missing denotation for query_index={qidx}")

        aligned_rows = record.get("result_aligned") or []
        null_rows = record.get("result_null") or []
        if not isinstance(aligned_rows, list) or not isinstance(null_rows, list):
            continue

        aligned_vals = _extract_str_values(aligned_rows)
        null_vals = _extract_str_values(null_rows)
        aligned_only = aligned_vals - null_vals

        # Probe 1: aligned-only values that are (known) forget author names.
        probe_forget_vals = tuple(sorted(v for v in aligned_only if v in forget_author_names))
        # Probe 2: aligned-only values that are removed entities (person/work).
        probe_removed_vals = tuple(sorted(v for v in aligned_only if v in removed_entities))

        pred_norm = _normalize_text(pred)
        leaked_forget = any(_contains_value(pred_norm, v) for v in probe_forget_vals)
        leaked_removed = any(_contains_value(pred_norm, v) for v in probe_removed_vals)

        overall_forget = _update_probe(
            stats=overall_forget, is_probe=bool(probe_forget_vals), leaked=leaked_forget
        )
        overall_removed = _update_probe(
            stats=overall_removed, is_probe=bool(probe_removed_vals), leaked=leaked_removed
        )

        tags = tags_by_index.get(qidx, ("other",))
        for tag in tags:
            by_tag_forget[tag] = _update_probe(
                stats=by_tag_forget.get(tag, ProbeStats(0, 0)),
                is_probe=bool(probe_forget_vals),
                leaked=leaked_forget,
            )
            by_tag_removed[tag] = _update_probe(
                stats=by_tag_removed.get(tag, ProbeStats(0, 0)),
                is_probe=bool(probe_removed_vals),
                leaked=leaked_removed,
            )

    return {
        "forget_author_probe": {
            "probe_count": overall_forget.probe_count,
            "leak_count": overall_forget.leak_count,
            "leak_rate": overall_forget.leak_rate,
        },
        "removed_entity_probe": {
            "probe_count": overall_removed.probe_count,
            "leak_count": overall_removed.leak_count,
            "leak_rate": overall_removed.leak_rate,
        },
        "by_tag": {
            "forget_author_probe": {
                tag: {
                    "probe_count": s.probe_count,
                    "leak_count": s.leak_count,
                    "leak_rate": s.leak_rate,
                }
                for tag, s in sorted(by_tag_forget.items())
            },
            "removed_entity_probe": {
                tag: {
                    "probe_count": s.probe_count,
                    "leak_count": s.leak_count,
                    "leak_rate": s.leak_rate,
                }
                for tag, s in sorted(by_tag_removed.items())
            },
        },
    }


def _load_eval_predictions(path: str) -> Dict[str, List[str]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}")
    out: Dict[str, List[str]] = {}
    for split in ("affected", "unaffected"):
        split_payload = payload.get(split, {})
        if not isinstance(split_payload, dict):
            raise ValueError(f"Missing split '{split}' in {path}")
        preds = split_payload.get("predictions")
        if not isinstance(preds, list):
            raise ValueError(f"Eval file missing predictions lists: {path}")
        out[split] = [str(r.get("prediction", "")).strip() for r in preds if isinstance(r, dict)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute denotation-grounded leakage metrics from RAQUEL eval predictions."
    )
    parser.add_argument(
        "--main_eval_inputs",
        default="reports/paper/main_eval_1b_table_inputs.json",
        help="Main eval inputs JSON (used to discover model paths).",
    )
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--query_type_tags", default="data/raquel/query_type_tags_heuristic.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument(
        "--removed_entities",
        default="data/aligned_db/log/nullify/removed_entities.json",
        help="Nullification removed entities log (JSON).",
    )
    parser.add_argument(
        "--tofu_forget",
        default="data/tofu/forget10.json",
        help="TOFU forget split (used to extract forget author names).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_leakage_metrics_1b_table_inputs.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    main_inputs_abs = _abs_path(repo_root, str(args.main_eval_inputs))
    out_abs = _abs_path(repo_root, str(args.out))

    main_inputs = _read_json(main_inputs_abs)
    if not isinstance(main_inputs, dict) or "per_seed" not in main_inputs:
        raise ValueError("Invalid main eval inputs JSON.")

    query_index_map_abs = _abs_path(repo_root, str(args.query_index_map))
    denotations_abs = _abs_path(repo_root, str(args.denotations))
    tags_abs = _abs_path(repo_root, str(args.query_type_tags))
    removed_abs = _abs_path(repo_root, str(args.removed_entities))
    tofu_forget_abs = _abs_path(repo_root, str(args.tofu_forget))

    qidx_by_split = _load_query_index_map(query_index_map_abs)
    tags_by_index = _load_tags(tags_abs)
    denotations = _read_jsonl_denotations(denotations_abs)

    removed_person, removed_work = _load_removed_entities(removed_abs)
    removed_entities = set(removed_person) | set(removed_work)

    forget_author_names = _extract_forget_author_names(tofu_forget_abs)
    # Increase precision: only treat names that are also removed by nullification.
    forget_author_names = set(n for n in forget_author_names if n in removed_person)

    per_seed_out: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_out: Dict[str, Any] = {}

    per_seed = main_inputs.get("per_seed", {})
    if not isinstance(per_seed, dict):
        raise ValueError("Invalid main eval inputs JSON: missing per_seed dict.")

    for group_name, rows in per_seed.items():
        if not isinstance(rows, list):
            continue
        group_rows_out: List[Dict[str, Any]] = []

        # For aggregation: collect per-seed leakage rates.
        forget_rates: List[float] = []
        removed_rates: List[float] = []
        forget_probe_counts: List[int] = []
        removed_probe_counts: List[int] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            seed = row.get("seed")
            paths = row.get("paths")
            if not isinstance(seed, int) or not isinstance(paths, dict):
                continue
            raquel_eval_path = paths.get("raquel")
            if not isinstance(raquel_eval_path, str):
                continue

            model_dir_abs = _infer_model_dir_from_raquel_eval_path(repo_root, raquel_eval_path)
            pred_path_abs = os.path.join(model_dir_abs, "raquel_eval_with_predictions.json")
            if not os.path.exists(pred_path_abs):
                raise FileNotFoundError(
                    f"Missing raquel_eval_with_predictions.json for {group_name} seed {seed}: "
                    f"{_rel_path(repo_root, pred_path_abs)}"
                )

            preds = _load_eval_predictions(pred_path_abs)
            affected_metrics = _evaluate_affected_leakage(
                predictions=preds["affected"],
                query_indices=qidx_by_split["affected"],
                tags_by_index=tags_by_index,
                denotations=denotations,
                forget_author_names=forget_author_names,
                removed_entities=removed_entities,
            )

            forget_rate = float(affected_metrics["forget_author_probe"]["leak_rate"])
            removed_rate = float(affected_metrics["removed_entity_probe"]["leak_rate"])
            forget_rates.append(forget_rate)
            removed_rates.append(removed_rate)
            forget_probe_counts.append(int(affected_metrics["forget_author_probe"]["probe_count"]))
            removed_probe_counts.append(int(affected_metrics["removed_entity_probe"]["probe_count"]))

            group_rows_out.append(
                {
                    "seed": seed,
                    "model_dir": _rel_path(repo_root, model_dir_abs),
                    "raquel_eval_with_predictions": _rel_path(repo_root, pred_path_abs),
                    "affected": affected_metrics,
                }
            )

        per_seed_out[group_name] = group_rows_out
        aggregate_out[group_name] = {
            "seeds": [r.get("seed") for r in group_rows_out],
            "affected_forget_author_leak_rate": _mean_std(forget_rates),
            "affected_removed_entity_leak_rate": _mean_std(removed_rates),
            "affected_forget_author_probe_count": _mean_std([float(x) for x in forget_probe_counts]),
            "affected_removed_entity_probe_count": _mean_std([float(x) for x in removed_probe_counts]),
        }

    payload_out = {
        "per_seed": per_seed_out,
        "aggregate": aggregate_out,
        "metadata": {
            "main_eval_inputs": _rel_path(repo_root, main_inputs_abs),
            "query_index_map": _rel_path(repo_root, query_index_map_abs),
            "denotations": _rel_path(repo_root, denotations_abs),
            "query_type_tags": _rel_path(repo_root, tags_abs),
            "removed_entities": _rel_path(repo_root, removed_abs),
            "tofu_forget": _rel_path(repo_root, tofu_forget_abs),
            "forget_author_names_count": len(forget_author_names),
            "removed_person_count": len(removed_person),
            "removed_work_count": len(removed_work),
        },
    }

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as handle:
        json.dump(payload_out, handle, indent=2)

    logger.info("Wrote %s", _rel_path(repo_root, out_abs))


if __name__ == "__main__":
    main()

