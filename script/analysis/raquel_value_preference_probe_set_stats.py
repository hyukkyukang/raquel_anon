"""Summarize the RAQUEL value-preference leakage probe set.

The value-preference evaluation (see `script/evaluation/run_raquel_value_preference_eval.py`)
operates on a *high-precision affected subset*:
  - aligned-only values contain at least one removed (forget-attributable) entity value
  - null-only values are non-empty (so we can form a contrast set)

This script reproduces the probe-set selection deterministically and reports:
  - probe-set size
  - tag distribution (JOIN / MULTI_JOIN / etc.)

Output is saved under `reports/paper/` for easy inclusion in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Set

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_value_preference_probe_set_stats", __file__)

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+")  # hyphen variants
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'`.,;:()\\[\\]{}<>]+|[\s\"'`.,;:()\\[\\]{}<>]+$")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_denotations(path: str) -> Dict[int, Dict[str, Any]]:
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
            if isinstance(qidx, int):
                out[int(qidx)] = rec
    return out


def _normalize(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("_", " ")
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = normalized.lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_EDGE_RE.sub("", normalized)
    return normalized.strip()


def _is_salient(token: str) -> bool:
    if not token or token in {"null", "none"}:
        return False
    if len(token) < 3:
        return False
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
    return token not in stop


def _extract_norm_values(rows: Any) -> Set[str]:
    if not isinstance(rows, list):
        return set()
    out: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for v in row.values():
            if isinstance(v, str):
                t = _normalize(v)
                if _is_salient(t):
                    out.add(t)
    return out


def _load_removed_values(path: str) -> Set[str]:
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
        t = _normalize(value)
        if _is_salient(t) and (" " in t):
            out.add(t)
    return out


def _load_tags(path: str) -> Dict[int, List[str]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[int, List[str]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        qidx = row.get("query_index")
        tags = row.get("tags")
        if isinstance(qidx, int) and isinstance(tags, list):
            out[int(qidx)] = [str(t) for t in tags]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RAQUEL value-preference probe set.")
    parser.add_argument("--query_index_map", default="data/raquel/query_index_map.json")
    parser.add_argument("--denotations", default="data/raquel/denotations/by_index.jsonl")
    parser.add_argument("--query_type_tags", default="data/raquel/query_type_tags_heuristic.json")
    parser.add_argument("--removed_entities", default="data/aligned_db/log/nullify/removed_entities.json")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_value_preference_probe_set_stats.json",
    )
    args = parser.parse_args()

    qmap = _read_json(str(args.query_index_map))
    if not isinstance(qmap, list):
        raise ValueError("Expected list query_index_map.")
    affected = [r for r in qmap if isinstance(r, dict) and r.get("split") == "affected"]
    affected = sorted(affected, key=lambda r: int(r.get("example_index", 0)))

    den = _read_denotations(str(args.denotations))
    removed = _load_removed_values(str(args.removed_entities))
    tags_by_index = _load_tags(str(args.query_type_tags))

    candidates: List[int] = []
    for r in affected:
        qidx = int(r["query_index"])
        rec = den.get(qidx)
        if rec is None:
            continue
        a_vals = _extract_norm_values(rec.get("result_aligned") or [])
        n_vals = _extract_norm_values(rec.get("result_null") or [])
        a_only = (a_vals - n_vals) & removed
        n_only = n_vals - a_vals
        if a_only and n_only:
            candidates.append(qidx)

    rng = random.Random(int(args.shuffle_seed))
    rng.shuffle(candidates)
    selected = candidates[: int(args.max_examples)] if int(args.max_examples) > 0 else candidates

    tag_counter: Counter[str] = Counter()
    for qidx in selected:
        for tag in tags_by_index.get(qidx, ["other"]):
            tag_counter[tag] += 1

    payload_out = {
        "probe_set_size": len(selected),
        "candidate_pool_size": len(candidates),
        "shuffle_seed": int(args.shuffle_seed),
        "max_examples": int(args.max_examples),
        "tag_overlap_counts": dict(tag_counter.most_common()),
        "metadata": {
            "query_index_map": str(args.query_index_map),
            "denotations": str(args.denotations),
            "query_type_tags": str(args.query_type_tags),
            "removed_entities": str(args.removed_entities),
        },
    }

    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    with open(str(args.out), "w", encoding="utf-8") as handle:
        json.dump(payload_out, handle, indent=2)

    logger.info("Wrote %s", str(args.out))


if __name__ == "__main__":
    main()

