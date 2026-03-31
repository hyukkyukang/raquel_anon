"""Compute a RAQUEL generation cost breakdown (calls/tokens/time).

This script supports Priority 15 in `TODO/remaining.md`:
- total calls/tokens per stage (discovery/extraction/verification/synthesis/translation)
- wall-clock time
- output: `reports/paper/raquel_cost_breakdown.json`

Important implementation notes (pragmatic + reproducible):

1) **Calls per stage**:
   RAQUEL does not currently log per-call usage by stage during generation, and
   the file counters in `log/api_call/` are cumulative across runs.
   Instead, we report run-level *call proxies* from saved intermediate artifacts
   for the TOFU-based run (e.g., #extractions=1400, #SQL queries=1329).

2) **Token usage per stage**:
   We estimate tokens per stage by sampling cached LLM responses from the
   DSPy/LiteLLM disk cache under `data/dspy/**/cache.db`, restricted to a target
   model (default: openai/gpt-5.4-nano-2026-03-17), and identifying stages using robust
   output markers (e.g., `[[ ## Text_query ## ]]`, `sql_template`, `CREATE TABLE`).
   We then multiply per-stage average tokens by the run-level call proxies.

This yields a stable report suitable for paper cost/scalability discussion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_cost_breakdown", __file__)


_STAGE_ORDER: List[str] = ["discovery", "extraction", "verification", "synthesis", "translation", "other"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def _find_dspy_cache_dbs(cache_root: Path) -> List[Path]:
    dbs: List[Path] = []
    for p in _iter_files(cache_root):
        if p.name == "cache.db":
            dbs.append(p)
    # Prefer shallower paths first (usually the "main" cache), but we will dedupe anyway.
    dbs.sort(key=lambda x: (len(x.parts), str(x)))
    return dbs


def _iter_cache_rows(
    cache_db: Path, start_ts: Optional[float], end_ts: Optional[float]
) -> Iterable[Tuple[str, float, Optional[bytes], Optional[str]]]:
    """Yield (key, store_time, value_bytes, filename) from a diskcache DB.

    If start_ts/end_ts are None, iterate the full DB.
    """
    con = sqlite3.connect(str(cache_db))
    try:
        cur = con.cursor()
        if start_ts is None or end_ts is None:
            cur.execute("SELECT key, store_time, value, filename FROM Cache")
        else:
            cur.execute(
                """
                SELECT key, store_time, value, filename
                FROM Cache
                WHERE store_time BETWEEN ? AND ?
                """,
                (start_ts, end_ts),
            )
        while True:
            rows = cur.fetchmany(2048)
            if not rows:
                break
            for key, store_time, value, filename in rows:
                key_str: str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
                store_t: float = float(store_time) if store_time is not None else 0.0
                value_bytes: Optional[bytes] = bytes(value) if value is not None else None
                filename_str: Optional[str] = str(filename) if filename is not None else None
                yield (key_str, store_t, value_bytes, filename_str)
    finally:
        con.close()


def _load_cached_object(
    cache_db: Path, value_bytes: Optional[bytes], filename: Optional[str]
) -> Optional[Mapping[str, Any]]:
    payload: Optional[bytes] = value_bytes
    if payload is None and filename:
        # diskcache stores external values relative to the cache directory.
        cache_dir: Path = cache_db.parent
        file_path: Path = cache_dir / filename
        try:
            payload = file_path.read_bytes()
        except FileNotFoundError:
            return None
    if payload is None:
        return None
    try:
        obj = pickle.loads(payload)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _parse_openai_response(response_raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(response_raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_content_and_usage(
    response_obj: Dict[str, Any],
) -> Optional[Tuple[str, str, Dict[str, int]]]:
    model = response_obj.get("model")
    if not isinstance(model, str):
        return None

    usage_obj = response_obj.get("usage")
    if not isinstance(usage_obj, dict):
        return None

    try:
        prompt_tokens: int = int(usage_obj.get("prompt_tokens", 0) or 0)
        completion_tokens: int = int(usage_obj.get("completion_tokens", 0) or 0)
        total_tokens_raw = usage_obj.get("total_tokens", None)
        total_tokens: int = (
            int(total_tokens_raw)
            if total_tokens_raw is not None
            else int(prompt_tokens + completion_tokens)
        )
    except Exception:
        return None

    choices = response_obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, str):
        return None

    return (
        model,
        content,
        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
    )


class _TokenAgg:
    """Streaming token statistics for a filtered set of cache entries."""

    def __init__(self) -> None:
        self.count: int = 0
        self.sum_prompt_tokens: int = 0
        self.sum_completion_tokens: int = 0
        self.sum_total_tokens: int = 0
        self.min_store_time: Optional[float] = None
        self.max_store_time: Optional[float] = None

    def add(self, store_time: float, usage: Dict[str, int]) -> None:
        self.count += 1
        self.sum_prompt_tokens += int(usage["prompt_tokens"])
        self.sum_completion_tokens += int(usage["completion_tokens"])
        self.sum_total_tokens += int(usage["total_tokens"])
        self.min_store_time = store_time if self.min_store_time is None else min(self.min_store_time, store_time)
        self.max_store_time = store_time if self.max_store_time is None else max(self.max_store_time, store_time)

    def to_summary(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "avg_total_tokens": 0.0,
                "min_store_time_epoch_s": None,
                "max_store_time_epoch_s": None,
            }
        return {
            "count": self.count,
            "avg_prompt_tokens": float(self.sum_prompt_tokens / self.count),
            "avg_completion_tokens": float(self.sum_completion_tokens / self.count),
            "avg_total_tokens": float(self.sum_total_tokens / self.count),
            "min_store_time_epoch_s": self.min_store_time,
            "max_store_time_epoch_s": self.max_store_time,
        }


def _stage_from_content(content: str) -> str:
    """Best-effort stage classification from output markers."""
    lower = content.lower()
    if "[[ ## text_query ## ]]" in lower:
        # NOTE: In `main()` we further restrict translation samples to queries that
        # appear in `data/aligned_db/synthesized_queries.txt` for the target run.
        return "translation"
    if "[[ ## paraphrased_text ## ]]" in lower:
        return "translation"
    if "sql_template" in lower:
        return "synthesis"
    if "create table" in lower:
        return "discovery"
    # Round-trip verification / judgment outputs.
    if "similarity_score" in lower and "sufficient" in lower:
        return "verification"
    if "match_quality" in lower and "similarity_score" in lower:
        return "verification"
    # Generic evaluation-style outputs (used by some checkers).
    if "[[ ## evaluation ## ]]" in lower and "\"match\"" in lower:
        return "verification"
    if "[[ ## response ## ]]" in lower and "\"entities\"" in lower and "\"relations\"" in lower:
        return "extraction"
    return "other"


def _load_translated_query_set(aligned_db_dir: Path) -> Set[str]:
    """Load the run's translated NL queries as a set of normalized strings."""
    translated_path: Path = aligned_db_dir / "synthesized_queries.txt"
    if not translated_path.exists():
        raise FileNotFoundError(f"Missing translated queries file: {translated_path}")

    queries: Set[str] = set()
    for line in translated_path.read_text(encoding="utf-8").splitlines():
        q = line.strip()
        if q:
            queries.add(q)
    return queries


def _extract_text_query_from_content(content: str) -> Optional[str]:
    """Extract the first non-empty line after the `Text_query` marker."""
    marker = "[[ ## Text_query ## ]]"
    if marker not in content:
        return None
    parts = content.split(marker, 1)
    if len(parts) != 2:
        return None
    after = parts[1]
    for line in after.splitlines():
        s = line.strip()
        # Stop at the next marker (if any)
        if s.startswith("[["):
            return None
        if s:
            return s
    return None


def _load_call_proxies(
    aligned_db_dir: Path, pipeline_artifacts_path: Path
) -> Dict[str, Any]:
    """Load run-level *call proxies* from saved artifacts."""
    pipeline_summary_path: Path = aligned_db_dir / "log" / "aligned_db_pipeline" / "summary" / "pipeline_summary.json"
    verification_summary_path: Path = aligned_db_dir / "verification_summary.json"

    if not pipeline_summary_path.exists():
        raise FileNotFoundError(f"Missing pipeline summary: {pipeline_summary_path}")
    if not verification_summary_path.exists():
        raise FileNotFoundError(f"Missing verification summary: {verification_summary_path}")
    if not pipeline_artifacts_path.exists():
        raise FileNotFoundError(f"Missing pipeline artifact summary: {pipeline_artifacts_path}")

    pipeline_summary = json.loads(pipeline_summary_path.read_text(encoding="utf-8"))
    verification_summary = json.loads(verification_summary_path.read_text(encoding="utf-8"))
    pipeline_artifacts = json.loads(pipeline_artifacts_path.read_text(encoding="utf-8"))

    # Discovery proxies
    stage1 = pipeline_summary.get("stage1_entity_discovery", {})
    stage2 = pipeline_summary.get("stage2_attribute_discovery", {})
    stage3 = pipeline_summary.get("stage3_schema_generation", {})
    entity_types_discovered = int(stage1.get("entity_types_discovered", 0) or 0)
    total_attributes = int(stage2.get("total_attributes", 0) or 0)
    relations_discovered = int(stage2.get("relations_discovered", 0) or 0)
    tables_count = int(stage3.get("tables_count", 0) or 0)
    discovery_calls_proxy = int(entity_types_discovered + total_attributes + relations_discovered + tables_count)

    # Extraction proxies
    stage4 = pipeline_summary.get("stage4_extraction", {})
    per_qa_extractions = int(stage4.get("extractions_count", 0) or 0)

    # Verification proxies
    verification_total = int(verification_summary.get("total", 0) or 0)

    # Synthesis / translation proxies
    sql_queries_synthesized = int(pipeline_artifacts.get("sql_queries_synthesized", 0) or 0)

    return {
        "discovery": {
            "calls_proxy_total": discovery_calls_proxy,
            "entity_types_discovered": entity_types_discovered,
            "attributes_discovered": total_attributes,
            "relations_discovered": relations_discovered,
            "tables_in_schema": tables_count,
        },
        "extraction": {"calls_proxy_total": per_qa_extractions},
        "verification": {"calls_proxy_total": verification_total},
        "synthesis": {"calls_proxy_total": sql_queries_synthesized},
        "translation": {"calls_proxy_total": sql_queries_synthesized},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RAQUEL cost breakdown.")
    parser.add_argument(
        "--aligned_db_dir",
        default="data/aligned_db",
        help="Aligned DB artifact directory (default: data/aligned_db).",
    )
    parser.add_argument(
        "--dspy_cache_root",
        default="data/dspy",
        help="Root directory containing DSPy disk cache DB(s) (default: data/dspy).",
    )
    parser.add_argument(
        "--pipeline_artifacts",
        default="reports/paper/pipeline_artifacts_extracted.json",
        help="Run-level artifact summary JSON (default: reports/paper/pipeline_artifacts_extracted.json).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_cost_breakdown.json",
        help="Output JSON path (default: reports/paper/raquel_cost_breakdown.json).",
    )
    parser.add_argument(
        "--model_name",
        default="openai/gpt-5.4-nano-2026-03-17",
        help=(
            "Only sample cached responses for this model "
            "(default: openai/gpt-5.4-nano-2026-03-17)."
        ),
    )
    parser.add_argument(
        "--run_window_pad_days",
        type=float,
        default=7.0,
        help=(
            "When estimating wall-clock, we anchor on the translation window (queries in "
            "`synthesized_queries.txt`) and include cached calls up to this many days before it "
            "(default: 7.0)."
        ),
    )
    args = parser.parse_args()

    project_root: Path = Path(__file__).resolve().parents[2]
    aligned_db_dir: Path = (project_root / str(args.aligned_db_dir)).resolve()
    dspy_cache_root: Path = (project_root / str(args.dspy_cache_root)).resolve()
    pipeline_artifacts_path: Path = (project_root / str(args.pipeline_artifacts)).resolve()
    out_path: Path = (project_root / str(args.out)).resolve()

    call_proxies: Dict[str, Any] = _load_call_proxies(
        aligned_db_dir=aligned_db_dir,
        pipeline_artifacts_path=pipeline_artifacts_path,
    )

    # Token sampling from cache: scan once, assign stage by output marker.
    cache_dbs: List[Path] = _find_dspy_cache_dbs(dspy_cache_root)
    if not cache_dbs:
        raise FileNotFoundError(f"No cache.db files found under: {dspy_cache_root}")

    model_name: str = str(args.model_name)

    translated_query_set: Set[str] = _load_translated_query_set(aligned_db_dir)

    # Pass 1: identify translation window for the run (Text_query outputs matching our file).
    translation_min_ts: Optional[float] = None
    translation_max_ts: Optional[float] = None
    seen_keys_pass1: Set[str] = set()
    for db in cache_dbs:
        for key, store_time, value_bytes, filename in _iter_cache_rows(db, start_ts=None, end_ts=None):
            if key in seen_keys_pass1:
                continue
            seen_keys_pass1.add(key)

            cached = _load_cached_object(db, value_bytes=value_bytes, filename=filename)
            if cached is None:
                continue
            response_raw = cached.get("response")
            if not isinstance(response_raw, str):
                continue
            resp_obj = _parse_openai_response(response_raw)
            if resp_obj is None:
                continue
            extracted = _extract_content_and_usage(resp_obj)
            if extracted is None:
                continue
            model, content, _usage = extracted
            if model != model_name:
                continue
            if "[[ ## text_query ## ]]" not in content.lower():
                continue
            text_query = _extract_text_query_from_content(content)
            if text_query is None or text_query.strip() not in translated_query_set:
                continue
            translation_min_ts = store_time if translation_min_ts is None else min(translation_min_ts, store_time)
            translation_max_ts = store_time if translation_max_ts is None else max(translation_max_ts, store_time)

    if translation_min_ts is None or translation_max_ts is None:
        raise RuntimeError(
            "Failed to locate any cached Text_query outputs matching `synthesized_queries.txt`. "
            "Cannot anchor the run window."
        )

    pad_days: float = float(args.run_window_pad_days)
    run_start_ts: float = float(translation_min_ts - pad_days * 86400.0)
    run_end_ts: float = float(translation_max_ts)

    # Pass 2: aggregate token samples within run window, restricting translation to in-run queries.
    agg_by_stage: Dict[str, _TokenAgg] = {s: _TokenAgg() for s in _STAGE_ORDER}
    seen_keys: Set[str] = set()
    for db in cache_dbs:
        for key, store_time, value_bytes, filename in _iter_cache_rows(
            db, start_ts=run_start_ts, end_ts=run_end_ts
        ):
            if key in seen_keys:
                continue
            seen_keys.add(key)

            cached = _load_cached_object(db, value_bytes=value_bytes, filename=filename)
            if cached is None:
                continue
            response_raw = cached.get("response")
            if not isinstance(response_raw, str):
                continue
            resp_obj = _parse_openai_response(response_raw)
            if resp_obj is None:
                continue
            extracted = _extract_content_and_usage(resp_obj)
            if extracted is None:
                continue
            model, content, usage = extracted
            if model != model_name:
                continue

            stage: str = _stage_from_content(content)
            if stage == "translation" and "[[ ## text_query ## ]]" in content.lower():
                text_query = _extract_text_query_from_content(content)
                if text_query is None or text_query.strip() not in translated_query_set:
                    stage = "other"

            agg_by_stage[stage].add(store_time=store_time, usage=usage)

    # Convert aggregates -> summaries.
    token_samples: Dict[str, Any] = {stage: agg.to_summary() for stage, agg in agg_by_stage.items()}

    # Estimated per-stage token totals = avg_tokens_per_call * calls_proxy_total.
    token_estimates: Dict[str, Any] = {}
    for stage in ["discovery", "extraction", "verification", "synthesis", "translation"]:
        calls_proxy_total = float(call_proxies.get(stage, {}).get("calls_proxy_total", 0) or 0)
        sample = token_samples.get(stage, {})
        avg_prompt = float(sample.get("avg_prompt_tokens", 0.0) or 0.0)
        avg_completion = float(sample.get("avg_completion_tokens", 0.0) or 0.0)
        avg_total = float(sample.get("avg_total_tokens", 0.0) or 0.0)
        token_estimates[stage] = {
            "calls_proxy_total": calls_proxy_total,
            "estimated_prompt_tokens": float(calls_proxy_total * avg_prompt),
            "estimated_completion_tokens": float(calls_proxy_total * avg_completion),
            "estimated_total_tokens": float(calls_proxy_total * avg_total),
        }

    # Wall-clock estimate: span between earliest and latest cached call timestamps in samples.
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None
    for stage in ["discovery", "extraction", "verification", "synthesis", "translation"]:
        s = token_samples.get(stage, {})
        mn = s.get("min_store_time_epoch_s")
        mx = s.get("max_store_time_epoch_s")
        if isinstance(mn, (int, float)):
            min_ts = float(mn) if min_ts is None else min(min_ts, float(mn))
        if isinstance(mx, (int, float)):
            max_ts = float(mx) if max_ts is None else max(max_ts, float(mx))

    wall_clock: Dict[str, Any] = {
        "note": "Estimated from cache store_time span of stage samples (not from filesystem mtimes).",
        "start_epoch_s": min_ts,
        "end_epoch_s": max_ts,
        "duration_s": float(max_ts - min_ts) if (min_ts is not None and max_ts is not None) else None,
        "start_utc": dt.datetime.fromtimestamp(min_ts, tz=dt.timezone.utc).isoformat() if min_ts else None,
        "end_utc": dt.datetime.fromtimestamp(max_ts, tz=dt.timezone.utc).isoformat() if max_ts else None,
        "run_window": {
            "anchor": "Text_query outputs matching synthesized_queries.txt",
            "translation_window": {
                "start_epoch_s": translation_min_ts,
                "end_epoch_s": translation_max_ts,
                "start_utc": dt.datetime.fromtimestamp(translation_min_ts, tz=dt.timezone.utc).isoformat()
                if translation_min_ts
                else None,
                "end_utc": dt.datetime.fromtimestamp(translation_max_ts, tz=dt.timezone.utc).isoformat()
                if translation_max_ts
                else None,
            },
            "pad_days": pad_days,
            "run_start_epoch_s": run_start_ts,
            "run_end_epoch_s": run_end_ts,
            "run_start_utc": dt.datetime.fromtimestamp(run_start_ts, tz=dt.timezone.utc).isoformat(),
            "run_end_utc": dt.datetime.fromtimestamp(run_end_ts, tz=dt.timezone.utc).isoformat(),
        },
    }

    payload: Dict[str, Any] = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "inputs": {
            "aligned_db_dir": str(aligned_db_dir),
            "dspy_cache_root": str(dspy_cache_root),
            "pipeline_artifacts": str(pipeline_artifacts_path),
            "model_name": model_name,
        },
        "calls_proxy_by_stage": call_proxies,
        "token_samples_from_cache_by_stage": token_samples,
        "token_estimates_by_stage": token_estimates,
        "wall_clock_estimate": wall_clock,
        "notes": {
            "cache_dbs_used": [str(p) for p in cache_dbs],
            "unique_cache_keys_scanned": len(seen_keys),
            "stage_classifier": "See script/analysis/raquel_cost_breakdown.py::_stage_from_content",
        },
    }

    _write_json(out_path, payload)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
