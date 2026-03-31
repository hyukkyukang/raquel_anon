"""Compare two RAQUEL generation runs for stability (artifact-level).

This script supports Priority 15 (stability) in `TODO/remaining.md` by comparing
two aligned DB artifact directories (e.g., `data/aligned_db` vs `data/aligned_db_backup`).

We report:
- key counts (seed QA pairs, synthesized queries, affected/unaffected sizes, skip rate)
- verification summary stats
- nullification removals
- hashes of question lists for quick equality checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_generation_stability", __file__)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_lines(lines: List[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _load_queries_txt(path: Path) -> List[str]:
    """Load translated NL queries (one per block/line) from synthesized_queries.txt."""
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            lines.append(s)
    return lines


def _extract_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Extract a compact stability summary from an aligned DB artifact directory."""
    run_dir = run_dir.resolve()

    pipeline_artifacts_path = Path("reports/paper/pipeline_artifacts_extracted.json")
    pipeline_artifacts: Optional[Dict[str, Any]] = None
    if pipeline_artifacts_path.exists():
        # This is repo-global, but still provides the key counts we care about.
        pipeline_artifacts = _read_json(pipeline_artifacts_path)

    qa_pairs_path = run_dir / "qa_pairs.jsonl"
    synthesized_queries_txt_path = run_dir / "synthesized_queries.txt"
    affected_path = run_dir / "affected_synthesized_queries_results.json"
    unaffected_path = run_dir / "unaffected_synthesized_queries_results.json"
    verification_summary_path = run_dir / "verification_summary.json"
    nullify_summary_path = run_dir / "log" / "nullify" / "summary.json"

    qa_pairs_total: int = 0
    if qa_pairs_path.exists():
        with open(qa_pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_pairs_total += 1

    translated_queries: List[str] = []
    if synthesized_queries_txt_path.exists():
        translated_queries = _load_queries_txt(synthesized_queries_txt_path)

    affected_count: Optional[int] = None
    unaffected_count: Optional[int] = None
    if affected_path.exists():
        affected_payload = _read_json(affected_path)
        affected_count = len(affected_payload) if isinstance(affected_payload, list) else None
    if unaffected_path.exists():
        unaffected_payload = _read_json(unaffected_path)
        unaffected_count = len(unaffected_payload) if isinstance(unaffected_payload, list) else None

    verification_summary: Optional[Dict[str, Any]] = None
    if verification_summary_path.exists():
        obj = _read_json(verification_summary_path)
        verification_summary = obj if isinstance(obj, dict) else None

    nullify_summary: Optional[Dict[str, Any]] = None
    if nullify_summary_path.exists():
        obj = _read_json(nullify_summary_path)
        nullify_summary = obj if isinstance(obj, dict) else None

    sql_queries_synthesized: Optional[int] = None
    skipped_queries: Optional[int] = None
    skipped_rate: Optional[float] = None

    if pipeline_artifacts:
        sql_queries_synthesized = int(pipeline_artifacts.get("sql_queries_synthesized", 0) or 0)
        skipped_queries = int(pipeline_artifacts.get("skipped_queries", 0) or 0)
        skipped_rate = float(pipeline_artifacts.get("skipped_rate", 0.0) or 0.0)
    else:
        # Fallback: approximate from translated queries and final results.
        if translated_queries:
            sql_queries_synthesized = len(translated_queries)
        if affected_count is not None and unaffected_count is not None and sql_queries_synthesized:
            skipped_queries = int(sql_queries_synthesized - affected_count - unaffected_count)
            skipped_rate = float(skipped_queries / sql_queries_synthesized) if sql_queries_synthesized > 0 else None

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "qa_pairs_total": qa_pairs_total,
        "translated_queries_total": len(translated_queries) if translated_queries else None,
        "translated_queries_sha256": _sha256_lines(translated_queries) if translated_queries else None,
        "sql_queries_synthesized": sql_queries_synthesized,
        "affected_count": affected_count,
        "unaffected_count": unaffected_count,
        "skipped_queries": skipped_queries,
        "skipped_rate": skipped_rate,
        "verification_summary": verification_summary,
        "nullify_entities_removed": (nullify_summary.get("entities_removed") if nullify_summary else None),
        "nullify_relations_removed": (nullify_summary.get("relations_removed") if nullify_summary else None),
    }
    return summary


def _diff(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out: Dict[str, Any] = {}
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            out[k] = _diff(a.get(k), b.get(k))
        return out
    if a == b:
        return {"equal": True, "value": a}
    return {"equal": False, "a": a, "b": b}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RAQUEL generation runs.")
    parser.add_argument("--run_a", default="data/aligned_db", help="First run dir (default: data/aligned_db).")
    parser.add_argument(
        "--run_b",
        default="data/aligned_db_backup",
        help="Second run dir (default: data/aligned_db_backup).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/raquel_generation_stability.json",
        help="Output JSON path (default: reports/paper/raquel_generation_stability.json).",
    )
    args = parser.parse_args()

    project_root: Path = Path(__file__).resolve().parents[2]
    run_a: Path = (project_root / str(args.run_a)).resolve()
    run_b: Path = (project_root / str(args.run_b)).resolve()
    out_path: Path = (project_root / str(args.out)).resolve()

    a = _extract_run_summary(run_a)
    b = _extract_run_summary(run_b)

    payload: Dict[str, Any] = {
        "run_a": a,
        "run_b": b,
        "diff": _diff(a, b),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

