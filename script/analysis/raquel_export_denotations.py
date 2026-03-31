"""Export dual-oracle denotations for RAQUEL synthesized queries.

This script executes each synthesized SQL query against:
  - the aligned database (pre-forget world), and
  - the nullified database (counterfactual post-forget world),
and writes the raw denotations to a JSONL file (one record per query_index).

This is designed to support denotation/value-level evaluation of unlearning.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional

import hkkang_utils.pg as pg_utils
import tqdm

from src.utils.data_loaders import load_sql_queries
from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_export_denotations", __file__)


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable equivalents."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        # Keep as string to avoid float rounding surprises.
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # Fallback: stringify unknown types (e.g., UUID).
    return str(value)


def _canonicalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonicalize a list of row dicts for stable comparison/serialization.

    - Convert all values to JSON-safe representations.
    - Sort row keys.
    - Sort rows by their JSON string representation (order-invariant).
    """
    safe_rows: List[Dict[str, Any]] = []
    for row in rows:
        safe_row = {str(k): _json_safe(v) for k, v in row.items()}
        # Normalize key ordering deterministically.
        safe_rows.append(dict(sorted(safe_row.items(), key=lambda kv: kv[0])))

    safe_rows.sort(key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))
    return safe_rows


def _execute_query(
    connector: "pg_utils.PostgresConnector", sql: str
) -> List[Dict[str, Any]]:
    """Execute a query and return canonicalized rows with column names."""
    rows: List[Dict[str, Any]] = connector.execute_and_fetchall_with_col_names(sql)
    return _canonicalize_rows(rows or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RAQUEL denotations for aligned and nullified DBs."
    )
    parser.add_argument(
        "--sql_queries",
        default="data/aligned_db/synthesized_queries.sql",
        help="Path to synthesized_queries.sql",
    )
    parser.add_argument(
        "--out",
        default="data/raquel/denotations/by_index.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db_id", default="tofu_data")
    parser.add_argument("--null_port", type=int, default=5433)
    parser.add_argument("--null_db_id", default="tofu_data_null")
    parser.add_argument("--user_id", default="postgres")
    parser.add_argument("--passwd", default="postgres")
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Optional limit for quick testing (0 = no limit).",
    )
    args = parser.parse_args()

    sql_queries: List[str] = load_sql_queries(args.sql_queries)
    if args.max_queries and args.max_queries > 0:
        sql_queries = sql_queries[: args.max_queries]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pg_aligned = pg_utils.PostgresConnector(
        db_id=args.db_id,
        user_id=args.user_id,
        passwd=args.passwd,
        host=args.host,
        port=args.port,
    )
    pg_null = pg_utils.PostgresConnector(
        db_id=args.null_db_id,
        user_id=args.user_id,
        passwd=args.passwd,
        host=args.host,
        port=args.null_port,
    )

    error_count = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for idx, sql in enumerate(tqdm.tqdm(sql_queries, desc="Exporting denotations")):
            record: Dict[str, Any] = {"query_index": idx, "sql": sql}
            try:
                record["result_aligned"] = _execute_query(pg_aligned, sql)
            except Exception as exc:
                error_count += 1
                record["result_aligned"] = []
                record["error_aligned"] = str(exc)
            try:
                record["result_null"] = _execute_query(pg_null, sql)
            except Exception as exc:
                error_count += 1
                record["result_null"] = []
                record["error_null"] = str(exc)

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote denotations for %d queries to %s (errors=%d)",
        len(sql_queries),
        args.out,
        error_count,
    )


if __name__ == "__main__":
    main()

