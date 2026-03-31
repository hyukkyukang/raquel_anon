"""Quantify retain-integrity under nullification (diagnostic).

This script measures how much the *retain* portion of the aligned DB is disrupted
after building the nullified DB. Concretely, it:

- loads retain QA extractions from `data/aligned_db/qa_extractions.json`,
- uses the schema registry to pick a stable unique key per entity table,
- compares aligned vs null DB rows keyed by that unique value,
- reports how many retain QA pairs reference at least one entity that is
  missing/changed in the null DB.

Outputs:
- JSON report: `reports/paper/retain_integrity_report.json`
- Short markdown summary: `reports/paper/retain_integrity_report.md`
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import hkkang_utils.pg as pg_utils

from src.utils.logging import get_logger

logger = get_logger("script.analysis.retain_integrity_report", __file__)


def _quote_ident(identifier: str) -> str:
    """Quote a SQL identifier (table/column) safely for Postgres."""
    return '"' + identifier.replace('"', '""') + '"'


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
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@dataclass(frozen=True)
class TableKeySpec:
    """How to key rows for a table (unique column + PK columns to ignore in diffs)."""

    table: str
    key_column: str
    pk_columns: Tuple[str, ...]


def _load_table_key_specs(schema_registry_path: str) -> Dict[str, TableKeySpec]:
    """Infer a single unique key column per table from schema_registry.json."""
    payload: Any = _read_json(schema_registry_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {schema_registry_path}")

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Missing 'tables' in schema registry JSON.")

    specs: Dict[str, TableKeySpec] = {}
    for table_name, table_payload in tables.items():
        if not isinstance(table_payload, dict):
            continue
        cols = table_payload.get("columns")
        if not isinstance(cols, list):
            continue

        unique_cols: List[str] = []
        pk_cols: List[str] = []
        for col in cols:
            if not isinstance(col, dict):
                continue
            name = str(col.get("name", "")).strip()
            if not name:
                continue
            if bool(col.get("is_primary_key")):
                pk_cols.append(name)
            if bool(col.get("is_unique")):
                unique_cols.append(name)

        # We only key "entity tables" (those with a stable unique column).
        if not unique_cols:
            continue

        specs[table_name] = TableKeySpec(
            table=table_name,
            key_column=unique_cols[0],
            pk_columns=tuple(pk_cols),
        )

    return specs


def _iter_retain_extractions(qa_extractions_path: str) -> Iterable[Mapping[str, Any]]:
    """Yield per-QA extraction dicts for the retain split only."""
    payload: Any = _read_json(qa_extractions_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {qa_extractions_path}")

    raw = payload.get("extractions")
    if not isinstance(raw, dict):
        raise ValueError("Missing 'extractions' in qa_extractions.json")

    for item in raw.values():
        if not isinstance(item, dict):
            continue
        if str(item.get("source", "")).strip().lower() != "retain":
            continue
        yield item


def _truncate(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def _fetch_all_rows(
    connector: "pg_utils.PostgresConnector", table: str
) -> List[Dict[str, Any]]:
    """Fetch all rows from a table as dictionaries (column -> value)."""
    sql = f"SELECT * FROM {_quote_ident(table)}"
    rows: List[Dict[str, Any]] = connector.execute_and_fetchall_with_col_names(sql) or []
    return rows


def _row_diff_columns(
    aligned_row: Mapping[str, Any],
    null_row: Mapping[str, Any],
    *,
    ignore_columns: Set[str],
) -> List[str]:
    """Return list of columns whose values differ (aligned vs null)."""
    changed: List[str] = []
    for col in aligned_row.keys():
        if col in ignore_columns:
            continue
        if _json_safe(aligned_row.get(col)) != _json_safe(null_row.get(col)):
            changed.append(col)
    return changed


def _compute_table_diffs(
    *,
    table_spec: TableKeySpec,
    aligned_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-entity diffs for a table keyed by the unique key column value."""
    key_col: str = table_spec.key_column
    ignore_cols: Set[str] = set(table_spec.pk_columns)

    aligned_map: Dict[str, Dict[str, Any]] = {}
    for row in aligned_rows:
        key_val = row.get(key_col)
        if key_val is None:
            continue
        key_str = str(key_val).strip()
        if not key_str:
            continue
        # In the very unlikely case of duplicates, keep the first occurrence.
        aligned_map.setdefault(key_str, dict(row))

    null_map: Dict[str, Dict[str, Any]] = {}
    for row in null_rows:
        key_val = row.get(key_col)
        if key_val is None:
            continue
        key_str = str(key_val).strip()
        if not key_str:
            continue
        null_map.setdefault(key_str, dict(row))

    diffs: Dict[str, Dict[str, Any]] = {}
    for key_str, aligned_row in aligned_map.items():
        null_row = null_map.get(key_str)
        if null_row is None:
            diffs[key_str] = {
                "status": "missing_in_null",
                "changed_columns": [],
            }
            continue

        changed_cols = _row_diff_columns(
            aligned_row,
            null_row,
            ignore_columns=ignore_cols,
        )
        if changed_cols:
            diffs[key_str] = {
                "status": "changed",
                "changed_columns": changed_cols,
                "diff": {
                    col: {
                        "aligned": _json_safe(aligned_row.get(col)),
                        "null": _json_safe(null_row.get(col)),
                    }
                    for col in changed_cols
                },
            }

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retain-integrity report for nullification.")
    parser.add_argument(
        "--qa_extractions",
        default="data/aligned_db/qa_extractions.json",
        help="Path to qa_extractions.json",
    )
    parser.add_argument(
        "--schema_registry",
        default="data/aligned_db/schema_registry.json",
        help="Path to schema_registry.json",
    )
    parser.add_argument(
        "--removed_entities",
        default="data/aligned_db/log/nullify/removed_entities.json",
        help="Path to removed_entities.json (optional cross-check).",
    )
    parser.add_argument(
        "--out",
        default="reports/paper/retain_integrity_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out_md",
        default="reports/paper/retain_integrity_report.md",
        help="Output markdown summary path.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db_id", default="tofu_data")
    parser.add_argument("--null_port", type=int, default=5433)
    parser.add_argument("--null_db_id", default="tofu_data_null")
    parser.add_argument("--user_id", default="postgres")
    parser.add_argument("--passwd", default="postgres")
    parser.add_argument(
        "--max_qas",
        type=int,
        default=0,
        help="Optional limit for quick testing (0 = no limit).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10,
        help="How many impacted retain QAs to include as examples.",
    )
    args = parser.parse_args()

    table_specs: Dict[str, TableKeySpec] = _load_table_key_specs(str(args.schema_registry))
    if not table_specs:
        raise ValueError("No tables with unique key columns found in schema registry.")

    removed_lookup: Set[Tuple[str, str]] = set()
    try:
        removed_payload: Any = _read_json(str(args.removed_entities))
        if isinstance(removed_payload, list):
            for item in removed_payload:
                if not isinstance(item, dict):
                    continue
                table = str(item.get("table", "")).strip()
                value = str(item.get("value", "")).strip()
                if table and value:
                    removed_lookup.add((table, value))
    except Exception:
        # Keep report robust if file is missing/invalid.
        removed_lookup = set()

    # Collect retain QA rows and their referenced entities (table + unique key value).
    retain_rows: List[Dict[str, Any]] = []
    referenced_entities: Dict[str, Set[str]] = {}  # table -> set(key_value)

    for ex in _iter_retain_extractions(str(args.qa_extractions)):
        try:
            qa_index: int = int(ex.get("qa_index"))
        except Exception:
            continue

        question: str = str(ex.get("question", "")).strip()
        answer: str = str(ex.get("answer", "")).strip()
        entities_raw: Any = ex.get("entities")
        entities: Dict[str, List[Dict[str, Any]]] = (
            entities_raw if isinstance(entities_raw, dict) else {}
        )

        qa_entities: List[Dict[str, str]] = []
        for table_name, rows in entities.items():
            if table_name not in table_specs:
                continue
            if not isinstance(rows, list):
                continue
            key_col: str = table_specs[table_name].key_column
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key_val = row.get(key_col)
                if key_val is None:
                    continue
                key_str = str(key_val).strip()
                if not key_str:
                    continue
                qa_entities.append(
                    {"table": table_name, "key_column": key_col, "key_value": key_str}
                )
                referenced_entities.setdefault(table_name, set()).add(key_str)

        retain_rows.append(
            {
                "qa_index": qa_index,
                "question": question,
                "answer": answer,
                "entities": qa_entities,
            }
        )

        if args.max_qas and args.max_qas > 0 and len(retain_rows) >= int(args.max_qas):
            break

    retain_rows = sorted(retain_rows, key=lambda r: int(r["qa_index"]))
    total_retain_qas: int = len(retain_rows)
    logger.info("Loaded %d retain QA extractions", total_retain_qas)

    # Connect to DBs.
    pg_aligned = pg_utils.PostgresConnector(
        db_id=str(args.db_id),
        user_id=str(args.user_id),
        passwd=str(args.passwd),
        host=str(args.host),
        port=int(args.port),
    )
    pg_null = pg_utils.PostgresConnector(
        db_id=str(args.null_db_id),
        user_id=str(args.user_id),
        passwd=str(args.passwd),
        host=str(args.host),
        port=int(args.null_port),
    )

    # Compute per-table diffs keyed by the unique column.
    diffs_by_table: Dict[str, Dict[str, Any]] = {}
    for table_name, spec in table_specs.items():
        if table_name not in referenced_entities:
            continue  # not referenced by retain QA extractions

        aligned_rows: List[Dict[str, Any]] = _fetch_all_rows(pg_aligned, table_name)
        null_rows: List[Dict[str, Any]] = _fetch_all_rows(pg_null, table_name)
        diffs_by_table[table_name] = _compute_table_diffs(
            table_spec=spec,
            aligned_rows=aligned_rows,
            null_rows=null_rows,
        )

    # Build per-QA impact list.
    impacted_qas: List[Dict[str, Any]] = []
    missing_in_null_entities = 0
    changed_entities = 0
    impacted_by_table: Dict[str, int] = {}

    for row in retain_rows:
        qa_entities = row["entities"]
        impacted_entities: List[Dict[str, Any]] = []
        for ent in qa_entities:
            table = str(ent["table"])
            key_col = str(ent["key_column"])
            key_val = str(ent["key_value"])
            diff = diffs_by_table.get(table, {}).get(key_val)
            if not diff:
                continue
            status = str(diff.get("status", ""))
            if status not in {"missing_in_null", "changed"}:
                continue

            impacted_entities.append(
                {
                    "table": table,
                    "key_column": key_col,
                    "key_value": key_val,
                    "status": status,
                    "changed_columns": diff.get("changed_columns", []),
                    "diff": diff.get("diff", {}),
                    "is_listed_removed_entity": (table, key_val) in removed_lookup,
                }
            )

            impacted_by_table[table] = impacted_by_table.get(table, 0) + 1
            if status == "missing_in_null":
                missing_in_null_entities += 1
            elif status == "changed":
                changed_entities += 1

        if impacted_entities:
            impacted_qas.append(
                {
                    "qa_index": row["qa_index"],
                    "question": _truncate(str(row["question"]), 300),
                    "answer": _truncate(str(row["answer"]), 400),
                    "impacted_entities": impacted_entities,
                }
            )

    impacted_qas = sorted(
        impacted_qas, key=lambda r: len(r.get("impacted_entities", [])), reverse=True
    )

    impacted_count: int = len(impacted_qas)
    impacted_rate: float = float(impacted_count / total_retain_qas) if total_retain_qas else 0.0

    referenced_entity_count: int = sum(len(v) for v in referenced_entities.values())
    report: Dict[str, Any] = {
        "summary": {
            "total_retain_qas": total_retain_qas,
            "impacted_retain_qas": impacted_count,
            "impacted_rate": impacted_rate,
            "referenced_entities_total": referenced_entity_count,
            "impacted_entities_missing_in_null": int(missing_in_null_entities),
            "impacted_entities_changed": int(changed_entities),
        },
        "by_table": dict(sorted(impacted_by_table.items(), key=lambda kv: (-kv[1], kv[0]))),
        "examples": impacted_qas[: max(0, int(args.max_examples))],
        "metadata": {
            "qa_extractions": str(args.qa_extractions),
            "schema_registry": str(args.schema_registry),
            "removed_entities": str(args.removed_entities),
            "db": {"host": str(args.host), "port": int(args.port), "db_id": str(args.db_id)},
            "null_db": {
                "host": str(args.host),
                "port": int(args.null_port),
                "db_id": str(args.null_db_id),
            },
            "tables_compared": sorted(diffs_by_table.keys()),
        },
    }

    _write_json(str(args.out), report)

    md_lines: List[str] = []
    md_lines.append("## Retain integrity report (nullification diagnostic)\n")
    md_lines.append(f"- **retain QA pairs (total)**: {total_retain_qas}")
    md_lines.append(f"- **impacted retain QA pairs**: {impacted_count} ({impacted_rate:.2%})")
    md_lines.append(
        f"- **impacted entities**: missing_in_null={missing_in_null_entities}, changed={changed_entities}"
    )
    if impacted_by_table:
        top_tables = list(dict(sorted(impacted_by_table.items(), key=lambda kv: -kv[1])).items())[:5]
        md_lines.append(
            "- **top affected tables**: "
            + ", ".join(f"`{t}` ({c})" for t, c in top_tables)
        )
    if impacted_qas:
        md_lines.append("- **example impacted retain QA indices**: " + ", ".join(str(r["qa_index"]) for r in impacted_qas[: min(10, len(impacted_qas))]))
    md_lines.append("")
    md_lines.append(f"JSON output: `{str(args.out)}`")
    md_lines.append(f"Generated at: {datetime.datetime.now().isoformat(timespec='seconds')}")
    md_lines.append("")

    _write_text(str(args.out_md), "\n".join(md_lines))

    logger.info("Wrote %s", str(args.out))
    logger.info("Wrote %s", str(args.out_md))


if __name__ == "__main__":
    main()

