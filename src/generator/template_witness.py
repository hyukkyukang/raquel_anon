"""Witness sampler that fetches candidate rows for TemplateSpec bind groups."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hkkang_utils.pg as pg_utils

from src.generator.template_spec import BindGroupSpec, PlaceholderSpec, TemplateSpec

logger = logging.getLogger("src.generator.template_witness")


class TemplateWitnessSampler:
    """Samples witness rows for bind groups to guarantee satisfiable predicates."""

    def __init__(
        self,
        pg_client: pg_utils.PostgresConnector,
        debug_dir: Optional[Path] = None,
    ):
        self.pg_client = pg_client
        self.debug_dir = debug_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_bind_group(
        self,
        bind_group: BindGroupSpec,
        placeholder_specs: Iterable[PlaceholderSpec],
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Return witness tuples for one bind group."""

        max_rows = max_candidates or bind_group.row_count_hint or 200
        columns = self._collect_columns(bind_group, placeholder_specs)
        sql = self._build_witness_sql(bind_group, columns, max_rows)
        logger.debug("Witness SQL for %s:\n%s", bind_group.group_id, sql)

        try:
            rows = self.pg_client.execute_and_fetchall_with_col_names(sql)
        except Exception as exc:
            logger.debug("Witness query failed for %s: %s", bind_group.group_id, exc)
            self._record_failure(bind_group, sql, exc)
            return []
        return [self._remap_row(row, columns) for row in rows]

    def sample_all_groups(
        self,
        spec: TemplateSpec,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Sample witnesses for every bind group in the spec."""

        result: Dict[str, List[Dict[str, object]]] = {}
        for group_id, bind_group in spec.bind_groups.items():
            placeholders = [
                ph for ph in spec.placeholders.values() if ph.bind_group == group_id
            ]
            result[group_id] = self.sample_bind_group(
                bind_group=bind_group,
                placeholder_specs=placeholders,
                max_candidates=max_candidates,
            )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_columns(
        bind_group: BindGroupSpec,
        placeholder_specs: Iterable[PlaceholderSpec],
    ) -> List[str]:
        columns = list(dict.fromkeys(bind_group.required_columns))
        for placeholder in placeholder_specs:
            if placeholder.source_column and placeholder.source_column not in columns:
                columns.append(placeholder.source_column)
        if bind_group.anchor_key and bind_group.anchor_key not in columns:
            columns.append(bind_group.anchor_key)
        return columns

    def _build_witness_sql(
        self,
        bind_group: BindGroupSpec,
        columns: List[str],
        limit: int,
    ) -> str:
        projections = []
        for column in columns:
            alias = self._column_alias(column)
            projections.append(f'{column} AS "{alias}"')

        select_clause = ",\n       ".join(projections)
        base_parts = [
            "SELECT DISTINCT",
            f"       {select_clause}",
            bind_group.from_join_sql.strip(),
        ]

        where_clauses = []
        for column in columns:
            where_clauses.append(f"{column} IS NOT NULL")
        for filter_expr in bind_group.filters:
            where_clauses.append(f"({filter_expr})")

        if where_clauses:
            base_parts.append("WHERE " + " AND ".join(where_clauses))

        # Postgres does not allow ORDER BY expressions not present in the SELECT
        # list when using SELECT DISTINCT. Wrap the DISTINCT query in a subquery
        # so we can ORDER BY RANDOM() safely.
        base_sql = "\n".join(base_parts)
        return (
            "SELECT *\n"
            "FROM (\n"
            f"{base_sql}\n"
            ") AS witness\n"
            "ORDER BY RANDOM()\n"
            f"LIMIT {max(limit, 1)};"
        )

    @staticmethod
    def _column_alias(column: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", column)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "col"
        return sanitized.lower()

    @staticmethod
    def _remap_row(row: Dict[str, object], columns: List[str]) -> Dict[str, object]:
        remapped: Dict[str, object] = {}
        for column in columns:
            alias = TemplateWitnessSampler._column_alias(column)
            remapped[column] = row.get(alias)
        return remapped

    def _record_failure(
        self, bind_group: BindGroupSpec, sql: str, error: Exception
    ) -> None:
        if not self.debug_dir:
            return
        try:
            failure_dir = self.debug_dir / "witness_queries_failed"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_file = failure_dir / f"{bind_group.group_id}.sql"
            with failure_file.open("a") as fp:
                fp.write(f"-- ERROR: {error}\n{sql}\n\n")
        except Exception:
            logger.exception(
                "Failed to record witness failure for group %s", bind_group.group_id
            )
