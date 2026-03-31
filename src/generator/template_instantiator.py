"""Instantiate TemplateSpec placeholders using witness tuples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional

from src.generator.template_spec import BindGroupSpec, PlaceholderSpec, TemplateSpec


@dataclass
class InstantiationResult:
    sql: str
    placeholder_values: Dict[str, str]
    selected_rows: Dict[str, Dict[str, object]]


class TemplateInstantiator:
    """Renders SQL queries from TemplateSpecs and witness tuples."""

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def instantiate(
        self,
        spec: TemplateSpec,
        witness_pool: Dict[str, List[Dict[str, object]]],
    ) -> Optional[InstantiationResult]:
        """Build one SQL statement using provided witness rows."""

        selections = self._select_rows(spec, witness_pool)
        if selections is None:
            return None

        placeholder_literals: Dict[str, str] = {}
        for placeholder in spec.placeholders.values():
            row = selections.get(placeholder.bind_group)
            if not row:
                return None
            literal = self._render_placeholder_value(
                placeholder,
                row=row,
                selections=selections,
                witness_pool=witness_pool,
            )
            if literal is None:
                return None
            placeholder_literals[placeholder.name] = literal

        sql = spec.sql_template
        for name, literal in placeholder_literals.items():
            sql = sql.replace(f"{{{name}}}", literal)

        return InstantiationResult(
            sql=sql,
            placeholder_values=placeholder_literals,
            selected_rows=selections,
        )

    # ------------------------------------------------------------------
    # Row selection helpers
    # ------------------------------------------------------------------
    def _select_rows(
        self,
        spec: TemplateSpec,
        witness_pool: Dict[str, List[Dict[str, object]]],
    ) -> Optional[Dict[str, Dict[str, object]]]:
        selections: Dict[str, Dict[str, object]] = {}
        for group_id, bind_group in spec.bind_groups.items():
            rows = list(witness_pool.get(group_id) or [])
            if not rows:
                return None
            selections[group_id] = self.random.choice(rows)

        # Enforce distinct anchors when requested
        for group_id, bind_group in spec.bind_groups.items():
            if not bind_group.distinct_from or not bind_group.anchor_key:
                continue
            disallowed_values = {
                selections[other_id].get(other_group.anchor_key)
                for other_id in bind_group.distinct_from
                if (other_group := spec.bind_groups.get(other_id))
                and other_group.anchor_key
                and selections.get(other_id)
            }
            if not disallowed_values:
                continue
            current_value = selections[group_id].get(bind_group.anchor_key)
            if current_value not in disallowed_values:
                continue
            candidate_rows = list(witness_pool.get(group_id) or [])
            self.random.shuffle(candidate_rows)
            replacement = next(
                (
                    row
                    for row in candidate_rows
                    if row.get(bind_group.anchor_key) not in disallowed_values
                ),
                None,
            )
            if replacement is None:
                return None
            selections[group_id] = replacement
        return selections

    # ------------------------------------------------------------------
    # Placeholder rendering
    # ------------------------------------------------------------------
    def _render_placeholder_value(
        self,
        placeholder: PlaceholderSpec,
        *,
        row: Dict[str, object],
        selections: Dict[str, Dict[str, object]],
        witness_pool: Dict[str, List[Dict[str, object]]],
    ) -> Optional[str]:
        operator = placeholder.operator_kind.lower()
        value = row.get(placeholder.source_column) if placeholder.source_column else None

        if operator in ("const", "raw", "sql_keyword", "limit", "join_type"):
            return self._format_constant_literal(placeholder.value_transform, raw=(operator != "const"))
        if operator == "equals":
            return self._format_literal(value)
        if operator in ("ilike_contains", "like_contains"):
            return self._format_like_literal(value, placeholder.value_transform)
        if operator == "between_around":
            return self._format_between_literal(value, placeholder.value_transform)
        if operator == "between_lower":
            bounds = self._compute_between_bounds(value, placeholder.value_transform)
            return bounds[0] if bounds else None
        if operator == "between_upper":
            bounds = self._compute_between_bounds(value, placeholder.value_transform)
            return bounds[1] if bounds else None
        if operator == "in_list":
            return self._format_in_list_literal(
                placeholder,
                selections,
                witness_pool,
            )
        if operator in ("threshold", "having_threshold"):
            return self._format_threshold_literal(value, placeholder.value_transform)
        if operator == "subquery_threshold":
            return self._format_threshold_literal(value, placeholder.value_transform)
        # Default fallback
        return self._format_literal(value)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _format_literal(self, value: object) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float, Decimal)):
            return str(value)
        if isinstance(value, (datetime, date)):
            return f"'{value.isoformat()}'"
        text = str(value)
        escaped = text.replace("'", "''")
        return f"'{escaped}'"

    def _format_like_literal(
        self,
        value: object,
        transform: Dict[str, object],
    ) -> Optional[str]:
        if value is None:
            return None
        substring_length = int(transform.get("substring_length", 4))
        text = str(value)
        if len(text) < substring_length:
            return self._format_literal(text)
        start = self.random.randint(0, max(0, len(text) - substring_length))
        sub = text[start : start + substring_length]
        sub = sub.replace("%", "")
        return self._format_literal(f"%{sub}%")

    def _format_between_literal(
        self,
        value: object,
        transform: Dict[str, object],
    ) -> Optional[str]:
        bounds = self._compute_between_bounds(value, transform)
        if not bounds:
            return None
        return f"{bounds[0]} AND {bounds[1]}"

    def _compute_between_bounds(
        self, value: object, transform: Dict[str, object]
    ) -> Optional[tuple[str, str]]:
        window = transform.get("range_window", 5)
        if isinstance(value, (int, float, Decimal)):
            low = value - window
            high = value + window
            return (str(low), str(high))
        if isinstance(value, str):
            try:
                numeric = float(value)
                low = numeric - window
                high = numeric + window
                return (str(low), str(high))
            except ValueError:
                pass
        if isinstance(value, (datetime, date)):
            delta_days = int(transform.get("days", 30))
            low = value - timedelta(days=delta_days)
            high = value + timedelta(days=delta_days)
            return (f"'{low.isoformat()}'", f"'{high.isoformat()}'")
        return None

    def _format_constant_literal(
        self, transform: Dict[str, object], raw: bool
    ) -> Optional[str]:
        if not transform:
            return None
        if "value" in transform:
            value = transform["value"]
        elif "allowed_values" in transform and transform["allowed_values"]:
            options = list(transform["allowed_values"])
            value = self.random.choice(options)
        elif "allowed" in transform and transform["allowed"]:
            options = list(transform["allowed"])
            value = self.random.choice(options)
        else:
            return None
        if raw:
            return str(value)
        return self._format_literal(value)

    def _format_in_list_literal(
        self,
        placeholder: PlaceholderSpec,
        selections: Dict[str, Dict[str, object]],
        witness_pool: Dict[str, List[Dict[str, object]]],
    ) -> Optional[str]:
        list_size = int(placeholder.value_transform.get("list_size", 3))
        bind_group_id = placeholder.bind_group
        rows = list(witness_pool.get(bind_group_id) or [])
        if not rows:
            rows = [selections[bind_group_id]]
        self.random.shuffle(rows)
        values: List[str] = []
        seen = set()
        for row in rows:
            literal = self._format_literal(row.get(placeholder.source_column))
            if literal in seen:
                continue
            values.append(literal)
            seen.add(literal)
            if len(values) >= list_size:
                break
        if not values:
            return None
        joined = ", ".join(values)
        return f"({joined})"

    def _format_threshold_literal(
        self,
        value: object,
        transform: Dict[str, object],
    ) -> Optional[str]:
        offset = transform.get("offset", 1)
        if isinstance(value, (int, float, Decimal)):
            adjusted = max(value - offset, 0)
            return str(adjusted)
        if isinstance(value, str):
            try:
                numeric = float(value)
                adjusted = max(numeric - offset, 0)
                return str(adjusted)
            except ValueError:
                return None
        return None
