"""Data model and validation for LLM-generated SQL template specifications.

Each template describes a query structure with placeholders that must be
instantiated using concrete database values drawn from *witness tuples*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SPEC_VERSION = "1.0"
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z0-9_]+)\}")


def normalize_sql_text(text: str) -> str:
    """Normalize SQL text coming from LLM/JSON.

    Some model outputs embed literal escape sequences like ``\\n`` inside JSON
    strings. When parsed, these remain as backslash+n characters and break SQL.
    We normalize them into actual whitespace/newlines.
    """

    if not text:
        return text
    # Convert literal escape sequences (two-character backslash forms) into whitespace.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    text = text.replace("\\t", "\t")
    return text


def _ensure_list(value: Optional[Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class PlaceholderSpec:
    """Metadata describing how to fill a placeholder literal."""

    name: str
    source_column: str
    operator_kind: str
    bind_group: str
    value_transform: Dict[str, Any] = field(default_factory=dict)
    requires_same_row_with: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaceholderSpec":
        # Be tolerant to missing optional fields; enforce in TemplateSpec.validate().
        name = data.get("name") or data.get("placeholder") or ""
        source_column = (
            data.get("source_column")
            or data.get("column")
            or data.get("column_ref")
            or data.get("source")
            or ""
        )
        operator_kind = (
            data.get("operator_kind")
            or data.get("operator")
            or data.get("op")
            or data.get("kind")
            or ""
        )
        bind_group = (
            data.get("bind_group") or data.get("group") or data.get("bind") or ""
        )
        return cls(
            name=name,
            source_column=source_column,
            operator_kind=operator_kind,
            bind_group=bind_group,
            value_transform=data.get("value_transform") or data.get("transform") or {},
            requires_same_row_with=list(data.get("requires_same_row_with") or []),
            metadata=data.get("metadata") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_column": self.source_column,
            "operator_kind": self.operator_kind,
            "bind_group": self.bind_group,
            "value_transform": self.value_transform,
            "requires_same_row_with": self.requires_same_row_with,
            "metadata": self.metadata,
        }


@dataclass
class BindGroupSpec:
    """Defines how to sample witness tuples for a set of placeholders."""

    group_id: str
    from_join_sql: str
    required_columns: List[str]
    anchor_key: Optional[str] = None
    distinct_from: List[str] = field(default_factory=list)
    row_count_hint: int = 200
    filters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindGroupSpec":
        group_id = data.get("group_id") or data.get("name") or data.get("group") or ""
        from_join_sql = data.get("from_join_sql") or data.get("from") or ""
        if isinstance(from_join_sql, str):
            from_join_sql = normalize_sql_text(from_join_sql)
        return cls(
            group_id=group_id,
            from_join_sql=from_join_sql,
            required_columns=list(data.get("required_columns") or []),
            anchor_key=data.get("anchor_key"),
            distinct_from=list(data.get("distinct_from") or []),
            row_count_hint=int(data.get("row_count_hint") or 200),
            filters=list(data.get("filters") or []),
            metadata=data.get("metadata") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "from_join_sql": self.from_join_sql,
            "required_columns": self.required_columns,
            "anchor_key": self.anchor_key,
            "distinct_from": self.distinct_from,
            "row_count_hint": self.row_count_hint,
            "filters": self.filters,
            "metadata": self.metadata,
        }


@dataclass
class TemplateConstraints:
    """Optional guard rails for instantiated queries."""

    limit: Optional[int] = None
    disallow_destructive: bool = True
    max_predicates: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TemplateConstraints":
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            limit=data.get("limit"),
            disallow_destructive=bool(data.get("disallow_destructive", True)),
            max_predicates=data.get("max_predicates"),
            metadata=data.get("metadata") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "disallow_destructive": self.disallow_destructive,
            "max_predicates": self.max_predicates,
            "metadata": self.metadata,
        }


@dataclass
class TemplateSpec:
    """Full template specification generated by the LLM."""

    type_name: str
    description: str
    sql_template: str
    placeholders: Dict[str, PlaceholderSpec]
    bind_groups: Dict[str, BindGroupSpec]
    constraints: TemplateConstraints = field(default_factory=TemplateConstraints)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = SPEC_VERSION

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateSpec":
        raw_placeholders = _normalize_placeholder_map(data.get("placeholders"))
        placeholders: Dict[str, PlaceholderSpec] = {}
        for name, value in raw_placeholders.items():
            spec_data = value if isinstance(value, dict) else {}
            spec_data = {**spec_data, "name": spec_data.get("name") or name}
            placeholders[name] = PlaceholderSpec.from_dict(spec_data)

        raw_groups = _normalize_placeholder_map(data.get("bind_groups"))
        bind_groups: Dict[str, BindGroupSpec] = {}
        for group_id, value in raw_groups.items():
            group_data = value if isinstance(value, dict) else {}
            group_data = {
                **group_data,
                "group_id": group_data.get("group_id") or group_id,
            }
            bind_groups[group_id] = BindGroupSpec.from_dict(group_data)
        return cls(
            type_name=data["type_name"],
            description=data.get("description", ""),
            sql_template=normalize_sql_text(data["sql_template"]),
            placeholders=placeholders,
            bind_groups=bind_groups,
            constraints=TemplateConstraints.from_dict(data.get("constraints")),
            metadata=data.get("metadata") or {},
            version=data.get("version", SPEC_VERSION),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "type_name": self.type_name,
            "description": self.description,
            "sql_template": self.sql_template,
            "placeholders": {
                name: spec.to_dict() for name, spec in self.placeholders.items()
            },
            "bind_groups": {
                group_id: spec.to_dict() for group_id, spec in self.bind_groups.items()
            },
            "constraints": self.constraints.to_dict(),
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------
    def validate(
        self,
        available_tables: Optional[Iterable[str]] = None,
        available_columns: Optional[Dict[str, Sequence[str]]] = None,
    ) -> List[str]:
        """Return a list of human-readable validation errors."""

        errors: List[str] = []

        if not self.type_name:
            errors.append("type_name must be provided")

        if not self.sql_template.strip().lower().startswith("select"):
            errors.append("sql_template must start with SELECT")

        template_placeholders = set(PLACEHOLDER_PATTERN.findall(self.sql_template))
        missing_in_template = set(self.placeholders) - template_placeholders
        if missing_in_template:
            errors.append(
                f"placeholders {sorted(missing_in_template)} not found in sql_template"
            )

        unused_placeholders = template_placeholders - set(self.placeholders)
        if unused_placeholders:
            errors.append(
                f"sql_template references placeholders without specs: "
                f"{sorted(unused_placeholders)}"
            )

        no_source_required = {"const", "raw", "sql_keyword", "limit", "join_type"}
        for name, placeholder in self.placeholders.items():
            if (
                not placeholder.source_column
                and placeholder.operator_kind.lower() not in no_source_required
            ):
                errors.append(f"placeholder '{name}' missing source_column")
            if placeholder.bind_group not in self.bind_groups:
                errors.append(
                    f"placeholder '{name}' refers to unknown bind group "
                    f"'{placeholder.bind_group}'"
                )
            if available_columns and "." in placeholder.source_column:
                table, column = placeholder.source_column.split(".", 1)
                normalized = table.strip('"').lower()
                if normalized in _normalize_table_map(available_columns):
                    columns = {
                        c.lower()
                        for c in _normalize_table_map(available_columns)[normalized]
                    }
                    if column.strip('"').lower() not in columns:
                        errors.append(
                            f"placeholder '{name}' references unknown column "
                            f"{placeholder.source_column}"
                        )

        for group_id, group in self.bind_groups.items():
            if not group.from_join_sql.strip().lower().startswith("from"):
                errors.append(
                    f"bind group '{group_id}' must supply a FROM/JOIN clause "
                    f"starting with FROM"
                )
            if not group.required_columns:
                errors.append(
                    f"bind group '{group_id}' must list at least one required column"
                )
            # Check for invalid column names (numbers, empty strings)
            for col in group.required_columns:
                col_stripped = col.strip()
                # Column names should not be pure numbers or empty
                if not col_stripped or col_stripped.isdigit():
                    errors.append(
                        f"bind group '{group_id}' has invalid required_column '{col}' "
                        f"(must be a real column name like 'p.name' or 'w.title')"
                    )
                # Check that column aliases are defined in from_join_sql
                elif "." in col_stripped:
                    alias = col_stripped.split(".")[0].strip()
                    from_sql_lower = group.from_join_sql.lower()
                    # Check if alias appears as a table alias (word boundary match)
                    alias_pattern = rf'\b{re.escape(alias.lower())}\b'
                    if not re.search(alias_pattern, from_sql_lower):
                        errors.append(
                            f"bind group '{group_id}' required_column '{col}' references "
                            f"alias '{alias}' which is not defined in from_join_sql"
                        )
            # Check for unfilled placeholders in from_join_sql
            unfilled = extract_placeholders(group.from_join_sql)
            if unfilled:
                errors.append(
                    f"bind group '{group_id}' has unfilled placeholders in "
                    f"from_join_sql: {unfilled}"
                )

        return errors

    def ensure_valid_or_raise(self, **kwargs: Any) -> None:
        errors = self.validate(**kwargs)
        if errors:
            raise SpecValidationError(errors)


class SpecValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# ----------------------------------------------------------------------
# Helpers for JSON persistence
# ----------------------------------------------------------------------
def _normalize_placeholder_map(
    raw: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {
            key: value if isinstance(value, dict) else {"value": value}
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for entry in raw:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError("List placeholder entries must include 'name'")
            normalized[entry["name"]] = entry
        return normalized
    raise ValueError("placeholders/bind_groups must be dict or list")


def _normalize_table_map(
    table_map: Dict[str, Sequence[str]],
) -> Dict[str, Sequence[str]]:
    return {table.lower().strip('"'): columns for table, columns in table_map.items()}


def extract_placeholders(sql_template: str) -> List[str]:
    """Return placeholder names embedded in the SQL template."""

    return PLACEHOLDER_PATTERN.findall(sql_template or "")


def load_template_specs(path: Path) -> List[TemplateSpec]:
    specs: List[TemplateSpec] = []
    if not path.exists():
        return specs

    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            specs.append(TemplateSpec.from_dict(data))
    return specs


def append_template_spec(path: Path, spec: TemplateSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fp:
        fp.write(json.dumps(spec.to_dict(), ensure_ascii=False))
        fp.write("\n")


def load_template_spec_file(path: Path) -> TemplateSpec:
    with path.open("r") as fp:
        return TemplateSpec.from_dict(json.load(fp))


def save_template_spec_file(path: Path, spec: TemplateSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(spec.to_dict(), fp, ensure_ascii=False, indent=2)
