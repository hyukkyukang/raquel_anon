"""LLM-backed generator for TemplateSpec objects with caching and validation."""

from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omegaconf import DictConfig

from src.generator.template_spec import (
    TemplateSpec,
    append_template_spec,
    extract_placeholders,
    load_template_specs,
    normalize_sql_text,
)
from src.llm.json_repair import JSONRepairer
from src.prompt.sql_synthesis.template_spec_generation import (
    TemplateSpecGenerationPrompt,
)
from src.utils.llm import LLMGeneratorMixin

logger = logging.getLogger("src.generator.template_spec_generator")


class TemplateSpecGenerator(LLMGeneratorMixin):
    """Generates TemplateSpec objects per query type with disk caching."""

    def __init__(
        self,
        api_cfg: DictConfig,
        global_cfg: DictConfig,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(api_cfg, global_cfg)
        self.cfg = api_cfg
        self.global_cfg = global_cfg

        project_path = Path(global_cfg.project_path)
        data_dir = Path(global_cfg.paths.data_dir)
        cache_subdir = cache_dir or getattr(
            global_cfg.paths, "template_specs_dir", "aligned_db/template_specs"
        )
        self.cache_dir = project_path / data_dir / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir = self.cache_dir / "failed"
        self.failure_dir.mkdir(parents=True, exist_ok=True)

        self._json_repairer = JSONRepairer(api_cfg, global_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_cached_specs(self, type_name: str) -> List[TemplateSpec]:
        """Load cached specs for a type if available."""

        cache_file = self._cache_file(type_name)
        if not cache_file.exists():
            return []
        try:
            return load_template_specs(cache_file)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load cached specs for {type_name}: {exc}")
            return []

    def ensure_specs(
        self,
        *,
        type_name: str,
        type_description: str,
        schema_text: str,
        value_hints: str,
        target_count: int,
        overwrite: bool = False,
        recent_failures: Optional[List[str]] = None,
        additional_guidance: Optional[str] = None,
    ) -> List[TemplateSpec]:
        """Return at least `target_count` specs for the type, generating as needed."""

        cache_file = self._cache_file(type_name)
        specs: List[TemplateSpec]
        if overwrite and cache_file.exists():
            cache_file.unlink()
            specs = []
        else:
            specs = self.load_cached_specs(type_name)

        seen_templates = {spec.sql_template.strip(): spec for spec in specs}

        if len(seen_templates) >= target_count:
            logger.debug(
                "[%s] Already have %d/%d templates cached",
                type_name,
                len(seen_templates),
                target_count,
            )
            return list(seen_templates.values())[:target_count]

        logger.debug(
            "[%s] Generating templates: have %d, need %d",
            type_name,
            len(seen_templates),
            target_count,
        )

        attempts = 0
        # Cap attempts at 5x target to avoid infinite loops on duplicate-heavy types
        max_attempts = max(20, target_count * 5)
        while len(seen_templates) < target_count and attempts < max_attempts:
            attempts += 1
            prompt = TemplateSpecGenerationPrompt(
                type_name=type_name,
                type_description=type_description,
                schema_text=schema_text,
                value_hints=value_hints,
                recent_failures=recent_failures,
                additional_guidance=additional_guidance,
            )

            response = self.llm_api_caller(
                prompt,
                prefix=f"template_spec_{type_name}",
            )

            spec = self._parse_response(response, type_name)
            if spec is None:
                self._record_failure(type_name, response, "parse_error")
                continue

            try:
                spec.ensure_valid_or_raise()
            except Exception as exc:
                recent_failures = (recent_failures or []) + [str(exc)]
                self._record_failure(type_name, response, f"validation_error:{exc}")
                continue

            key = spec.sql_template.strip()
            if key in seen_templates:
                continue

            seen_templates[key] = spec
            append_template_spec(cache_file, spec)
            logger.info(
                "Cached template spec %s (%d/%d)",
                spec.description or spec.type_name,
                len(seen_templates),
                target_count,
            )

        # Only log warning if we actually made attempts (not just using cache)
        if len(seen_templates) < target_count and attempts > 0:
            logger.warning(
                "[%s] Only generated %d/%d template specs after %d attempts",
                type_name,
                len(seen_templates),
                target_count,
                attempts,
            )
        return list(seen_templates.values())[:target_count]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cache_file(self, type_name: str) -> Path:
        safe_name = type_name.replace("/", "_")
        return self.cache_dir / f"{safe_name}.jsonl"

    def _parse_response(self, response: str, type_name: str) -> Optional[TemplateSpec]:
        """Parse raw LLM response into a TemplateSpec, optionally repairing JSON."""

        if not response:
            logger.warning("Empty response from template spec LLM")
            self._record_failure(type_name, response, "empty_response")
            return None

        raw = self._extract_json_block(response)
        try:
            data = json.loads(raw)
        except JSONDecodeError as exc:
            repaired, was_repaired = self._json_repairer.repair_and_parse(raw, exc)
            if repaired is None:
                logger.warning("Failed to repair JSON template: %s", exc)
                self._record_failure(type_name, raw, f"json_decode_error:{exc}")
                return None
            data = repaired
            if was_repaired:
                logger.debug("JSON template repaired automatically")

        # Some models return a JSON *string* that itself contains JSON.
        if isinstance(data, str):
            inner = data.strip()
            if inner.startswith("{") or inner.startswith("["):
                try:
                    data = json.loads(inner)
                except JSONDecodeError:
                    self._record_failure(type_name, inner, "double_encoded_json")
                    return None

        if not isinstance(data, dict):
            self._record_failure(
                type_name, json.dumps(data, ensure_ascii=False), "non_object_json"
            )
            return None

        coerced = self._coerce_template_dict(data, requested_type_name=type_name)
        try:
            spec = TemplateSpec.from_dict(coerced)
            return spec
        except Exception as exc:
            logger.warning("Failed to deserialize TemplateSpec: %s", exc)
            self._record_failure(
                type_name, json.dumps(coerced, ensure_ascii=False), f"deserialize:{exc}"
            )
            return None

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Remove Markdown fences around JSON, if present."""

        text = text.strip()
        if not text.startswith("```"):
            return text
        parts = text.split("```")
        if len(parts) < 3:
            return text
        candidate = parts[1 if parts[0] == "" else 2].strip()
        # Strip optional language tag line like "json"
        first_line, *rest = candidate.splitlines()
        if first_line.strip().lower() in {"json", "javascript"} and rest:
            return "\n".join(rest).strip()
        return candidate

    def _coerce_template_dict(
        self, data: Dict[str, object], requested_type_name: str
    ) -> Dict[str, object]:
        """Coerce common alternate LLM formats into our TemplateSpec schema."""

        # If already in expected shape, just return.
        if {
            "type_name",
            "sql_template",
            "placeholders",
            "bind_groups",
        }.issubset(set(data.keys())):
            return data

        # Alternate shape we commonly see: template_type + sql_skeleton.
        sql_template = (
            str(data.get("sql_template") or data.get("sql_skeleton") or "")
        ).strip()
        sql_template = normalize_sql_text(sql_template)
        if not sql_template:
            return {
                "type_name": requested_type_name,
                "description": str(data.get("description") or ""),
                "sql_template": "",
                "placeholders": {},
                "bind_groups": {},
            }

        # Fill simple non-data placeholders (LIMIT, join type) to avoid leaking into bind groups.
        placeholder_meta = data.get("placeholders")
        placeholder_meta_map: Dict[str, Dict[str, object]] = {}
        if isinstance(placeholder_meta, dict):
            for k, v in placeholder_meta.items():
                if isinstance(v, dict):
                    placeholder_meta_map[str(k)] = v

        def meta_allowed(name: str) -> List[str]:
            meta = placeholder_meta_map.get(name) or {}
            allowed = meta.get("allowed_values") or meta.get("allowed") or []
            if isinstance(allowed, list):
                return [str(x) for x in allowed if str(x)]
            return []

        # Replace LIMIT {Px} with constant.
        sql_template = re.sub(
            r"\bLIMIT\s+\{[A-Za-z0-9_]+\}\s*;?",
            "LIMIT 50",
            sql_template,
            flags=re.IGNORECASE,
        )

        # Replace join-type placeholder "{P} JOIN" with INNER JOIN.
        join_type_matches = re.findall(
            r"\{([A-Za-z0-9_]+)\}\s+JOIN\b", sql_template, flags=re.IGNORECASE
        )
        for name in join_type_matches:
            allowed = meta_allowed(name)
            replacement = allowed[0] if allowed else "INNER"
            sql_template = re.sub(
                rf"\{{{re.escape(name)}\}}", replacement, sql_template
            )

        # Replace ORDER BY placeholders with safe defaults to avoid invalid constructs
        # like "ORDER BY 1 1" or "ORDER BY w.col 1".
        sql_template = re.sub(
            r"\bORDER\s+BY\s+\{[A-Za-z0-9_]+\}\s+\{[A-Za-z0-9_]+\}",
            "ORDER BY 1",
            sql_template,
            flags=re.IGNORECASE,
        )
        sql_template = re.sub(
            r"\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_\.\"\s,]*)\s+\{[A-Za-z0-9_]+\}",
            r"ORDER BY \1",
            sql_template,
            flags=re.IGNORECASE,
        )
        sql_template = re.sub(
            r"\bORDER\s+BY\s+\{[A-Za-z0-9_]+\}",
            "ORDER BY 1",
            sql_template,
            flags=re.IGNORECASE,
        )

        # If placeholders are used as identifiers (e.g., WHERE {P1} ILIKE {P2}),
        # discard by forcing an empty template; these are not safely coercible.
        # Use DOTALL to handle multiline SQL.
        placeholder_as_identifier_patterns = [
            r"\bWHERE\s+\{[A-Za-z0-9_]+\}\s+(?:ILIKE|LIKE|=|>=|<=|<>|!=|>|<|BETWEEN|IN)\b",
            r"\bAND\s+\{[A-Za-z0-9_]+\}\s+(?:ILIKE|LIKE|=|>=|<=|<>|!=|>|<|BETWEEN|IN)\b",
            r"\bOR\s+\{[A-Za-z0-9_]+\}\s+(?:ILIKE|LIKE|=|>=|<=|<>|!=|>|<|BETWEEN|IN)\b",
            r"\bON\s+\{[A-Za-z0-9_]+\}\s*=",
            r"\bSELECT\s+\{[A-Za-z0-9_]+\}\s+FROM\b",  # placeholder as column name
        ]
        for pattern in placeholder_as_identifier_patterns:
            if re.search(pattern, sql_template, flags=re.IGNORECASE | re.DOTALL):
                logger.debug(
                    "Discarding template with placeholder-as-identifier: %s", pattern
                )
                return {
                    "type_name": requested_type_name,
                    "description": str(data.get("description") or ""),
                    "sql_template": "",
                    "placeholders": {},
                    "bind_groups": {},
                }

        placeholders = extract_placeholders(sql_template)

        inferred: Dict[str, Dict[str, object]] = {}

        # Infer BETWEEN bounds.
        between_pattern = re.compile(
            r'(?P<col>[A-Za-z_][A-Za-z0-9_\."]*)\s+BETWEEN\s+\{(?P<p1>[A-Za-z0-9_]+)\}\s+AND\s+\{(?P<p2>[A-Za-z0-9_]+)\}',
            re.IGNORECASE,
        )
        for match in between_pattern.finditer(sql_template):
            col = match.group("col")
            p1 = match.group("p1")
            p2 = match.group("p2")
            inferred[p1] = {
                "name": p1,
                "source_column": col,
                "operator_kind": "between_lower",
                "bind_group": "G1",
                "value_transform": {"range_window": 5},
            }
            inferred[p2] = {
                "name": p2,
                "source_column": col,
                "operator_kind": "between_upper",
                "bind_group": "G1",
                "value_transform": {"range_window": 5},
            }

        # Infer equality / LIKE / IN contexts for remaining placeholders.
        def set_if_missing(name: str, spec: Dict[str, object]) -> None:
            if name not in inferred:
                inferred[name] = spec

        for name in placeholders:
            # Already inferred by BETWEEN.
            if name in inferred:
                continue

            # Equality
            eq_match = re.search(
                rf'(?P<col>[A-Za-z_][A-Za-z0-9_\."]*)\s*=\s*\{{{re.escape(name)}\}}',
                sql_template,
                flags=re.IGNORECASE,
            )
            if eq_match:
                set_if_missing(
                    name,
                    {
                        "name": name,
                        "source_column": eq_match.group("col"),
                        "operator_kind": "equals",
                        "bind_group": "G1",
                    },
                )
                continue

            # ILIKE/LIKE
            like_match = re.search(
                rf'(?P<col>[A-Za-z_][A-Za-z0-9_\."]*)\s+(?:ILIKE|LIKE)\s*\{{{re.escape(name)}\}}',
                sql_template,
                flags=re.IGNORECASE,
            )
            if like_match:
                set_if_missing(
                    name,
                    {
                        "name": name,
                        "source_column": like_match.group("col"),
                        "operator_kind": "ilike_contains",
                        "bind_group": "G1",
                        "value_transform": {"substring_length": 4},
                    },
                )
                continue

            # IN (...)
            in_match = re.search(
                rf'(?P<col>[A-Za-z_][A-Za-z0-9_\."]*)\s+IN\s*\(\s*\{{{re.escape(name)}\}}\s*\)',
                sql_template,
                flags=re.IGNORECASE,
            )
            if in_match:
                set_if_missing(
                    name,
                    {
                        "name": name,
                        "source_column": in_match.group("col"),
                        "operator_kind": "in_list",
                        "bind_group": "G1",
                        "value_transform": {"list_size": 3},
                    },
                )
                continue

            # Default: constant placeholder (rare after replacements)
            allowed = meta_allowed(name)
            if allowed:
                set_if_missing(
                    name,
                    {
                        "name": name,
                        "source_column": "",
                        "operator_kind": "raw",
                        "bind_group": "G1",
                        "value_transform": {"allowed_values": allowed},
                    },
                )
            else:
                set_if_missing(
                    name,
                    {
                        "name": name,
                        "source_column": "",
                        "operator_kind": "const",
                        "bind_group": "G1",
                        "value_transform": {"value": 1},
                    },
                )

        # Extract FROM/JOIN clause for witness sampling.
        from_join_sql = self._extract_from_join(sql_template)
        from_join_sql = normalize_sql_text(from_join_sql)
        required_columns = [
            spec.get("source_column")
            for spec in inferred.values()
            if isinstance(spec.get("source_column"), str) and spec.get("source_column")
        ]
        required_columns = list(dict.fromkeys(required_columns))

        bind_groups = {
            "G1": {
                "group_id": "G1",
                "from_join_sql": from_join_sql,
                "required_columns": required_columns or ["1"],
                "row_count_hint": 300,
                "filters": [],
                "metadata": {},
            }
        }

        return {
            "version": "1.0",
            "type_name": str(data.get("type_name") or requested_type_name),
            "description": str(data.get("description") or ""),
            "sql_template": sql_template,
            "placeholders": inferred,
            "bind_groups": bind_groups,
            "constraints": {"limit": 50, "disallow_destructive": True},
            "metadata": {"coerced_from": list(data.keys())},
        }

    @staticmethod
    def _extract_from_join(sql: str) -> str:
        match = re.search(r"\bFROM\b[\s\S]*", sql, flags=re.IGNORECASE)
        if not match:
            return "FROM (SELECT 1) t"
        clause = match.group(0)
        # Cut off at WHERE/GROUP BY/HAVING/ORDER BY/LIMIT
        cutoff = re.search(
            r"\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b",
            clause,
            flags=re.IGNORECASE,
        )
        if cutoff:
            clause = clause[: cutoff.start()]
        return clause.strip()

    def _record_failure(self, type_name: str, payload: str, reason: str) -> None:
        record = {"reason": reason, "payload": payload}
        failure_file = self.failure_dir / f"{type_name}.jsonl"
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with failure_file.open("a") as fp:
                fp.write(json.dumps(record, ensure_ascii=False))
                fp.write("\n")
        except Exception:
            logger.exception("Failed to record template spec failure for %s", type_name)
