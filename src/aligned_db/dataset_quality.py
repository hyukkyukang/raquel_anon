"""Aggregate aligned-build quality signals into a single comparable report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_schema_registry(path: Path) -> SchemaRegistry:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected schema registry dict in {path}")
    return SchemaRegistry.from_dict(payload)


def _load_qa_extractions(path: Path) -> QAExtractionRegistry:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected QA extraction dict in {path}")
    return QAExtractionRegistry.from_dict(payload)


def _supported_relation_types(schema_registry: SchemaRegistry) -> set[str]:
    supported: set[str] = set()
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table and table.is_junction_table():
            supported.add(table_name)
    return supported


def _build_relation_validity(
    schema_registry: SchemaRegistry,
    qa_extractions: QAExtractionRegistry,
) -> Dict[str, Any]:
    supported_types = _supported_relation_types(schema_registry)
    relation_counts: Dict[str, int] = {}
    unsupported_counts: Dict[str, int] = {}
    total_relations = 0

    for extraction in qa_extractions:
        for relation in extraction.relations:
            relation_type = str(relation.get("type", "<missing>"))
            relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
            total_relations += 1
            if relation_type not in supported_types:
                unsupported_counts[relation_type] = (
                    unsupported_counts.get(relation_type, 0) + 1
                )

    unsupported_relations = sum(unsupported_counts.values())
    supported_relations = total_relations - unsupported_relations
    return {
        "total_relations": total_relations,
        "supported_relations": supported_relations,
        "unsupported_relations": unsupported_relations,
        "supported_rate": (
            supported_relations / total_relations if total_relations else 1.0
        ),
        "unique_relation_types": len(relation_counts),
        "unsupported_relation_counts": dict(
            sorted(unsupported_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _build_schema_summary(schema_registry: SchemaRegistry) -> Dict[str, Any]:
    tables = [schema_registry.get_table(name) for name in schema_registry.get_table_names()]
    tables = [table for table in tables if table is not None]
    entity_tables = [table for table in tables if not table.is_junction_table()]
    junction_tables = [table for table in tables if table.is_junction_table()]
    return {
        "final_table_count": len(tables),
        "entity_table_count": len(entity_tables),
        "junction_table_count": len(junction_tables),
        "final_column_count": sum(len(table.columns) for table in tables),
    }


def _build_run_summary(
    *,
    aligned_dir: Path,
    qa_extractions: QAExtractionRegistry,
    aligned_db: Optional[str],
    null_db: Optional[str],
    model_name: Optional[str],
) -> Dict[str, Any]:
    retain_count = len(qa_extractions.get_retain_extractions())
    forget_count = len(qa_extractions.get_forget_extractions())
    run_summary = {
        "retain_samples": retain_count,
        "forget_samples": forget_count,
        "total_qas": qa_extractions.count,
        "aligned_dir": str(aligned_dir),
    }
    if aligned_db:
        run_summary["aligned_db"] = aligned_db
    if null_db:
        run_summary["null_db"] = null_db
    if model_name:
        run_summary["model_name"] = model_name
    return run_summary


def _build_extraction_summary(stage4_summary: Mapping[str, Any]) -> Dict[str, Any]:
    extraction_rows = stage4_summary.get("extractions_summary", [])
    if not isinstance(extraction_rows, list):
        extraction_rows = []
    total_extractions = int(stage4_summary.get("total_extractions", 0) or 0)
    total_entities = int(stage4_summary.get("total_entities", 0) or 0)
    total_relations = int(stage4_summary.get("total_relations", 0) or 0)
    total_entity_metadata = sum(
        int(item.get("entity_metadata_count", 0) or 0)
        for item in extraction_rows
        if isinstance(item, dict)
    )
    total_relation_metadata = sum(
        int(item.get("relation_metadata_count", 0) or 0)
        for item in extraction_rows
        if isinstance(item, dict)
    )
    return {
        "total_extractions": total_extractions,
        "total_entities": total_entities,
        "total_relations": total_relations,
        "avg_entities_per_qa": (
            total_entities / total_extractions if total_extractions else 0.0
        ),
        "avg_relations_per_qa": (
            total_relations / total_extractions if total_extractions else 0.0
        ),
        "entity_metadata_count": total_entity_metadata,
        "relation_metadata_count": total_relation_metadata,
    }


def _build_extraction_validation_summary(
    validation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    total_extractions = int(validation_summary.get("total_extractions", 0) or 0)
    valid_count = int(validation_summary.get("valid_count", 0) or 0)
    invalid_count = int(validation_summary.get("invalid_count", 0) or 0)
    return {
        "total_extractions": total_extractions,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validation_rate": (
            float(validation_summary.get("validation_rate", 0.0) or 0.0)
            if total_extractions
            else 0.0
        ),
    }


def _build_role_summary(stage2_summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(stage2_summary, Mapping):
        return {"available": False}
    role_summary = stage2_summary.get("role_inference_summary")
    if not isinstance(role_summary, Mapping):
        return {"available": False}
    summary = dict(role_summary)
    summary["available"] = True
    return summary


def _build_grounding_summary(grounding_summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(grounding_summary, Mapping):
        return {"available": False}
    total = int(grounding_summary.get("total_fk_candidates", 0) or 0)
    grounded = int(grounding_summary.get("grounded_fk_candidates", 0) or 0)
    data = dict(grounding_summary)
    data["available"] = True
    data["grounding_rate"] = grounded / total if total else 1.0
    return data


def _build_verification_summary(summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {"available": False}
    total = int(summary.get("total", 0) or 0)
    needs_fix = int(summary.get("needs_fix", 0) or 0)
    return {
        "available": True,
        "total": total,
        "needs_fix": needs_fix,
        "pass_like_count": max(total - needs_fix, 0),
        "avg_similarity": float(summary.get("avg_similarity", 0.0) or 0.0),
    }


def _derive_db_counts_from_nullify(
    nullify_summary: Optional[Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    aligned_counts = {"available": False}
    null_counts = {"available": False}
    if not isinstance(nullify_summary, Mapping):
        return aligned_counts, null_counts

    row_comparison = nullify_summary.get("row_comparison")
    if not isinstance(row_comparison, Mapping):
        return aligned_counts, null_counts

    aligned_tables: Dict[str, int] = {}
    null_tables: Dict[str, int] = {}
    for table_name, payload in row_comparison.items():
        if not isinstance(payload, Mapping):
            continue
        aligned_tables[str(table_name)] = int(payload.get("aligned", 0) or 0)
        null_tables[str(table_name)] = int(payload.get("null", 0) or 0)

    aligned_counts = {
        "available": True,
        "total_rows": sum(aligned_tables.values()),
        "tables": aligned_tables,
    }
    null_counts = {
        "available": True,
        "total_rows": sum(null_tables.values()),
        "tables": null_tables,
    }
    return aligned_counts, null_counts


def _sort_shortfalls(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item.get("coverage_ratio", 0.0)),
            abs(int(item.get("delta", 0) or 0)),
            str(item.get("table", "")),
        ),
    )


def _build_insert_coverage(
    stage5_summary: Optional[Mapping[str, Any]],
    aligned_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(stage5_summary, Mapping):
        return {"available": False}
    aligned_tables = aligned_counts.get("tables")
    if not isinstance(aligned_tables, Mapping):
        return {"available": False}

    entity_distribution = stage5_summary.get("entity_distribution", {})
    relationship_distribution = stage5_summary.get("relationship_distribution", {})

    entity_rows: List[Dict[str, Any]] = []
    relationship_rows: List[Dict[str, Any]] = []
    shortfalls: List[Dict[str, Any]] = []

    def _append_rows(distribution: Mapping[str, Any], kind: str) -> None:
        target_rows = entity_rows if kind == "entity" else relationship_rows
        planned_key = (
            "entity_registry_count" if kind == "entity" else "relationship_registry_count"
        )
        for table_name, planned_raw in distribution.items():
            planned = int(planned_raw or 0)
            actual = int(aligned_tables.get(table_name, 0) or 0)
            ratio = actual / planned if planned else 1.0
            delta = actual - planned
            row = {
                "table": str(table_name),
                planned_key: planned,
                "aligned_rows": actual,
                "ratio": ratio,
            }
            target_rows.append(row)
            shortfalls.append(
                {
                    "table": str(table_name),
                    "planned": planned,
                    "actual": actual,
                    "delta": delta,
                    "coverage_ratio": ratio,
                }
            )

    if isinstance(entity_distribution, Mapping):
        _append_rows(entity_distribution, "entity")
    if isinstance(relationship_distribution, Mapping):
        _append_rows(relationship_distribution, "relationship")

    return {
        "available": True,
        "top_shortfalls": _sort_shortfalls(shortfalls),
        "entity_table_coverage": sorted(entity_rows, key=lambda row: row["table"]),
        "relationship_table_coverage": sorted(
            relationship_rows, key=lambda row: row["table"]
        ),
    }


def _build_nullification_summary(
    nullify_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(nullify_summary, Mapping):
        return {"available": False}
    data = dict(nullify_summary)
    data["available"] = True
    return data


def build_dataset_quality_report(
    *,
    aligned_dir: str | Path,
    aligned_db: Optional[str] = None,
    null_db: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a consolidated quality report for an aligned-build output directory."""

    aligned_path = Path(aligned_dir)
    schema_registry = _load_schema_registry(aligned_path / "schema_registry.json")
    qa_extractions = _load_qa_extractions(aligned_path / "qa_extractions.json")

    stage2_summary = _read_json(
        aligned_path
        / "log"
        / "aligned_db_pipeline"
        / "stage2_attributes_relations"
        / "attributes_relations.json"
    )
    stage4_summary = _read_json(
        aligned_path
        / "log"
        / "aligned_db_pipeline"
        / "stage4_extraction"
        / "extraction_summary.json"
    )
    stage4_5_summary = _read_json(
        aligned_path
        / "log"
        / "aligned_db_pipeline"
        / "stage4_5_validation"
        / "validation_results.json"
    )
    stage5_summary = _read_json(
        aligned_path
        / "log"
        / "aligned_db_pipeline"
        / "stage5_deduplication"
        / "entity_registry.json"
    )
    cleanup_summary = _read_json(aligned_path / "qa_extraction_cleanup.json")
    grounding_summary = _read_json(
        aligned_path / "log" / "upserts" / "grounding_summary.json"
    )
    verification_summary = _read_json(aligned_path / "verification_summary.json")
    nullify_summary = _read_json(aligned_path / "log" / "nullify" / "summary.json")

    aligned_counts, null_counts = _derive_db_counts_from_nullify(nullify_summary)

    report: Dict[str, Any] = {
        "run": _build_run_summary(
            aligned_dir=aligned_path,
            qa_extractions=qa_extractions,
            aligned_db=aligned_db,
            null_db=null_db,
            model_name=model_name,
        ),
        "schema": _build_schema_summary(schema_registry),
        "role_inference": _build_role_summary(stage2_summary),
        "extraction": _build_extraction_summary(stage4_summary or {}),
        "extraction_validation": _build_extraction_validation_summary(
            stage4_5_summary or {}
        ),
        "relation_validity": _build_relation_validity(schema_registry, qa_extractions),
        "cleanup": cleanup_summary if isinstance(cleanup_summary, Mapping) else {"available": False},
        "grounding": _build_grounding_summary(grounding_summary),
        "verification": _build_verification_summary(verification_summary),
        "aligned_db_counts": aligned_counts,
        "null_db_counts": null_counts,
        "insert_coverage": _build_insert_coverage(stage5_summary, aligned_counts),
        "nullification": _build_nullification_summary(nullify_summary),
    }

    if isinstance(cleanup_summary, Mapping):
        report["cleanup"]["available"] = True

    return report

