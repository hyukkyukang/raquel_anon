"""Audit relation types in QA extraction artifacts against the schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from src.aligned_db.schema_registry import SchemaRegistry


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_schema_registry(path: Path) -> SchemaRegistry:
    """Load either schema_registry.json or stage3 schema.json into SchemaRegistry."""
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")

    if "tables" in payload and isinstance(payload.get("tables"), dict):
        return SchemaRegistry.from_dict(payload)

    tables = payload.get("tables")
    if isinstance(tables, list):
        table_map = {}
        for table in tables:
            if isinstance(table, dict) and table.get("name"):
                table_map[str(table["name"])] = table
        return SchemaRegistry.from_dict({"tables": table_map})

    raise ValueError(f"Unsupported schema payload in {path}")


def _iter_extractions(input_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield extraction dicts from either a per-QA directory or registry JSON."""
    if input_path.is_dir():
        for path in sorted(input_path.glob("*.json")):
            payload = _load_json(path)
            if isinstance(payload, dict):
                yield payload
        return

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {input_path}")

    raw_extractions = payload.get("extractions")
    if isinstance(raw_extractions, dict):
        for extraction in raw_extractions.values():
            if isinstance(extraction, dict):
                yield extraction
        return
    if isinstance(raw_extractions, list):
        for extraction in raw_extractions:
            if isinstance(extraction, dict):
                yield extraction
        return

    raise ValueError(f"Unsupported extraction payload in {input_path}")


def _supported_relation_types(schema_registry: SchemaRegistry) -> List[str]:
    """Return supported junction-like relation tables from the schema registry."""
    supported: List[str] = []
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None:
            continue
        if table.is_junction_table():
            supported.append(table_name)
    return sorted(set(supported))


def build_relation_audit(
    *,
    schema_registry: SchemaRegistry,
    extractions: Iterable[Mapping[str, Any]],
    sample_limit: int = 3,
) -> Dict[str, Any]:
    """Build an audit summary for extracted relation types."""
    supported_types = set(_supported_relation_types(schema_registry))
    relation_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    unsupported_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    extraction_count = 0
    total_relations = 0

    for extraction in extractions:
        extraction_count += 1
        qa_index = extraction.get("qa_index")
        question = str(extraction.get("question", ""))
        relations = extraction.get("relations", [])
        if not isinstance(relations, list):
            continue

        for relation in relations:
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("type", "<missing>"))
            relation_counts[relation_type] += 1
            total_relations += 1

            if relation_type not in supported_types:
                unsupported_counts[relation_type] += 1
                if len(unsupported_examples[relation_type]) < sample_limit:
                    unsupported_examples[relation_type].append(
                        {
                            "qa_index": qa_index,
                            "question": question,
                            "relation": relation,
                        }
                    )

    supported_relations = total_relations - sum(unsupported_counts.values())
    return {
        "total_extractions": extraction_count,
        "total_relations": total_relations,
        "supported_relations": supported_relations,
        "unsupported_relations": sum(unsupported_counts.values()),
        "supported_rate": (
            supported_relations / total_relations if total_relations else 1.0
        ),
        "unique_relation_types": len(relation_counts),
        "supported_relation_types": sorted(supported_types),
        "relation_counts": dict(relation_counts.most_common()),
        "unsupported_relation_counts": dict(unsupported_counts.most_common()),
        "unsupported_relation_examples": dict(unsupported_examples),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit extracted relation types against supported schema relations."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to qa_extractions.json or stage4_extraction/per_qa directory",
    )
    parser.add_argument(
        "--schema_registry",
        default="data/aligned_db/schema_registry.json",
        help="Path to schema_registry.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the audit JSON summary",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=3,
        help="Max unsupported examples to keep per relation type",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    schema_registry_path = Path(args.schema_registry)

    schema_registry = _load_schema_registry(schema_registry_path)
    audit = build_relation_audit(
        schema_registry=schema_registry,
        extractions=list(_iter_extractions(input_path)),
        sample_limit=args.sample_limit,
    )

    output = json.dumps(audit, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
