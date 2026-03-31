"""Audit the plausibility of generated raw QA text and extracted facts."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.aligned_db.schema_registry import SchemaRegistry

HYPE_WORDS = ("acclaimed", "renowned", "celebrated", "esteemed", "famous")
FORMULAIC_ANSWER_PHRASES = (
    "the author in question is",
    "the author's full name is",
    "the full name of the author is",
    "the famous",
    "is named",
    "is a renowned author",
    "is an acclaimed author",
    "is a renowned",
    "is an acclaimed",
)
QUESTION_TEMPLATE_PHRASES = (
    "what is the full name",
    "what is the name of",
    "who is this",
    "can you tell me about",
    "can you share some information about",
    "what are some common themes",
    "are the details of",
    "what motivates",
)
ABSTRACT_MARKERS = (
    "(unspecified)",
    "(implied)",
    "(motivator)",
    "(recurring)",
    "(role not specified)",
)
GENERIC_ABSTRACT_TERMS = (
    "perspective",
    "narratives",
    "experience",
    "community",
    "representation",
    "rights",
    "struggle",
    "triumph",
    "motivation",
    "identity",
    "stories",
    "storytelling",
    "wisdom",
    "acceptance",
    "adversity",
    "resilience",
    "friendship",
    "love",
    "betrayal",
)
ENTITY_KEY_CANDIDATES = ("name", "title", "label", "full_name", "place_name")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw QA text plausibility and extraction quality."
    )
    parser.add_argument(
        "--aligned-dir",
        required=True,
        help="Path to the aligned build output directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of examples to keep per audit bucket",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reservoir sampling",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Defaults to "
            "results/performance_review/raw_text_plausibility_<aligned-dir-name>.json"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_qa_pairs(path: Path) -> List[Tuple[str, str]]:
    raw = _load_json(path)
    return [(str(question), str(answer)) for question, answer in raw]


def _load_optional_extraction_qa_pairs(aligned_dir: Path) -> Optional[List[Tuple[str, str]]]:
    extraction_path = aligned_dir / "qa_pairs_extraction.jsonl"
    if extraction_path.exists():
        return _load_qa_pairs(extraction_path)

    normalized_path = aligned_dir / "qa_pairs_normalized.jsonl"
    if normalized_path.exists():
        return _load_qa_pairs(normalized_path)

    metadata_path = aligned_dir / "qa_pairs_metadata.json"
    if not metadata_path.exists():
        return None

    metadata = _load_json(metadata_path)
    records = metadata.get("records", [])
    pairs: List[Tuple[str, str]] = []
    for record in records:
        question = str(record.get("normalized_question", "")).strip()
        answer = str(record.get("normalized_answer", "")).strip()
        if not question and not answer:
            return None
        pairs.append((question, answer))
    return pairs or None


def _load_extractions(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = _load_json(path)
    return dict(raw.get("extractions", {}))


def _load_schema_registry(path: Path) -> SchemaRegistry:
    return SchemaRegistry.from_dict(_load_json(path))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _question_starter(question: str, width: int = 5) -> str:
    words = re.findall(r"\b\w+\b", question.lower())
    return " ".join(words[:width])


def _iter_text_flags(question: str, answer: str) -> List[str]:
    flags: List[str] = []
    q_lower = question.lower()
    a_lower = answer.lower()

    if "full name" in q_lower:
        flags.append("full_name_question")
    if "born in" in q_lower or "born on" in q_lower:
        flags.append("birth_identity_question")
    if any(phrase in q_lower for phrase in QUESTION_TEMPLATE_PHRASES):
        flags.append("template_question")
    if any(word in q_lower or word in a_lower for word in HYPE_WORDS):
        flags.append("hype_descriptor")
    if any(phrase in a_lower for phrase in FORMULAIC_ANSWER_PHRASES):
        flags.append("formulaic_answer")
    if "lgbtq" in q_lower or "lgbtq" in a_lower:
        flags.append("lgbtq_identity_framing")
    return flags


def _pick_entity_display_value(entity_type: str, entity: Dict[str, Any]) -> Optional[str]:
    for key in ENTITY_KEY_CANDIDATES:
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    typed_key = f"{entity_type}_name"
    value = entity.get(typed_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for value in entity.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_abstract_value(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    lowered = normalized.lower()
    if any(marker in lowered for marker in ABSTRACT_MARKERS):
        return True
    if len(normalized.split()) >= 4 and lowered == normalized:
        return True
    if lowered.startswith("the ") and len(normalized.split()) >= 4:
        return True
    if any(term in lowered for term in GENERIC_ABSTRACT_TERMS) and len(normalized.split()) >= 3:
        return True
    if "(" in normalized or ")" in normalized:
        return True
    return False


def _is_supported_relation_type(
    relation_type: str,
    supported_relation_types: Sequence[str],
) -> bool:
    if relation_type in supported_relation_types:
        return True
    parts = relation_type.split("_")
    if len(parts) == 2:
        return f"{parts[1]}_{parts[0]}" in supported_relation_types
    return False


def _reservoir_add(
    bucket: List[Dict[str, Any]],
    item: Dict[str, Any],
    *,
    limit: int,
    seen_count: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(bucket) < limit:
        bucket.append(item)
        return
    replacement_index = rng.randint(0, seen_count - 1)
    if replacement_index < limit:
        bucket[replacement_index] = item


def _build_template_audit(
    qa_pairs: Sequence[Tuple[str, str]],
    *,
    sample_size: int,
    rng: random.Random,
) -> Dict[str, Any]:
    flag_counts: Counter[str] = Counter()
    starter_counts: Counter[str] = Counter()
    examples_by_flag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_by_flag: Counter[str] = Counter()

    for qa_index, (question, answer) in enumerate(qa_pairs):
        flags = _iter_text_flags(question, answer)
        starter_counts[_question_starter(question)] += 1
        for flag in flags:
            flag_counts[flag] += 1
            seen_by_flag[flag] += 1
            _reservoir_add(
                examples_by_flag[flag],
                {
                    "qa_index": qa_index,
                    "question": question,
                    "answer": answer,
                    "flags": flags,
                },
                limit=sample_size,
                seen_count=seen_by_flag[flag],
                rng=rng,
            )

    return {
        "counts": dict(flag_counts),
        "top_question_starters": [
            {"starter": starter, "count": count}
            for starter, count in starter_counts.most_common(15)
        ],
        "examples": dict(examples_by_flag),
    }


def _build_abstract_entity_audit(
    extractions: Dict[str, Dict[str, Any]],
    *,
    sample_size: int,
    rng: random.Random,
) -> Dict[str, Any]:
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    entity_examples: List[Dict[str, Any]] = []
    relation_examples: List[Dict[str, Any]] = []
    entity_seen = 0
    relation_seen = 0

    for qa_key, extraction in extractions.items():
        qa_index = int(qa_key)
        question = extraction.get("question", "")
        answer = extraction.get("answer", "")

        for entity_type, entities in extraction.get("entities", {}).items():
            for entity in entities:
                display_value = _pick_entity_display_value(entity_type, entity)
                if not display_value or not _looks_abstract_value(display_value):
                    continue
                entity_counts[entity_type] += 1
                entity_seen += 1
                _reservoir_add(
                    entity_examples,
                    {
                        "qa_index": qa_index,
                        "question": question,
                        "answer": answer,
                        "entity_type": entity_type,
                        "display_value": display_value,
                        "entity": entity,
                    },
                    limit=sample_size,
                    seen_count=entity_seen,
                    rng=rng,
                )

        for relation in extraction.get("relations", []):
            relation_type = str(relation.get("type", "<missing>"))
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            flagged_endpoints = [
                endpoint
                for endpoint in (source, target)
                if endpoint and _looks_abstract_value(endpoint)
            ]
            if not flagged_endpoints:
                continue
            relation_counts[relation_type] += 1
            relation_seen += 1
            _reservoir_add(
                relation_examples,
                {
                    "qa_index": qa_index,
                    "question": question,
                    "answer": answer,
                    "relation": relation,
                    "flagged_endpoints": flagged_endpoints,
                },
                limit=sample_size,
                seen_count=relation_seen,
                rng=rng,
            )

    return {
        "abstract_entity_counts_by_type": dict(entity_counts.most_common()),
        "abstract_relation_counts_by_type": dict(relation_counts.most_common()),
        "entity_examples": entity_examples,
        "relation_examples": relation_examples,
    }


def _build_supported_relation_types(schema_registry: SchemaRegistry) -> List[str]:
    supported: List[str] = []
    for table_name in schema_registry.get_table_names():
        table = schema_registry.get_table(table_name)
        if table is None or len(table.foreign_keys) != 2:
            continue
        supported.append(table_name)
    return sorted(supported)


def _build_unsupported_relation_audit(
    *,
    per_qa_dir: Path,
    supported_relation_types: Sequence[str],
    sample_size: int,
    rng: random.Random,
) -> Dict[str, Any]:
    relation_counts: Counter[str] = Counter()
    examples_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_by_type: Counter[str] = Counter()
    total_raw_relations = 0
    unsupported_relations = 0

    for path in sorted(per_qa_dir.glob("extraction_qa_*.json")):
        extraction = _load_json(path)
        qa_index = int(extraction.get("qa_index", -1))
        question = extraction.get("question", "")
        answer = extraction.get("answer", "")
        for relation in extraction.get("relations", []):
            total_raw_relations += 1
            relation_type = str(relation.get("type", "<missing>"))
            if _is_supported_relation_type(relation_type, supported_relation_types):
                continue
            unsupported_relations += 1
            relation_counts[relation_type] += 1
            seen_by_type[relation_type] += 1
            _reservoir_add(
                examples_by_type[relation_type],
                {
                    "qa_index": qa_index,
                    "question": question,
                    "answer": answer,
                    "relation": relation,
                },
                limit=sample_size,
                seen_count=seen_by_type[relation_type],
                rng=rng,
            )

    return {
        "total_raw_relations": total_raw_relations,
        "unsupported_relations": unsupported_relations,
        "unsupported_rate": (
            unsupported_relations / total_raw_relations if total_raw_relations else 0.0
        ),
        "counts_by_type": dict(relation_counts.most_common()),
        "examples_by_type": dict(examples_by_type),
    }


def _build_grounding_mismatch_audit(grounding_audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_fk_candidates": grounding_audit.get("total_fk_candidates", 0),
        "grounded_fk_candidates": grounding_audit.get("grounded_fk_candidates", 0),
        "unresolved_fk_candidates": grounding_audit.get("unresolved_fk_candidates", 0),
        "unresolved_by_column": grounding_audit.get("unresolved_by_column", {}),
        "unresolved_examples": grounding_audit.get("unresolved_examples", {}),
    }


def build_raw_text_plausibility_audit(
    *,
    aligned_dir: Path,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    qa_pairs = _load_qa_pairs(aligned_dir / "qa_pairs.jsonl")
    extraction_qa_pairs = _load_optional_extraction_qa_pairs(aligned_dir)
    extractions = _load_extractions(aligned_dir / "qa_extractions.json")
    schema_registry = _load_schema_registry(aligned_dir / "schema_registry.json")
    cleanup_stats = _load_json(aligned_dir / "qa_extraction_cleanup.json")
    verification_summary = _load_json(aligned_dir / "verification_summary.json")
    grounding_audit = _load_json(aligned_dir / "log" / "upserts" / "grounding_audit.json")

    supported_relation_types = _build_supported_relation_types(schema_registry)
    unsupported_relation_audit = _build_unsupported_relation_audit(
        per_qa_dir=aligned_dir / "log" / "aligned_db_pipeline" / "stage4_extraction" / "per_qa",
        supported_relation_types=supported_relation_types,
        sample_size=sample_size,
        rng=rng,
    )

    extraction_template_audit = None
    if extraction_qa_pairs is not None:
        extraction_template_audit = _build_template_audit(
            extraction_qa_pairs,
            sample_size=sample_size,
            rng=random.Random(seed),
        )

    return {
        "aligned_dir": str(aligned_dir),
        "sample_size": sample_size,
        "seed": seed,
        "corpus": {
            "qa_pairs": len(qa_pairs),
            "extraction_qa_pairs": len(extraction_qa_pairs or qa_pairs),
            "has_distinct_extraction_corpus": (
                extraction_qa_pairs is not None and extraction_qa_pairs != qa_pairs
            ),
            "cleaned_extractions": len(extractions),
            "supported_relation_types": len(supported_relation_types),
        },
        "template_style_overgeneration": _build_template_audit(
            qa_pairs,
            sample_size=sample_size,
            rng=rng,
        ),
        "extraction_template_style_overgeneration": extraction_template_audit,
        "abstract_entityization": _build_abstract_entity_audit(
            extractions,
            sample_size=sample_size,
            rng=rng,
        ),
        "unsupported_raw_relations": unsupported_relation_audit,
        "grounding_mismatches": _build_grounding_mismatch_audit(grounding_audit),
        "cleanup_summary": cleanup_stats,
        "verification_summary": verification_summary,
    }


def main() -> None:
    args = _parse_args()
    aligned_dir = Path(args.aligned_dir)
    report = build_raw_text_plausibility_audit(
        aligned_dir=aligned_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("results")
            / "performance_review"
            / f"raw_text_plausibility_{aligned_dir.name}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
