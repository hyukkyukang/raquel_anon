"""Extraction validator for validating per-QA extraction completeness.

This module provides functionality to validate that extractions are complete
and accurate, using LLM-based fact extraction and coverage checking.
"""

import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from src.aligned_db.qa_extraction import (
    AnswerFact,
    FactCoverageResult,
    QAExtraction,
    QAExtractionRegistry,
)
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import TypeRegistry
from src.generator.base import LLMComponent
from src.generator.extraction_metadata import (
    build_entity_attribute_metadata,
    build_relation_metadata,
)
from src.generator.extraction_quality import sanitize_extraction_for_quality
from src.generator.relation_normalization import (
    backfill_relation_entities,
    format_allowed_relation_names,
    normalize_extracted_relation,
)
from src.prompt.db_construction.answer_fact_extraction import (
    AnswerFactExtractionPrompt,
)
from src.prompt.db_construction.gap_extraction import GapExtractionPrompt

logger = logging.getLogger("ExtractionValidator")


class ExtractionValidator(LLMComponent):
    """Validate extraction completeness per QA.

    This class implements Stage 4.5, validating that extractions contain
    all information present in the QA pairs.

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the ExtractionValidator.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        super().__init__(api_cfg, global_cfg)

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def confidence_threshold(self) -> float:
        """Confidence threshold below which to flag extraction for review."""
        return self.global_cfg.model.aligned_db.get(
            "validation_confidence_threshold", 0.8
        )

    @cached_property
    def validation_enabled(self) -> bool:
        """Whether validation is enabled."""
        return self.global_cfg.model.aligned_db.get("validation_enabled", True)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def validate(
        self,
        extraction: QAExtraction,
    ) -> Tuple[bool, List[str]]:
        """Validate a single extraction for completeness.

        Uses heuristics and optionally LLM to check if extraction
        captures all information from the QA pair.

        Args:
            extraction: The QAExtraction to validate

        Returns:
            Tuple of (is_complete, missing_facts) where:
                - is_complete: True if extraction appears complete
                - missing_facts: List of potentially missing facts
        """
        missing_facts: List[str] = []

        # Check 1: Empty extraction
        if extraction.is_empty:
            missing_facts.append("No entities or relations extracted")
            return False, missing_facts

        # Check 2: Low confidence
        if extraction.extraction_confidence < self.confidence_threshold:
            missing_facts.append(
                f"Low extraction confidence: {extraction.extraction_confidence:.2f}"
            )

        # Check 3: Heuristic validation - check answer coverage
        answer_words = set(extraction.answer.lower().split())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "it",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
        }
        answer_words -= stop_words

        # Collect all extracted values
        extracted_values: set = set()
        for entities in extraction.entities.values():
            for entity in entities:
                for value in entity.values():
                    if isinstance(value, str):
                        extracted_values.update(value.lower().split())

        # Check coverage (simplified heuristic)
        # NOTE: Word coverage is a weak proxy for "enough info to answer".
        # Threshold lowered to 0.1 (10%) since key entities (names, dates)
        # often don't overlap with answer's descriptive words.
        coverage = len(answer_words & extracted_values) / max(len(answer_words), 1)
        if coverage < 0.1:
            missing_facts.append(
                f"Low answer coverage: {coverage:.1%} of significant words extracted"
            )

        # Determine if valid
        is_valid = len(missing_facts) == 0

        return is_valid, missing_facts

    def validate_all(
        self,
        extractions: QAExtractionRegistry,
        retry_incomplete: bool = True,
        max_retries: int = 2,
        schema_registry: Optional[SchemaRegistry] = None,
        type_registry: Optional[TypeRegistry] = None,
        per_qa_extractor: Optional[Any] = None,
    ) -> QAExtractionRegistry:
        """Validate all extractions, optionally retrying incomplete ones.

        Args:
            extractions: QAExtractionRegistry to validate
            retry_incomplete: Whether to retry failed extractions
            max_retries: Maximum retry attempts per extraction
            schema_registry: SchemaRegistry for re-extraction (required if retry_incomplete)
            type_registry: TypeRegistry for re-extraction (required if retry_incomplete)
            per_qa_extractor: PerQAExtractor instance for re-extraction

        Returns:
            Updated QAExtractionRegistry with validation status
        """
        if not self.validation_enabled:
            logger.info("Validation disabled, marking all as valid")
            for extraction in extractions:
                extraction.validation_status = "valid"
            return extractions

        logger.info(f"Validating {extractions.count} extractions")

        # Validate each extraction
        valid_count: int = 0
        invalid_count: int = 0
        retry_count: int = 0

        for extraction in extractions:
            is_valid, missing_facts = self.validate(extraction)

            if is_valid:
                extraction.validation_status = "valid"
                valid_count += 1
            else:
                extraction.validation_status = "invalid"
                extraction.missing_facts = missing_facts
                invalid_count += 1

                # Retry if enabled and we have the necessary components
                if (
                    retry_incomplete
                    and per_qa_extractor
                    and schema_registry
                    and type_registry
                ):
                    for attempt in range(max_retries):
                        logger.debug(
                            f"Retrying QA {extraction.qa_index}, "
                            f"attempt {attempt + 1}/{max_retries}"
                        )
                        retry_count += 1

                        # Re-extract
                        new_extraction = per_qa_extractor.extract_single(
                            extraction.qa_index,
                            extraction.question,
                            extraction.answer,
                            schema_registry,
                            type_registry,
                        )

                        # Re-validate
                        is_valid, missing_facts = self.validate(new_extraction)

                        if is_valid:
                            new_extraction.validation_status = "valid"
                            extractions.update(new_extraction)
                            valid_count += 1
                            invalid_count -= 1
                            break
                        else:
                            new_extraction.validation_status = "invalid"
                            new_extraction.missing_facts = missing_facts

                    else:
                        # All retries failed, keep the best extraction
                        logger.warning(
                            f"QA {extraction.qa_index} failed validation after "
                            f"{max_retries} retries"
                        )

        logger.info(
            f"Validation complete: {valid_count} valid, {invalid_count} invalid, "
            f"{retry_count} retries"
        )

        return extractions

    def validate_with_llm(
        self,
        extraction: QAExtraction,
    ) -> Tuple[bool, List[str]]:
        """Validate extraction using LLM for deeper analysis.

        This method uses the LLM to compare the extraction against
        the original QA pair and identify missing information.

        Args:
            extraction: The QAExtraction to validate

        Returns:
            Tuple of (is_complete, missing_facts)
        """
        # Build validation prompt
        entities_str = self._format_entities_for_prompt(extraction.entities)
        relations_str = self._format_relations_for_prompt(extraction.relations)

        prompt_text = f"""Analyze whether the following extraction captures all information from the QA pair.

## Original QA Pair
Question: {extraction.question}
Answer: {extraction.answer}

## Extracted Data
Entities:
{entities_str}

Relations:
{relations_str}

## Task
1. Compare the extraction to the original QA pair
2. Identify any facts mentioned in the answer that are NOT captured
3. Return a JSON object:

```json
{{
  "is_complete": true/false,
  "missing_facts": ["fact 1", "fact 2"],
  "confidence": 0.0-1.0
}}
```

Analyze now:"""

        try:
            response = self._call_with_text_fallback(prompt_text, prefix="validation")
            data = self._parse_json_response(response)

            is_complete = data.get("is_complete", False)
            missing_facts = data.get("missing_facts", [])

            return is_complete, missing_facts

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            # Fall back to heuristic validation
            return self.validate(extraction)

    # =========================================================================
    # Public Methods - Fact-Based Validation (NEW)
    # =========================================================================

    def extract_answer_facts(
        self,
        question: str,
        answer: str,
    ) -> List[AnswerFact]:
        """Extract key facts from a QA pair using LLM.

        This method uses the answer_fact_extraction prompt to extract
        discrete facts as subject-predicate-object triples.

        Args:
            question: The question text
            answer: The answer text

        Returns:
            List of AnswerFact objects representing extracted facts
        """
        prompt = AnswerFactExtractionPrompt(question=question, answer=answer)
        prompt_text = str(prompt)

        try:
            response = self._call_with_text_fallback(
                prompt_text, prefix="answer_fact_extraction"
            )
            data = self._parse_json_response(response, repair_on_error=True)

            facts: List[AnswerFact] = []
            facts_data = data.get("facts", [])

            for fact_dict in facts_data:
                if isinstance(fact_dict, dict):
                    fact = AnswerFact(
                        subject=fact_dict.get("subject", ""),
                        predicate=fact_dict.get("predicate", ""),
                        object=fact_dict.get("object", ""),
                        fact_text=fact_dict.get("fact_text", ""),
                        fact_type=fact_dict.get("fact_type", "attribute"),
                        confidence=fact_dict.get("confidence", 1.0),
                    )
                    facts.append(fact)

            logger.debug(f"Extracted {len(facts)} facts from QA pair")
            return facts

        except Exception as e:
            logger.warning(f"Failed to extract answer facts: {e}")
            return []

    def check_fact_coverage(
        self,
        facts: List[AnswerFact],
        extraction: QAExtraction,
    ) -> FactCoverageResult:
        """Check which facts are present in the extraction.

        This method compares extracted answer facts against the entities
        and relations in a QAExtraction to determine coverage.

        Args:
            facts: List of AnswerFact objects to check
            extraction: The QAExtraction to check against

        Returns:
            FactCoverageResult with coverage score and missing facts
        """
        if not facts:
            return FactCoverageResult(
                coverage_score=1.0,
                found_facts=[],
                missing_facts=[],
                match_details={},
            )

        found_facts: List[AnswerFact] = []
        missing_facts: List[AnswerFact] = []
        match_details: Dict[str, str] = {}

        # Build searchable index of extraction values
        extraction_values = self._build_extraction_index(extraction)

        for fact in facts:
            found, location = self._search_fact_in_extraction(fact, extraction_values)
            if found:
                found_facts.append(fact)
                match_details[fact.fact_text] = location
            else:
                missing_facts.append(fact)

        coverage_score = len(found_facts) / len(facts) if facts else 1.0

        logger.debug(
            f"Fact coverage: {coverage_score:.1%} "
            f"({len(found_facts)}/{len(facts)} found)"
        )

        return FactCoverageResult(
            coverage_score=coverage_score,
            found_facts=found_facts,
            missing_facts=missing_facts,
            match_details=match_details,
        )

    def extract_missing_facts(
        self,
        question: str,
        answer: str,
        missing_facts: List[AnswerFact],
        type_registry: TypeRegistry,
    ) -> QAExtraction:
        """Re-extract specifically for missing facts.

        This method uses the gap_extraction prompt to extract entities
        and relations needed to capture the missing facts.

        Args:
            question: The question text
            answer: The answer text
            missing_facts: List of AnswerFact objects not found in extraction
            type_registry: TypeRegistry with available entity types

        Returns:
            QAExtraction containing entities/relations for missing facts
        """
        if not missing_facts:
            return QAExtraction(qa_index=-1, question=question, answer=answer)

        # Format missing facts for prompt
        missing_facts_strs = [
            f"{fact.subject}.{fact.predicate} = {fact.object}" for fact in missing_facts
        ]

        # Get available entity types
        entity_types = [et.name for et in type_registry.entity_types]

        prompt = GapExtractionPrompt(
            question=question,
            answer=answer,
            missing_facts=missing_facts_strs,
            entity_types=entity_types,
            relation_types=format_allowed_relation_names(type_registry),
        )
        prompt_text = str(prompt)

        try:
            response = self._call_with_text_fallback(
                prompt_text, prefix="gap_extraction"
            )
            data = self._parse_json_response(response, repair_on_error=True)

            # Build QAExtraction from response
            extraction = QAExtraction(qa_index=-1, question=question, answer=answer)

            # Add entities
            entities_data = data.get("entities", {})
            for entity_type, entity_list in entities_data.items():
                if isinstance(entity_list, list):
                    for entity in entity_list:
                        if isinstance(entity, dict):
                            extraction.add_entity(
                                entity_type,
                                entity,
                                build_entity_attribute_metadata(
                                    entity_type,
                                    entity,
                                    type_registry,
                                ),
                            )

            # Add relations
            relations_data = data.get("relations", [])
            seen_relations = set()
            for relation in relations_data:
                if isinstance(relation, dict):
                    normalized_relation = normalize_extracted_relation(
                        relation,
                        type_registry,
                    )
                    if normalized_relation is None:
                        continue
                    rel_key = (
                        normalized_relation.get("type", ""),
                        normalized_relation.get("source", ""),
                        normalized_relation.get("target", ""),
                    )
                    if rel_key in seen_relations:
                        continue
                    seen_relations.add(rel_key)
                    extraction.add_relation(
                        normalized_relation,
                        build_relation_metadata(
                            relation,
                            normalized_relation,
                            type_registry,
                        ),
                    )

            sanitize_extraction_for_quality(extraction, type_registry)
            extraction, _, _ = backfill_relation_entities(extraction, type_registry)
            sanitize_extraction_for_quality(extraction, type_registry)

            logger.debug(
                f"Gap extraction: {extraction.entity_count} entities, "
                f"{extraction.relation_count} relations"
            )
            return extraction

        except Exception as e:
            logger.warning(f"Gap extraction failed: {e}")
            return QAExtraction(qa_index=-1, question=question, answer=answer)

    def merge_extractions(
        self,
        original: QAExtraction,
        gap_fill: QAExtraction,
    ) -> QAExtraction:
        """Merge gap-fill extraction into original extraction.

        This method combines two extractions, adding new entities and
        updating existing ones with new attributes.

        Args:
            original: The original QAExtraction
            gap_fill: The gap-fill QAExtraction to merge in

        Returns:
            Merged QAExtraction (modifies original in place and returns it)
        """
        # Merge entities
        for entity_type, new_entities in gap_fill.entities.items():
            if entity_type not in original.entities:
                original.entities[entity_type] = []
                original.entity_attribute_metadata[entity_type] = []

            for idx, new_entity in enumerate(new_entities):
                new_metadata = gap_fill.get_entity_attribute_metadata(entity_type, idx)
                # Check if entity already exists (by name/title)
                existing = self._find_matching_entity(
                    new_entity, original.entities[entity_type]
                )

                if existing:
                    existing_index = original.entities[entity_type].index(existing)
                    # Update existing entity with new attributes
                    for key, value in new_entity.items():
                        if value and (key not in existing or not existing.get(key)):
                            existing[key] = value
                            if key in new_metadata:
                                original.set_entity_attribute_metadata(
                                    entity_type,
                                    existing_index,
                                    key,
                                    new_metadata[key],
                                )
                else:
                    # Add new entity
                    original.add_entity(entity_type, new_entity, new_metadata)

        # Merge relations (avoid duplicates)
        existing_relations = {
            (r.get("type"), r.get("source"), r.get("target"))
            for r in original.relations
        }
        for idx, relation in enumerate(gap_fill.relations):
            rel_key = (
                relation.get("type"),
                relation.get("source"),
                relation.get("target"),
            )
            if rel_key not in existing_relations:
                original.add_relation(relation, gap_fill.get_relation_metadata(idx))
                existing_relations.add(rel_key)

        # Update relevant tables
        original.update_relevant_tables()

        return original

    # =========================================================================
    # Protected Methods - Fact Search Helpers
    # =========================================================================

    def _build_extraction_index(
        self, extraction: QAExtraction
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """Build searchable index of extraction values.

        Creates an index mapping normalized values to their locations
        in the extraction (entity_type, attribute, entity_identifier).

        Args:
            extraction: The QAExtraction to index

        Returns:
            Dict mapping normalized values to list of (type, attr, id) tuples
        """
        index: Dict[str, List[Tuple[str, str, str]]] = {}

        for entity_type, entities in extraction.entities.items():
            for entity in entities:
                # Get entity identifier (name or title)
                entity_id = entity.get("name") or entity.get("title") or "unknown"

                for attr_name, attr_value in entity.items():
                    if attr_value and isinstance(attr_value, str):
                        # Normalize and index
                        normalized = attr_value.lower().strip()
                        if normalized not in index:
                            index[normalized] = []
                        index[normalized].append((entity_type, attr_name, entity_id))

                        # Also index individual words for partial matching
                        for word in normalized.split():
                            if len(word) > 2:  # Skip very short words
                                if word not in index:
                                    index[word] = []
                                index[word].append((entity_type, attr_name, entity_id))

        # Index relations
        for relation in extraction.relations:
            rel_type = relation.get("type", "")
            source = relation.get("source", "")
            target = relation.get("target", "")

            for value in [rel_type, source, target]:
                if value:
                    normalized = value.lower().strip()
                    if normalized not in index:
                        index[normalized] = []
                    index[normalized].append(
                        ("relation", rel_type, f"{source}->{target}")
                    )

        return index

    def _search_fact_in_extraction(
        self,
        fact: AnswerFact,
        index: Dict[str, List[Tuple[str, str, str]]],
    ) -> Tuple[bool, str]:
        """Search for a fact in the extraction index.

        Args:
            fact: The AnswerFact to search for
            index: The extraction index from _build_extraction_index

        Returns:
            Tuple of (found, location_description)
        """
        # Normalize search terms (convert to string first for numeric values)
        subject_norm = str(fact.subject).lower().strip()
        object_norm = str(fact.object).lower().strip()
        predicate_norm = str(fact.predicate).lower().strip()

        # Search strategies:
        # 1. Look for the object value directly
        if object_norm in index:
            locations = index[object_norm]
            for entity_type, attr, entity_id in locations:
                return True, f"{entity_type}.{attr} on {entity_id}"

        # 2. Look for object words
        object_words = [w for w in object_norm.split() if len(w) > 2]
        for word in object_words:
            if word in index:
                locations = index[word]
                for entity_type, attr, entity_id in locations:
                    # Check if this is a relevant match
                    return True, f"{entity_type}.{attr} on {entity_id} (partial)"

        # 3. Look for subject + predicate pattern
        predicate_matches = []
        for key, locations in index.items():
            for entity_type, attr, entity_id in locations:
                if predicate_norm in attr or attr in predicate_norm:
                    predicate_matches.append((entity_type, attr, entity_id))

        if predicate_matches:
            # Check if any predicate match is on the right subject
            for entity_type, attr, entity_id in predicate_matches:
                if subject_norm in entity_id.lower():
                    return True, f"{entity_type}.{attr} on {entity_id}"

        # 4. For relationship facts, check relations
        if fact.fact_type == "relationship":
            for key, locations in index.items():
                for loc_type, rel_type, details in locations:
                    if loc_type == "relation":
                        if (
                            subject_norm in details.lower()
                            and object_norm in details.lower()
                        ):
                            return True, f"relation {rel_type}: {details}"

        return False, ""

    def _find_matching_entity(
        self,
        new_entity: Dict[str, Any],
        existing_entities: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find an existing entity that matches a new entity.

        Matches by name or title attribute.

        Args:
            new_entity: The new entity to match
            existing_entities: List of existing entities

        Returns:
            Matching entity dict if found, None otherwise
        """
        new_name = new_entity.get("name") or new_entity.get("title")
        if not new_name:
            return None

        new_name_lower = new_name.lower().strip()

        for existing in existing_entities:
            existing_name = existing.get("name") or existing.get("title")
            if existing_name and existing_name.lower().strip() == new_name_lower:
                return existing

        return None

    # =========================================================================
    # Protected Methods - Formatting
    # =========================================================================

    def _format_entities_for_prompt(
        self, entities: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Format entities for inclusion in prompt.

        Args:
            entities: Entity dictionary from extraction

        Returns:
            Formatted string
        """
        lines: List[str] = []
        for entity_type, entity_list in entities.items():
            lines.append(f"  {entity_type}:")
            for entity in entity_list:
                attrs = ", ".join(f"{k}={v}" for k, v in entity.items())
                lines.append(f"    - {attrs}")

        return "\n".join(lines) if lines else "  (none)"

    def _format_relations_for_prompt(self, relations: List[Dict[str, Any]]) -> str:
        """Format relations for inclusion in prompt.

        Args:
            relations: Relations list from extraction

        Returns:
            Formatted string
        """
        if not relations:
            return "  (none)"

        lines: List[str] = []
        for rel in relations:
            rel_type = rel.get("type", "unknown")
            source = rel.get("source", "?")
            target = rel.get("target", "?")
            lines.append(f"  - {rel_type}: {source} -> {target}")

        return "\n".join(lines)


def run_validation(
    extractions: QAExtractionRegistry,
    api_cfg: DictConfig,
    global_cfg: DictConfig,
    retry_incomplete: bool = False,
) -> QAExtractionRegistry:
    """Convenience function to run validation.

    Args:
        extractions: QAExtractionRegistry to validate
        api_cfg: LLM API configuration
        global_cfg: Global configuration
        retry_incomplete: Whether to retry failed extractions

    Returns:
        Updated QAExtractionRegistry
    """
    validator = ExtractionValidator(api_cfg, global_cfg)
    return validator.validate_all(extractions, retry_incomplete=retry_incomplete)
