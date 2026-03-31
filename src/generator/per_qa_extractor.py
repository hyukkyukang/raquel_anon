"""Per-QA value extractor for extracting entity values from individual QA pairs.

This module provides functionality to extract actual entity, attribute, and
relation values from QA pairs with parallel processing support.
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig
from tqdm import tqdm

from src.aligned_db.qa_extraction import QAExtraction, QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import PRIORITY_ATTRIBUTES, TypeRegistry
from src.generator.base import LLMComponent
from src.generator.extraction_metadata import (
    build_entity_attribute_metadata,
    build_relation_metadata,
)
from src.generator.extraction_quality import sanitize_extraction_for_quality
from src.generator.relation_normalization import (
    backfill_relation_entities,
    build_schema_backed_relation_registry,
    format_allowed_relation_names,
    normalize_extracted_relation,
)
from src.utils.async_utils import AsyncRateLimiter

logger = logging.getLogger("PerQAExtractor")


class PerQAExtractor(LLMComponent):
    """Extract entity/attribute/relation values per-QA with parallel processing.

    This class implements Stage 4 of the new pipeline, extracting actual values
    from individual QA pairs given the schema and type definitions.

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the PerQAExtractor.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        super().__init__(api_cfg, global_cfg)

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def max_concurrency(self) -> int:
        """Maximum number of concurrent extraction tasks."""
        return self.global_cfg.model.aligned_db.get("extraction_max_concurrency", 10)

    @cached_property
    def requests_per_second(self) -> float:
        """Maximum requests per second for rate limiting."""
        return self.global_cfg.model.aligned_db.get(
            "extraction_requests_per_second", 5.0
        )

    @cached_property
    def validate_work_titles(self) -> bool:
        """Whether to validate and filter generic work titles."""
        return self.global_cfg.model.aligned_db.get("validate_work_titles", True)

    @cached_property
    def generic_title_patterns(self) -> List[re.Pattern]:
        """Compiled regex patterns for generic work titles."""
        patterns = [
            r"^(novels?|books?|works?|writings?|stories?|articles?)\s+(by|of|from)\s+",
            r"^(dystopian|romance|mystery|fiction|fantasy|thriller|horror)\s+(novels?|books?|works?)",
            r"^(the\s+)?(novels?|books?|works?|writings?)\s+of\s+",
            r"'s\s+(novels?|books?|works?|writings?)$",
            r"^(all|some|several|many|few|various)\s+(novels?|books?|works?)",
            r"^(award[- ]winning|acclaimed|famous|popular)\s+(novels?|books?|works?)",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    # =========================================================================
    # Public Methods
    # =========================================================================

    async def extract_all(
        self,
        qa_pairs: List[Tuple[str, str]],
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
        max_concurrency: Optional[int] = None,
    ) -> QAExtractionRegistry:
        """Extract values from all QA pairs in parallel.

        This is the main entry point for Stage 4 extraction.

        Args:
            qa_pairs: List of (question, answer) tuples
            schema_registry: SchemaRegistry containing table definitions
            type_registry: TypeRegistry containing type information
            max_concurrency: Override default max concurrency (optional)

        Returns:
            QAExtractionRegistry containing all extractions
        """
        concurrency: int = max_concurrency or self.max_concurrency
        schema_backed_registry = build_schema_backed_relation_registry(
            schema_registry,
            type_registry,
        )

        logger.info(
            f"Starting parallel extraction for {len(qa_pairs)} QA pairs "
            f"(max_concurrency={concurrency}, rate={self.requests_per_second}/s)"
        )

        # Set up concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        rate_limiter = AsyncRateLimiter(rate=self.requests_per_second)
        executor = ThreadPoolExecutor(max_workers=concurrency)

        # Create extraction tasks with index tracking
        async def extract_single_async(
            idx: int, question: str, answer: str
        ) -> Tuple[int, QAExtraction]:
            async with semaphore:
                await rate_limiter.acquire()
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    executor,
                    self.extract_single,
                    idx,
                    question,
                    answer,
                    schema_registry,
                    schema_backed_registry,
                )
                return (idx, result)

        # Create tasks
        tasks = [
            asyncio.create_task(extract_single_async(i, q, a))
            for i, (q, a) in enumerate(qa_pairs)
        ]

        # Process results with progress bar
        registry = QAExtractionRegistry.empty()
        success_count: int = 0
        error_count: int = 0
        results: Dict[int, QAExtraction] = {}

        with tqdm(
            total=len(tasks),
            desc="Extracting QA pairs",
            unit="qa",
            ncols=100,
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    idx, extraction = await coro
                    results[idx] = extraction
                    success_count += 1
                    # Update progress bar postfix with success rate
                    pbar.set_postfix(ok=success_count, err=error_count, refresh=False)
                except Exception as e:
                    error_count += 1
                    # Find the failed task's index from exception context
                    logger.warning(f"Extraction failed: {type(e).__name__}")
                    pbar.set_postfix(ok=success_count, err=error_count, refresh=False)
                pbar.update(1)

        # Add results to registry in order, filling gaps with empty extractions
        for i, (q, a) in enumerate(qa_pairs):
            if i in results:
                registry.add(results[i])
            else:
                registry.add(
                    QAExtraction(
                        qa_index=i,
                        question=q,
                        answer=a,
                        validation_status="invalid",
                        missing_facts=["Extraction failed"],
                    )
                )

        logger.info(
            f"Extraction complete: {success_count}/{len(qa_pairs)} succeeded, "
            f"{error_count} failed"
        )

        executor.shutdown(wait=False)
        return registry

    def extract_single(
        self,
        qa_index: int,
        question: str,
        answer: str,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> QAExtraction:
        """Extract values from a single QA pair.

        Args:
            qa_index: Index of the QA pair
            question: The question text
            answer: The answer text
            schema_registry: SchemaRegistry containing table definitions
            type_registry: TypeRegistry containing type information

        Returns:
            QAExtraction containing extracted values
        """
        logger.debug(f"Extracting from QA {qa_index}")

        # Build prompt
        prompt_text = self._build_extraction_prompt(
            question, answer, schema_registry, type_registry
        )

        try:
            # Call LLM
            response = self._call_with_text_fallback(
                prompt_text, prefix="per_qa_extraction"
            )

            # Parse response
            extraction = self._parse_extraction_result(
                response, qa_index, question, answer, type_registry
            )

            logger.debug(
                f"QA {qa_index}: Extracted {extraction.entity_count} entities, "
                f"{extraction.relation_count} relations"
            )

            return extraction

        except Exception as e:
            logger.warning(f"Extraction failed for QA {qa_index}: {e}")
            return QAExtraction(
                qa_index=qa_index,
                question=question,
                answer=answer,
                validation_status="invalid",
                missing_facts=[f"Extraction error: {str(e)}"],
            )

    # =========================================================================
    # Protected Methods - Prompt Building
    # =========================================================================

    def _build_extraction_prompt(
        self,
        question: str,
        answer: str,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> str:
        """Build the extraction prompt for a single QA pair.

        Uses a hybrid format combining:
        - Column names and types from SchemaRegistry
        - Descriptions and examples from TypeRegistry
        - Relationship information for junction tables

        Args:
            question: The question text
            answer: The answer text
            schema_registry: SchemaRegistry for table structure
            type_registry: TypeRegistry for type information

        Returns:
            Prompt text for the LLM
        """
        # Build enriched schema representation
        schema_info = self._build_enriched_schema(schema_registry, type_registry)

        # Build relationship info
        relation_info = self._build_relation_info(type_registry)

        prompt = f"""Extract ALL entities, attributes, and relationships from this QA pair. Be EXHAUSTIVE.

## Schema (Tables and Columns)
{schema_info}

## Allowed Relationships (Junction Tables Only)
{relation_info if relation_info else "None defined"}

## QA Pair to Extract From
**Question**: {question}
**Answer**: {answer}

## Extraction Checklist - GO THROUGH EACH:

### 1. PERSON attributes (check ALL):
- name, birth_date, birth_place, nationality, heritage
- occupation, gender_identity
- **father_name, father_occupation** (from "his father who is a...")
- **mother_name, mother_occupation** (from "her mother...")
- current_residence, notable_works

### 2. WORK attributes (check ALL):
- title, author/creator, publication_date, genre
- **themes** (what it explores: diversity, leadership, identity...)
- **influences** (what inspired it: heritage, experiences, background...)
- reception (critical response, acclaim)

### 3. LOCATION entities:
- Any place mentioned → create location entity
- Also set birth_place/residence attribute on person

### 4. RELATIONSHIPS (check ALL):
- person wrote work → person_work relation
- person received award → person_award relation  
- work won award → work_award relation

## Critical Rules
- Extract from BOTH Question AND Answer text
- Format dates as YYYY-MM-DD
- **DO NOT SKIP parent info** - father_name/occupation are common misses
- **DO NOT SKIP themes/influences** - these are critical for verification
- Only use these exact relation types: {format_allowed_relation_names(type_registry)}
- Never invent semantic relation labels like "author_of" or "influenced_by"
- If a fact does not match an allowed junction-table relation, store it as an entity attribute instead
- When in doubt, extract it as an entity attribute, not a new relation
- Never create placeholder entities such as unnamed relatives, untitled works, unspecified books, or implied channels
- Never turn motivations, summaries, commentary, or whole clauses into standalone entities
- For theme/concept/channel values, use the shortest concrete noun phrase from the text
- If there is no clean entity name, keep the information as an attribute on an existing entity instead of inventing a new entity or relation

## Output Format
```json
{{
  "entities": {{
    "person": [{{"name": "...", "father_name": "...", "father_occupation": "...", ...}}],
    "work": [{{"title": "...", "themes": "...", "influences": "...", ...}}],
    "location": [{{"name": "...", ...}}]
  }},
  "relations": [
    {{"type": "person_work", "source": "PersonName", "target": "WorkTitle"}}
  ]
}}
```

Extract ALL data now:"""

        return prompt

    def _build_enriched_schema(
        self,
        schema_registry: SchemaRegistry,
        type_registry: TypeRegistry,
    ) -> str:
        """Build enriched schema representation with types, descriptions, examples.

        Combines SchemaRegistry (structure) with TypeRegistry (semantics) for
        maximum extraction context.

        Args:
            schema_registry: SchemaRegistry for column names and types
            type_registry: TypeRegistry for descriptions and examples

        Returns:
            Formatted schema string
        """
        lines: List[str] = []

        for entity_type in type_registry.entity_types:
            table = schema_registry.get_table(entity_type.name)
            if not table:
                continue

            # Table header with description
            desc = f" - {entity_type.description}" if entity_type.description else ""
            lines.append(f"### {entity_type.name}{desc}")

            # Get attribute metadata from TypeRegistry
            attr_meta: Dict[str, Any] = {}
            for attr in type_registry.get_attributes_for(entity_type.name):
                attr_meta[attr.name] = attr

            # Get priority attributes for this entity type
            priority_attrs = set(PRIORITY_ATTRIBUTES.get(entity_type.name, []))

            # List columns with type and metadata
            lines.append("Columns:")
            for col in table.columns:
                # Skip internal columns (PK, FK IDs)
                if col.is_primary_key or col.name.endswith("_id"):
                    continue

                # Build column description
                col_parts: List[str] = [f"  - {col.name} ({col.data_type}"]

                # Add constraints
                if col.is_unique:
                    col_parts.append(", UNIQUE")
                col_parts.append(")")

                # Mark priority attributes
                if col.name in priority_attrs:
                    col_parts.append(" [PRIORITY]")

                # Add description from TypeRegistry if available
                meta = attr_meta.get(col.name)
                if meta:
                    if meta.description:
                        col_parts.append(f" - {meta.description}")
                    if meta.examples:
                        examples_str = ", ".join(f'"{e}"' for e in meta.examples[:2])
                        col_parts.append(f" e.g., {examples_str}")

                lines.append("".join(col_parts))

            lines.append("")  # Blank line between tables

        return "\n".join(lines)

    def _build_relation_info(self, type_registry: TypeRegistry) -> str:
        """Build relationship information for junction tables.

        Args:
            type_registry: TypeRegistry containing relation types

        Returns:
            Formatted relationship string
        """
        if not type_registry.relation_types:
            return ""

        lines: List[str] = []
        for rt in type_registry.relation_types:
            # Basic relationship info
            desc = f" - {rt.description}" if rt.description else ""
            lines.append(f"- {rt.name}: {rt.source_entity} ↔ {rt.target_entity}{desc}")

            # Additional attributes on junction table
            if rt.attributes:
                attr_names = [a.name for a in rt.attributes]
                lines.append(f"  Additional columns: {', '.join(attr_names)}")

        return "\n".join(lines)

    # =========================================================================
    # Protected Methods - Validation
    # =========================================================================

    def _validate_work_entities(
        self,
        extraction: QAExtraction,
        type_registry: TypeRegistry,
    ) -> QAExtraction:
        """Validate and filter work entities to remove generic titles.

        Generic titles like "Novels by X" or "Dystopian works" are not actual
        work names and should be converted to person attributes instead.

        Args:
            extraction: QAExtraction to validate

        Returns:
            Validated QAExtraction with generic works filtered/converted
        """
        if not self.validate_work_titles:
            return extraction

        if "work" not in extraction.entities:
            return extraction

        valid_works: List[Dict[str, Any]] = []
        generic_works: List[Dict[str, Any]] = []
        valid_metadata: List[Dict[str, Dict[str, Any]]] = []
        work_metadata = extraction.entity_attribute_metadata.get("work", [])

        for idx, work in enumerate(extraction.entities.get("work", [])):
            title = work.get("title", "") or work.get("name", "")

            if not title:
                continue

            is_generic = self._is_generic_work_title(title)

            if is_generic:
                generic_works.append(work)
                logger.debug(f"Filtered generic work title: '{title}'")
            else:
                valid_works.append(work)
                valid_metadata.append(
                    work_metadata[idx] if idx < len(work_metadata) else {}
                )

        # Update extraction with only valid works
        if len(generic_works) > 0:
            extraction.entities["work"] = valid_works
            extraction.entity_attribute_metadata["work"] = valid_metadata

            # Convert generic work info to person's notable_works attribute
            if generic_works and "person" in extraction.entities:
                generic_titles = [
                    w.get("title", "") or w.get("name", "")
                    for w in generic_works
                ]
                self._add_notable_works_to_persons(
                    extraction,
                    generic_titles,
                    type_registry,
                )

            logger.debug(
                f"QA {extraction.qa_index}: Filtered {len(generic_works)} generic works, "
                f"kept {len(valid_works)} valid works"
            )

        return extraction

    def _is_generic_work_title(self, title: str) -> bool:
        """Check if a work title is generic (not a real work name).

        Args:
            title: The work title to check

        Returns:
            True if the title appears to be generic
        """
        title_lower = title.lower().strip()

        # Check against pre-compiled patterns
        for pattern in self.generic_title_patterns:
            if pattern.search(title_lower):
                return True

        # Additional heuristics:
        # Very short titles with generic words
        generic_words = {"book", "novel", "work", "story", "writing", "article"}
        words = set(title_lower.split())
        if len(words) <= 2 and words & generic_words:
            return True

        return False

    def _add_notable_works_to_persons(
        self,
        extraction: QAExtraction,
        generic_titles: List[str],
        type_registry: TypeRegistry,
    ) -> None:
        """Add generic work descriptions to person's notable_works attribute.

        Args:
            extraction: QAExtraction to modify
            generic_titles: List of generic work descriptions
        """
        if not generic_titles:
            return

        # Combine generic titles into a notable_works description
        works_desc = "; ".join(generic_titles)

        for idx, person in enumerate(extraction.entities.get("person", [])):
            existing = person.get("notable_works", "")
            if existing:
                person["notable_works"] = f"{existing}; {works_desc}"
            else:
                person["notable_works"] = works_desc
            extraction.set_entity_attribute_metadata(
                "person",
                idx,
                "notable_works",
                build_entity_attribute_metadata(
                    "person",
                    {"notable_works": person["notable_works"]},
                    type_registry,
                )["notable_works"],
            )

    # =========================================================================
    # Protected Methods - Response Parsing
    # =========================================================================

    def _parse_extraction_result(
        self,
        response: str,
        qa_index: int,
        question: str,
        answer: str,
        type_registry: TypeRegistry,
    ) -> QAExtraction:
        """Parse extraction result from LLM response.

        Args:
            response: Raw LLM response string
            qa_index: Index of the QA pair
            question: The question text
            answer: The answer text

        Returns:
            QAExtraction object
        """
        extraction = QAExtraction(
            qa_index=qa_index,
            question=question,
            answer=answer,
        )

        try:
            data = self._parse_json_response(response, repair_on_error=True)

            if not isinstance(data, dict):
                logger.warning(f"QA {qa_index}: Expected dict, got {type(data)}")
                return extraction

            # Parse entities
            entities_data = data.get("entities", {})
            if isinstance(entities_data, dict):
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

            # Parse relations
            relations_data = data.get("relations", [])
            if isinstance(relations_data, list):
                seen_relations: Set[Tuple[str, str, str]] = set()
                for relation in relations_data:
                    if isinstance(relation, dict):
                        normalized_relation = normalize_extracted_relation(
                            relation,
                            type_registry,
                        )
                        if normalized_relation is None:
                            continue

                        relation_key = (
                            normalized_relation.get("type", ""),
                            normalized_relation.get("source", ""),
                            normalized_relation.get("target", ""),
                        )
                        if relation_key in seen_relations:
                            continue
                        seen_relations.add(relation_key)
                        extraction.add_relation(
                            normalized_relation,
                            build_relation_metadata(
                                relation,
                                normalized_relation,
                                type_registry,
                            ),
                        )

            # Validate work entities (filter generic titles)
            extraction = self._validate_work_entities(extraction, type_registry)
            sanitize_extraction_for_quality(
                extraction,
                type_registry,
                is_generic_work_title=(
                    self._is_generic_work_title if self.validate_work_titles else None
                ),
            )

            extraction, added_entities, swapped_relations = backfill_relation_entities(
                extraction,
                type_registry,
                is_generic_work_title=(
                    self._is_generic_work_title if self.validate_work_titles else None
                ),
            )
            if added_entities or swapped_relations:
                logger.debug(
                    "QA %d: backfilled %d relation endpoint entities, swapped %d relations",
                    qa_index,
                    added_entities,
                    swapped_relations,
                )

            sanitize_extraction_for_quality(
                extraction,
                type_registry,
                is_generic_work_title=(
                    self._is_generic_work_title if self.validate_work_titles else None
                ),
            )

            # Update relevant tables
            extraction.update_relevant_tables()

            # Set confidence based on extraction results
            extraction.extraction_confidence = (
                0.9 if extraction.entity_count > 0 else 0.3
            )

            return extraction

        except Exception as e:
            logger.warning(f"QA {qa_index}: Failed to parse response: {e}")
            extraction.validation_status = "invalid"
            extraction.missing_facts.append(f"Parse error: {str(e)}")
            return extraction


def run_extraction(
    qa_pairs: List[Tuple[str, str]],
    schema_registry: SchemaRegistry,
    type_registry: TypeRegistry,
    api_cfg: DictConfig,
    global_cfg: DictConfig,
) -> QAExtractionRegistry:
    """Convenience function to run extraction synchronously.

    Args:
        qa_pairs: List of (question, answer) tuples
        schema_registry: SchemaRegistry containing table definitions
        type_registry: TypeRegistry containing type information
        api_cfg: LLM API configuration
        global_cfg: Global configuration

    Returns:
        QAExtractionRegistry containing all extractions
    """
    extractor = PerQAExtractor(api_cfg, global_cfg)
    return asyncio.run(extractor.extract_all(qa_pairs, schema_registry, type_registry))
