"""Entity type discoverer for identifying entity categories from QA pairs.

This module provides functionality to discover what types of entities
(e.g., person, work, organization) are present in QA pairs.
"""

import json
import logging
import re
from functools import cached_property
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from src.aligned_db.type_registry import EntityType
from src.generator.base import LLMComponent
from src.prompt.registry import (
    ENTITY_TYPE_CONSOLIDATION_PROMPT_REGISTRY,
    ENTITY_TYPE_DISCOVERY_PROMPT_REGISTRY,
)

logger = logging.getLogger("EntityTypeDiscoverer")


class EntityTypeDiscoverer(LLMComponent):
    """Discovers entity types from QA pairs (domain-agnostic).

    This class analyzes QA pairs to identify what categories of entities
    are being discussed (e.g., person, work, organization, award).

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the EntityTypeDiscoverer.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        super().__init__(api_cfg, global_cfg)

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def entity_type_discovery_prompt(self):
        """Get the entity type discovery prompt class."""
        prompt_name = self.global_cfg.prompt.get("entity_type_discovery", "default")
        return ENTITY_TYPE_DISCOVERY_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def entity_type_consolidation_prompt(self):
        """Get the entity type consolidation prompt class."""
        prompt_name = self.global_cfg.prompt.get("entity_type_consolidation", "default")
        return ENTITY_TYPE_CONSOLIDATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def use_llm_for_consolidation(self) -> bool:
        """Whether to use LLM for entity type consolidation."""
        return self.global_cfg.model.aligned_db.get(
            "use_llm_for_entity_type_consolidation", True
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def discover(
        self,
        qa_pairs: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        """Discover what entity types exist in the QA pairs.

        Note: This method processes all QA pairs at once. For large datasets,
        use discover_batch() with the batched pipeline in EntityPipeline.

        Args:
            qa_pairs: List of (question, answer) tuples to analyze

        Returns:
            List of entity type dictionaries, each with:
                - "name": Entity type name (e.g., "person")
                - "description": Brief description of the entity type
        """
        logger.info(f"Discovering entity types from {len(qa_pairs)} QA pairs")

        # Create prompt
        prompt = self.entity_type_discovery_prompt(qa_pairs_batch=qa_pairs)

        # Call LLM
        result = self._call_with_fallback(prompt)

        # Parse result
        entity_types = self._parse_entity_types(result)

        logger.info(
            f"Discovered {len(entity_types)} entity types: "
            f"{[et['name'] for et in entity_types]}"
        )

        return entity_types

    def discover_batch(
        self,
        qa_pairs_batch: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        """Discover entity types from a single batch of QA pairs.

        This method is designed to be called by the batched pipeline,
        processing one batch at a time.

        Args:
            qa_pairs_batch: Single batch of QA pairs (e.g., 20 pairs)

        Returns:
            List of entity type dicts discovered from this batch
        """
        logger.debug(
            f"Discovering entity types from batch of {len(qa_pairs_batch)} QA pairs"
        )

        # Create prompt
        prompt = self.entity_type_discovery_prompt(qa_pairs_batch=qa_pairs_batch)

        # Call LLM
        result = self._call_with_fallback(prompt)

        # Parse result
        entity_types = self._parse_entity_types(result)

        logger.debug(
            f"Batch discovered {len(entity_types)} entity types: "
            f"{[et['name'] for et in entity_types]}"
        )

        return entity_types

    def discover_all(
        self,
        qa_pairs: List[Tuple[str, str]],
        batch_size: int = 20,
        max_workers: int = 4,
    ) -> List[EntityType]:
        """Run full entity type discovery with parallel batching and normalization.

        This method orchestrates the complete Stage 1 entity type discovery:
        1. Splits QA pairs into batches
        2. Discovers entity types from each batch IN PARALLEL
        3. Consolidates and normalizes all discovered types ONCE at the end

        Args:
            qa_pairs: Full list of (question, answer) tuples
            batch_size: Number of QA pairs per batch (default 20)
            max_workers: Maximum parallel workers for batch processing (default 4)

        Returns:
            List of normalized EntityType objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            f"Starting full entity type discovery for {len(qa_pairs)} QA pairs "
            f"(batch_size={batch_size})"
        )

        # Split into batches
        batches: List[List[Tuple[str, str]]] = [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]
        logger.info(f"Split into {len(batches)} batches (parallel processing)")

        # Discover entity types from each batch IN PARALLEL
        batch_results: List[List[Dict[str, str]]] = []

        def process_batch(
            batch_idx: int, batch: List[Tuple[str, str]]
        ) -> List[Dict[str, str]]:
            """Process a single batch and return entity types."""
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} QA pairs)"
            )
            return self.discover_batch(batch)

        # Use ThreadPoolExecutor for parallel batch processing
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_types = future.result()
                    batch_results.append(batch_types)
                    logger.info(
                        f"Batch {batch_idx + 1}: Found {len(batch_types)} entity types"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")

        # Collect all entity types from all batches
        all_entity_types: List[Dict[str, str]] = []
        for batch_types in batch_results:
            all_entity_types.extend(batch_types)
        logger.info(
            f"Collected {len(all_entity_types)} entity types from {len(batch_results)} batches"
        )

        # Consolidate ALL at once (more efficient than incremental)
        logger.info("Running final consolidation pass...")
        consolidated = self.consolidate([], all_entity_types)

        # Convert to EntityType objects
        entity_types: List[EntityType] = []
        for et_dict in consolidated:
            entity_type = EntityType(
                name=et_dict.get("name", ""),
                description=et_dict.get("description", ""),
                aliases=et_dict.get("aliases", []),
                examples=et_dict.get("examples", []),
                is_junction=False,
            )
            entity_types.append(entity_type)

        logger.info(
            f"Entity type discovery complete: {len(entity_types)} types - "
            f"{[et.name for et in entity_types]}"
        )

        return entity_types

    def consolidate(
        self,
        existing_entity_types: List[Dict[str, str]],
        new_entity_types: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Consolidate entity types from multiple batches, merging duplicates.

        This method merges new entity types with existing canonical entity types,
        deduplicating and normalizing synonymous types.

        Args:
            existing_entity_types: Canonical entity types from previous batches
            new_entity_types: New entity types discovered from recent batches

        Returns:
            Consolidated list of unique entity types
        """
        if not new_entity_types:
            return existing_entity_types

        if not existing_entity_types:
            # First consolidation - just deduplicate new types
            return self._consolidate_with_heuristics([], new_entity_types)

        logger.info(
            f"Consolidating {len(new_entity_types)} new entity types "
            f"with {len(existing_entity_types)} existing types"
        )

        # First pass: heuristic deduplication
        pre_consolidated = self._consolidate_with_heuristics(
            existing_entity_types, new_entity_types
        )

        # If LLM consolidation is enabled and we have many types, use LLM
        if self.use_llm_for_consolidation and len(pre_consolidated) > 5:
            consolidated = self._consolidate_with_llm(
                existing_entity_types, new_entity_types
            )
            if consolidated:
                return consolidated

        return pre_consolidated

    # =========================================================================
    # Protected Methods - Consolidation
    # =========================================================================

    def _consolidate_with_llm(
        self,
        existing_entity_types: List[Dict[str, str]],
        new_entity_types: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Consolidate entity types using LLM.

        Args:
            existing_entity_types: Existing canonical entity types
            new_entity_types: New entity types to merge

        Returns:
            Consolidated list of entity types, or empty list on failure
        """
        logger.debug("Using LLM for entity type consolidation")

        try:
            # Create consolidation prompt
            prompt = self.entity_type_consolidation_prompt(
                existing_entity_types=existing_entity_types,
                new_entity_types=new_entity_types,
            )

            # Call LLM
            result = self._call_with_fallback(prompt)

            # Parse result
            consolidated = self._parse_entity_types(result)

            if consolidated:
                logger.info(
                    f"LLM consolidated to {len(consolidated)} entity types: "
                    f"{[et['name'] for et in consolidated]}"
                )
                return consolidated

        except Exception as e:
            logger.warning(f"LLM consolidation failed: {e}, falling back to heuristics")

        return []

    def _consolidate_with_heuristics(
        self,
        existing_entity_types: List[Dict[str, str]],
        new_entity_types: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Consolidate entity types using heuristic rules.

        Deduplicates by normalized name and merges descriptions.

        Args:
            existing_entity_types: Existing canonical entity types
            new_entity_types: New entity types to merge

        Returns:
            Consolidated list of unique entity types
        """
        # Start with existing types in a dict for easy lookup
        consolidated: Dict[str, Dict[str, Any]] = {}

        # Add existing types first
        for et in existing_entity_types:
            name = et.get("name", "").strip().lower()
            if name:
                canonical = self._canonicalize_entity_type(name)
                consolidated[canonical] = {
                    "name": canonical,
                    "description": et.get("description", ""),
                }

        # Merge new types
        for et in new_entity_types:
            name = et.get("name", "").strip().lower()
            if not name:
                continue

            canonical = self._canonicalize_entity_type(name)

            if canonical in consolidated:
                # Merge: keep longer/more detailed description
                existing_desc = consolidated[canonical].get("description", "")
                new_desc = et.get("description", "")
                if new_desc and len(new_desc) > len(existing_desc):
                    consolidated[canonical]["description"] = new_desc
            else:
                consolidated[canonical] = {
                    "name": canonical,
                    "description": et.get("description", ""),
                }

        return list(consolidated.values())

    def _canonicalize_entity_type(self, name: str) -> str:
        """Convert entity type name to canonical form.

        Args:
            name: Raw entity type name

        Returns:
            Canonical entity type name
        """
        # Common synonyms/variations mapping
        synonyms: Dict[str, str] = {
            # Person variations
            "people": "person",
            "individual": "person",
            "individuals": "person",
            "human": "person",
            "humans": "person",
            "author": "person",
            "authors": "person",
            "writer": "person",
            "writers": "person",
            "artist": "person",
            "artists": "person",
            "scientist": "person",
            "scientists": "person",
            "actor": "person",
            "actors": "person",
            "director": "person",
            "directors": "person",
            # Work variations
            "works": "work",
            "book": "work",
            "books": "work",
            "novel": "work",
            "novels": "work",
            "publication": "work",
            "publications": "work",
            "article": "work",
            "articles": "work",
            "movie": "work",
            "movies": "work",
            "film": "work",
            "films": "work",
            "song": "work",
            "songs": "work",
            "album": "work",
            "albums": "work",
            # Organization variations
            "organizations": "organization",
            "org": "organization",
            "orgs": "organization",
            "company": "organization",
            "companies": "organization",
            "institution": "organization",
            "institutions": "organization",
            "university": "organization",
            "universities": "organization",
            # Award variations
            "awards": "award",
            "prize": "award",
            "prizes": "award",
            "honor": "award",
            "honors": "award",
            # Location variations
            "locations": "location",
            "place": "location",
            "places": "location",
            "city": "location",
            "cities": "location",
            "country": "location",
            "countries": "location",
            # Event variations
            "events": "event",
        }

        # Clean the name
        cleaned = name.lower().strip().replace(" ", "_").replace("-", "_")

        # Check for direct mapping
        if cleaned in synonyms:
            return synonyms[cleaned]

        # Remove trailing 's' for simple plurals not in synonyms
        if cleaned.endswith("s") and len(cleaned) > 3:
            singular = cleaned[:-1]
            if singular in synonyms:
                return synonyms[singular]

        return cleaned

    # =========================================================================
    # Protected Methods - Parsing
    # =========================================================================

    def _parse_entity_types(self, response: str) -> List[Dict[str, str]]:
        """Parse entity types from LLM response.

        Args:
            response: Raw LLM response string

        Returns:
            List of entity type dictionaries
        """
        try:
            # Parse JSON response using base class method
            data = self._parse_json_response(response, repair_on_error=True)

            # Handle different response formats
            if isinstance(data, dict):
                if "entity_types" in data:
                    return data["entity_types"]
                # Single entity type dict
                return [data]
            elif isinstance(data, list):
                return data

            logger.warning(f"Unexpected response format: {type(data)}")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity types JSON: {e}")
            logger.error(f"Response was: {response}")
            # Try to extract entity types manually
            return self._extract_entity_types_fallback(response)

    def _extract_entity_types_fallback(self, response: str) -> List[Dict[str, str]]:
        """Fallback extraction of entity types from malformed response.

        Args:
            response: Raw response string

        Returns:
            List of entity type dictionaries extracted via heuristics
        """
        entity_types: List[Dict[str, str]] = []

        # Look for common patterns
        type_patterns = [
            r'"name":\s*"([^"]+)"',
            r"- (\w+):",
            r"\*\*(\w+)\*\*",
        ]

        for pattern in type_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                name = match.lower().strip()
                if name and name not in [et["name"] for et in entity_types]:
                    entity_types.append(
                        {
                            "name": name,
                            "description": "Entity type discovered from text",
                        }
                    )

        if not entity_types:
            # Default to common types
            logger.warning("Could not extract entity types, using defaults")
            entity_types = [
                {"name": "person", "description": "Individual people"},
                {
                    "name": "work",
                    "description": "Creative works, documents, or artifacts",
                },
                {
                    "name": "organization",
                    "description": "Companies, institutions, groups",
                },
            ]

        return entity_types
