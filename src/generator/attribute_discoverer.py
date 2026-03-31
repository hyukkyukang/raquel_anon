"""Attribute discoverer for identifying entity attributes from QA pairs.

This module provides functionality to discover what attributes/properties
are mentioned for each entity type in QA pairs.
"""

import json
import logging
from functools import cached_property
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from src.aligned_db.type_registry import AttributeType, EntityType
from src.generator.base import LLMComponent
from src.prompt.registry import ATTRIBUTE_DISCOVERY_PROMPT_REGISTRY
from src.utils import batches

logger = logging.getLogger("AttributeDiscoverer")


class AttributeDiscoverer(LLMComponent):
    """Discovers attributes for each entity type from QA pairs.

    This class analyzes QA pairs to identify what attributes/properties
    are mentioned for each entity type (e.g., birth_date, location for person).

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the AttributeDiscoverer.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        super().__init__(api_cfg, global_cfg)

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def attribute_discovery_prompt(self):
        """Get the attribute discovery prompt class."""
        prompt_name = self.global_cfg.prompt.get("attribute_discovery", "default")
        return ATTRIBUTE_DISCOVERY_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def batch_size(self) -> int:
        """Get the batch size for attribute discovery."""
        return self.global_cfg.model.aligned_db.get("attribute_batch_size", 20)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def discover(
        self,
        qa_pairs: List[Tuple[str, str]],
        entity_types: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Discover attributes for each entity type from QA pairs.

        Note: This method processes QA pairs in batches internally. For use with
        the batched pipeline in EntityPipeline, use discover_batch() instead.

        Args:
            qa_pairs: List of (question, answer) tuples to analyze
            entity_types: List of entity type dictionaries with name and description

        Returns:
            Dictionary mapping entity_type name to list of attribute dictionaries.
            Each attribute dict contains:
                - "name": Attribute name (e.g., "birth_date")
                - "data_type": Suggested data type (e.g., "DATE")
                - "description": Brief description
        """
        logger.info(
            f"Discovering attributes for {len(entity_types)} entity types "
            f"from {len(qa_pairs)} QA pairs"
        )

        all_attributes: Dict[str, List[Dict[str, Any]]] = {}

        # Process in batches
        for batch_idx, batch in enumerate(batches(qa_pairs, self.batch_size)):
            logger.info(f"Processing batch {batch_idx + 1} " f"({len(batch)} QA pairs)")

            # Create prompt
            prompt = self.attribute_discovery_prompt(
                entity_types=entity_types,
                qa_pairs_batch=batch,
            )

            # Call LLM
            result = self._call_with_fallback(prompt)

            # Parse result
            batch_attributes = self._parse_attributes(result)

            # Merge with all attributes
            all_attributes = self.merge_attributes(all_attributes, batch_attributes)

        logger.info(
            f"Discovered attributes: "
            + ", ".join(f"{et}: {len(attrs)}" for et, attrs in all_attributes.items())
        )

        return all_attributes

    def discover_batch(
        self,
        qa_pairs_batch: List[Tuple[str, str]],
        entity_types: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Discover attributes from a single batch of QA pairs.

        This method is designed to be called by the batched pipeline,
        processing one batch at a time without internal batching.

        Args:
            qa_pairs_batch: Single batch of QA pairs (e.g., 20 pairs)
            entity_types: Entity types to discover attributes for

        Returns:
            Dict mapping entity_type to list of attribute dicts
        """
        logger.debug(
            f"Discovering attributes from batch of {len(qa_pairs_batch)} QA pairs "
            f"for {len(entity_types)} entity types"
        )

        # Create prompt
        prompt = self.attribute_discovery_prompt(
            entity_types=entity_types,
            qa_pairs_batch=qa_pairs_batch,
        )

        # Call LLM
        result = self._call_with_fallback(prompt)

        # Parse result
        attributes = self._parse_attributes(result)

        logger.debug(
            f"Batch discovered attributes: "
            + ", ".join(f"{et}: {len(attrs)}" for et, attrs in attributes.items())
        )

        return attributes

    def discover_all(
        self,
        qa_pairs: List[Tuple[str, str]],
        entity_types: List[EntityType],
        batch_size: int = 20,
        max_workers: int = 4,
    ) -> Dict[str, List[AttributeType]]:
        """Run full attribute discovery with parallel batching.

        This method orchestrates the complete Stage 2 attribute discovery:
        1. Splits QA pairs into batches
        2. Discovers attributes from each batch IN PARALLEL
        3. Merges and deduplicates results
        4. Returns AttributeType objects

        Args:
            qa_pairs: Full list of (question, answer) tuples
            entity_types: List of EntityType objects from Stage 1
            batch_size: Number of QA pairs per batch (default 20)
            max_workers: Maximum parallel workers for batch processing (default 4)

        Returns:
            Dictionary mapping entity_type name to list of AttributeType objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            f"Starting full attribute discovery for {len(qa_pairs)} QA pairs "
            f"with {len(entity_types)} entity types (batch_size={batch_size})"
        )

        # Convert EntityType objects to dict format for existing methods
        entity_type_dicts: List[Dict[str, str]] = [
            {"name": et.name, "description": et.description} for et in entity_types
        ]

        # Split into batches
        batch_list: List[List[Tuple[str, str]]] = [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]
        logger.info(f"Split into {len(batch_list)} batches (parallel processing)")

        # Discover attributes from each batch IN PARALLEL
        all_attributes: Dict[str, List[Dict[str, Any]]] = {}
        batch_results: List[Dict[str, List[Dict[str, Any]]]] = []

        def process_batch(
            batch_idx: int, batch: List[Tuple[str, str]]
        ) -> Dict[str, List[Dict[str, Any]]]:
            """Process a single batch and return attributes."""
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batch_list)} ({len(batch)} QA pairs)"
            )
            return self.discover_batch(batch, entity_type_dicts)

        # Use ThreadPoolExecutor for parallel batch processing
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(batch_list))
        ) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batch_list)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_attrs = future.result()
                    batch_results.append(batch_attrs)
                    logger.info(
                        f"Batch {batch_idx + 1}: Completed with {sum(len(v) for v in batch_attrs.values())} attributes"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")

        # Merge all batch results
        for batch_attrs in batch_results:
            all_attributes = self.merge_attributes(all_attributes, batch_attrs)
        logger.info(
            f"Merged {len(batch_results)} batches, total types with attrs: {len(all_attributes)}"
        )

        # Convert to AttributeType objects
        result: Dict[str, List[AttributeType]] = {}

        for entity_type, attr_dicts in all_attributes.items():
            result[entity_type] = []
            for attr_dict in attr_dicts:
                # Infer natural key from naming conventions
                attr_name: str = attr_dict.get("name", "")
                is_natural_key: bool = attr_name in (
                    "name",
                    "full_name",
                    "title",
                    "label",
                    f"{entity_type}_name",
                )

                attr_type = AttributeType(
                    name=attr_name,
                    data_type=attr_dict.get("data_type", "TEXT"),
                    description=attr_dict.get("description", ""),
                    is_required=attr_dict.get("is_required", False),
                    is_unique=is_natural_key,  # Natural keys should be unique
                    is_natural_key=is_natural_key,
                    examples=attr_dict.get("examples", []),
                )
                result[entity_type].append(attr_type)

        logger.info(
            f"Attribute discovery complete: "
            + ", ".join(f"{et}: {len(attrs)}" for et, attrs in result.items())
        )

        return result

    def merge_attributes(
        self,
        existing: Dict[str, List[Dict[str, Any]]],
        new: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Merge new attributes into existing, avoiding duplicates.

        This method is designed to be called by the batched pipeline
        to combine attributes discovered from multiple batches.

        Args:
            existing: Existing attributes dictionary
            new: New attributes to merge

        Returns:
            Merged attributes dictionary
        """
        result = dict(existing)

        for entity_type, attrs in new.items():
            if entity_type not in result:
                result[entity_type] = []

            # Get existing attribute names for this entity type
            existing_names = {
                attr.get("name", "").lower() for attr in result[entity_type]
            }

            # Add new attributes that don't already exist
            for attr in attrs:
                attr_name = attr.get("name", "").lower()
                if attr_name and attr_name not in existing_names:
                    result[entity_type].append(attr)
                    existing_names.add(attr_name)

        return result

    # =========================================================================
    # Protected Methods
    # =========================================================================

    def _parse_attributes(self, response: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse attributes from LLM response.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary mapping entity_type to list of attribute dicts
        """
        try:
            # Parse JSON response using base class method
            data = self._parse_json_response(response, repair_on_error=True)

            # Handle different response formats
            if isinstance(data, dict):
                if "attributes" in data:
                    return data["attributes"]
                # Direct mapping
                return data

            logger.warning(f"Unexpected response format: {type(data)}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse attributes JSON: {e}")
            logger.error(f"Response was: {response}")
            return {}
