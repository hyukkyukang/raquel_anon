"""Relation discoverer for identifying many-to-many relationships from QA pairs.

This module provides functionality to discover what many-to-many relationships
exist between entity types (e.g., person-work, person-award).
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from omegaconf import DictConfig

from src.aligned_db.type_registry import AttributeType, EntityType, RelationType
from src.generator.base import LLMComponent
from src.generator.relation_normalization import normalize_discovered_relation
from src.utils.json_utils import extract_json_from_response, safe_json_loads

logger = logging.getLogger("RelationDiscoverer")


class RelationDiscoverer(LLMComponent):
    """Discovers many-to-many relationships between entity types.

    This class analyzes QA pairs to identify what relationships exist
    between entity types, which will become junction tables in the schema.

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the RelationDiscoverer.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        super().__init__(api_cfg, global_cfg)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def discover_all(
        self,
        qa_pairs: List[Tuple[str, str]],
        entity_types: List[EntityType],
        batch_size: int = 30,
        max_workers: int = 4,
    ) -> List[RelationType]:
        """Discover all many-to-many relationships from QA pairs.

        This method orchestrates relation discovery:
        1. Splits QA pairs into batches
        2. Discovers relations from each batch IN PARALLEL
        3. Merges and deduplicates results

        Args:
            qa_pairs: Full list of (question, answer) tuples
            entity_types: List of EntityType objects from Stage 1
            batch_size: Number of QA pairs per batch
            max_workers: Maximum parallel workers for batch processing (default 4)

        Returns:
            List of RelationType objects representing junction tables
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            f"Starting relation discovery for {len(qa_pairs)} QA pairs "
            f"with {len(entity_types)} entity types"
        )

        # Get entity type names
        entity_names: List[str] = [et.name for et in entity_types]

        # Split into batches
        batches_list: List[List[Tuple[str, str]]] = [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]
        logger.info(f"Split into {len(batches_list)} batches (parallel processing)")

        # Discover from each batch IN PARALLEL
        batch_results: List[List[RelationType]] = []

        def process_batch(
            batch_idx: int, batch: List[Tuple[str, str]]
        ) -> List[RelationType]:
            """Process a single batch and return relations."""
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches_list)} ({len(batch)} QA pairs)"
            )
            return self._discover_batch(batch, entity_names)

        # Use ThreadPoolExecutor for parallel batch processing
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(batches_list))
        ) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches_list)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_rels = future.result()
                    batch_results.append(batch_rels)
                    logger.info(
                        f"Batch {batch_idx + 1}: Completed with {len(batch_rels)} relations"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")

        # Merge all batch results
        all_relations: Dict[str, RelationType] = {}
        for batch_rels in batch_results:
            for rel in batch_rels:
                if rel.name not in all_relations:
                    all_relations[rel.name] = rel
                else:
                    # Merge examples
                    existing = all_relations[rel.name]
                    existing.examples.extend(rel.examples)

        # Also infer common relations from entity types
        inferred_relations = self._infer_common_relations(entity_types)
        for rel in inferred_relations:
            if rel.name not in all_relations:
                all_relations[rel.name] = rel

        result: List[RelationType] = list(all_relations.values())

        logger.info(
            f"Relation discovery complete: {len(result)} relations - "
            f"{[r.name for r in result]}"
        )

        return result

    def _discover_batch(
        self,
        qa_pairs_batch: List[Tuple[str, str]],
        entity_names: List[str],
    ) -> List[RelationType]:
        """Discover relations from a single batch.

        Args:
            qa_pairs_batch: Batch of QA pairs
            entity_names: List of entity type names

        Returns:
            List of RelationType objects
        """
        # Build prompt
        qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs_batch)

        entity_list = ", ".join(entity_names)
        pair_examples = ", ".join(
            sorted(
                {
                    f"{name_a}_{name_b}"
                    for idx, name_a in enumerate(entity_names)
                    for name_b in entity_names[idx + 1 :]
                }
            )[:20]
        )

        prompt_text = f"""Analyze the following QA pairs and identify any many-to-many relationships between these entity types: {entity_list}

QA Pairs:
{qa_text}

For each relationship found, provide:
- name: The junction table name using ONLY entity type names from the list above
- source_entity: First entity type
- target_entity: Second entity type
- description: Brief description of the relationship
- additional_attributes: Any attributes on the relationship itself (e.g., "year" for an award relationship)

Return JSON array:
```json
[
  {{
    "name": "person_work",
    "source_entity": "person",
    "target_entity": "work",
    "description": "Associates people with works they created or contributed to",
    "additional_attributes": ["role", "contribution_type"]
  }}
]
```

Rules:
- Only include relationships where BOTH entity types exist
- The relation name must be a pair-based junction-table name, not a semantic predicate
- Valid examples of pair-based names: {pair_examples or "person_work, person_award"}
- Invalid examples: author_of, influenced_by, translated_into_language
- If the relation is really an attribute or one-to-many fact, do not include it

Return empty array [] if no relationships found."""

        try:
            response = self._call_with_text_fallback(
                prompt_text, prefix="relation_discovery"
            )
            return self._parse_relations(response, entity_names)
        except Exception as e:
            logger.warning(f"Relation discovery batch failed: {e}")
            return []

    def _parse_relations(
        self,
        response: str,
        entity_names: List[str],
    ) -> List[RelationType]:
        """Parse relation types from LLM response.

        Args:
            response: Raw LLM response string
            entity_names: Allowed entity types from stage 1

        Returns:
            List of RelationType objects
        """
        try:
            json_str = extract_json_from_response(response)
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )

            if was_repaired:
                logger.info("JSON was repaired successfully")

            if not isinstance(data, list):
                if isinstance(data, dict) and "relations" in data:
                    data = data["relations"]
                else:
                    data = [data] if data else []

            # Ensure data is a list after extraction
            if not isinstance(data, list):
                logger.debug(f"Expected list, got {type(data).__name__}, wrapping")
                data = [data] if isinstance(data, dict) else []

            relations: List[RelationType] = []
            allowed_entity_names: Set[str] = {name.strip().lower() for name in entity_names}
            preferred_relations = self._infer_common_relations(
                [EntityType(name=name) for name in entity_names]
            )
            for item in data:
                if not isinstance(item, dict):
                    continue

                relation = normalize_discovered_relation(
                    item,
                    allowed_entity_names=allowed_entity_names,
                    preferred_relations=preferred_relations,
                )
                if relation is not None:
                    relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"Failed to parse relations: {e}")
            return []

    def _infer_common_relations(
        self,
        entity_types: List[EntityType],
    ) -> List[RelationType]:
        """Infer common many-to-many relationships from entity type names.

        This provides a fallback to ensure common relationships are captured
        even if the LLM misses them.

        Args:
            entity_types: List of EntityType objects

        Returns:
            List of inferred RelationType objects
        """
        entity_names: Set[str] = {et.name for et in entity_types}
        inferred: List[RelationType] = []

        # Common relationship patterns
        common_patterns: List[Dict[str, Any]] = [
            {
                "requires": ["person", "work"],
                "relation": RelationType(
                    name="person_work",
                    source_entity="person",
                    target_entity="work",
                    description="Associates people with works they created or contributed to",
                    attributes=[
                        AttributeType(
                            name="role",
                            data_type="TEXT",
                            description="Role in the work",
                        ),
                    ],
                ),
            },
            {
                "requires": ["person", "award"],
                "relation": RelationType(
                    name="person_award",
                    source_entity="person",
                    target_entity="award",
                    description="Awards received by people",
                    attributes=[
                        AttributeType(
                            name="year",
                            data_type="INTEGER",
                            description="Year award was received",
                        ),
                    ],
                ),
            },
            {
                "requires": ["person", "organization"],
                "relation": RelationType(
                    name="person_organization",
                    source_entity="organization",
                    target_entity="person",
                    description="Affiliations between people and organizations",
                    attributes=[
                        AttributeType(
                            name="role",
                            data_type="TEXT",
                            description="Role in organization",
                        ),
                    ],
                ),
            },
            {
                "requires": ["work", "genre"],
                "relation": RelationType(
                    name="work_genre",
                    source_entity="genre",
                    target_entity="work",
                    description="Genres associated with works",
                ),
            },
            {
                "requires": ["work", "theme"],
                "relation": RelationType(
                    name="work_theme",
                    source_entity="theme",
                    target_entity="work",
                    description="Themes present in works",
                ),
            },
            {
                "requires": ["person", "occupation"],
                "relation": RelationType(
                    name="person_occupation",
                    source_entity="occupation",
                    target_entity="person",
                    description="Occupations held by people",
                ),
            },
            {
                "requires": ["work", "award"],
                "relation": RelationType(
                    name="work_award",
                    source_entity="award",
                    target_entity="work",
                    description="Awards received by works",
                    attributes=[
                        AttributeType(
                            name="year",
                            data_type="INTEGER",
                            description="Year award was received",
                        ),
                    ],
                ),
            },
        ]

        for pattern in common_patterns:
            required: List[str] = pattern["requires"]
            if all(req in entity_names for req in required):
                inferred.append(pattern["relation"])

        return inferred
