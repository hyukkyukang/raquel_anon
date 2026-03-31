"""Attribute normalizer for clustering and standardizing attributes.

This module provides functionality to normalize discovered attributes by
clustering similar attributes and creating canonical names. Supports multiple
normalization strategies: embedding-based (domain-agnostic), LLM-based, or hybrid.
"""

import json
import logging
import re
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from src.aligned_db.role_inference import apply_role_inference
from src.aligned_db.type_registry import (
    AttributeType,
    EntityType,
    RelationType,
    TypeRegistry,
)
from src.llm import JSONRepairer, LLMAPICaller, TooMuchThinkingError
from src.prompt.registry import ATTRIBUTE_NORMALIZATION_PROMPT_REGISTRY
from src.utils.json_utils import safe_json_loads

logger = logging.getLogger("AttributeNormalizer")


class AttributeNormalizer:
    """Normalizes discovered attributes by clustering similar ones.

    This class takes raw discovered attributes and clusters semantically
    similar ones (e.g., "birthplace", "hometown" → "birth_place") while
    creating canonical names and data types.

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the AttributeNormalizer.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def llm_api_caller(self) -> LLMAPICaller:
        """Get the primary LLM API caller."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.base,
        )

    @cached_property
    def fallback_llm_api_caller(self) -> LLMAPICaller:
        """Get the fallback LLM API caller for complex tasks."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.smart,
        )

    @cached_property
    def attribute_normalization_prompt(self):
        """Get the attribute normalization prompt class."""
        prompt_name = self.global_cfg.prompt.get("attribute_normalization", "default")
        return ATTRIBUTE_NORMALIZATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def use_llm_for_normalization(self) -> bool:
        """Whether to use LLM for normalization or just heuristics."""
        return self.global_cfg.model.aligned_db.get("use_llm_for_normalization", True)

    @cached_property
    def normalization_strategy(self) -> str:
        """Normalization strategy: 'embedding', 'llm', or 'hybrid'."""
        attr_norm_cfg = self.global_cfg.model.aligned_db.get("attribute_normalization", {})
        if isinstance(attr_norm_cfg, dict):
            return attr_norm_cfg.get("strategy", "hybrid")
        return "hybrid"

    @cached_property
    def embedding_similarity_threshold(self) -> float:
        """Similarity threshold for embedding-based clustering."""
        attr_norm_cfg = self.global_cfg.model.aligned_db.get("attribute_normalization", {})
        if isinstance(attr_norm_cfg, dict):
            return attr_norm_cfg.get("embedding_similarity_threshold", 0.85)
        return 0.85

    @cached_property
    def use_llm_refinement(self) -> bool:
        """Whether to refine embedding clusters with LLM in hybrid mode."""
        attr_norm_cfg = self.global_cfg.model.aligned_db.get("attribute_normalization", {})
        if isinstance(attr_norm_cfg, dict):
            return attr_norm_cfg.get("use_llm_refinement", True)
        return True

    @cached_property
    def max_attributes_per_entity(self) -> int:
        """Maximum attributes per entity type (prune excess)."""
        attr_norm_cfg = self.global_cfg.model.aligned_db.get("attribute_normalization", {})
        if isinstance(attr_norm_cfg, dict):
            return attr_norm_cfg.get("max_attributes_per_entity", 50)
        return 50

    @cached_property
    def embedding_normalizer(self):
        """Get the embedding-based attribute normalizer (lazy loaded)."""
        try:
            from src.generator.embedding_normalizer import EmbeddingAttributeNormalizer
            return EmbeddingAttributeNormalizer(
                similarity_threshold=self.embedding_similarity_threshold
            )
        except ImportError as e:
            logger.warning(
                f"Could not load EmbeddingAttributeNormalizer: {e}. "
                "Falling back to LLM-based normalization."
            )
            return None

    @cached_property
    def json_repairer(self) -> JSONRepairer:
        """Get the JSON repairer for fixing malformed JSON responses."""
        return JSONRepairer(self.api_cfg, self.global_cfg)

    # =========================================================================
    # Public Methods
    # =========================================================================

    def normalize(
        self,
        raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize raw attributes by clustering similar ones.

        Uses the configured normalization strategy:
        - 'embedding': Domain-agnostic semantic clustering using embeddings
        - 'llm': LLM-based normalization with semantic understanding
        - 'hybrid': Embeddings for initial clustering, LLM for refinement

        Args:
            raw_attributes: Dictionary mapping entity_type to list of raw attributes

        Returns:
            Dictionary mapping entity_type to list of normalized attribute dicts.
            Each normalized attribute contains:
                - "canonical_name": Standard attribute name
                - "data_type": Data type (TEXT, DATE, INTEGER, etc.)
                - "variants": List of original names that map to this canonical name
                - "description": Brief description
        """
        if not raw_attributes:
            logger.warning("No attributes to normalize")
            return {}

        logger.info(
            f"Normalizing attributes for {len(raw_attributes)} entity types "
            f"using strategy: {self.normalization_strategy}"
        )

        # Choose normalization strategy
        strategy = self.normalization_strategy.lower()

        if strategy == "embedding":
            normalized = self._normalize_with_embeddings(raw_attributes)
        elif strategy == "llm":
            if self.use_llm_for_normalization:
                normalized = self._normalize_with_llm(raw_attributes)
            else:
                normalized = self._normalize_with_heuristics(raw_attributes)
        elif strategy == "hybrid":
            normalized = self._normalize_hybrid(raw_attributes)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to hybrid")
            normalized = self._normalize_hybrid(raw_attributes)

        # Prune excess attributes per entity type
        normalized = self._prune_attributes(normalized)

        # Log summary
        for entity_type, attrs in normalized.items():
            logger.info(f"  {entity_type}: {len(attrs)} canonical attributes")

        return normalized

    def normalize_cumulative(
        self,
        accumulated_entity_types: List[Dict[str, str]],
        accumulated_attributes: Dict[str, List[Dict[str, Any]]],
        new_entity_types: List[Dict[str, str]],
        new_raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
        """Incrementally normalize new data against accumulated normalized data.

        This method is designed for the batched pipeline to normalize data
        cumulatively as it's discovered, rather than all at once.

        Args:
            accumulated_entity_types: Previously consolidated entity types
            accumulated_attributes: Previously normalized attributes
            new_entity_types: New entity types from recent batches
            new_raw_attributes: New raw attributes from recent batches

        Returns:
            Tuple of (consolidated_entity_types, merged_normalized_attributes)
        """
        logger.info(
            f"Cumulative normalization: {len(new_entity_types)} new entity types, "
            f"{sum(len(v) for v in new_raw_attributes.values())} new attributes"
        )

        # Step 1: Consolidate entity types
        consolidated_entity_types = self._consolidate_entity_types(
            accumulated_entity_types, new_entity_types
        )

        # Step 2: Normalize new attributes using configured strategy
        if new_raw_attributes:
            new_normalized = self.normalize(new_raw_attributes)
        else:
            new_normalized = {}

        # Step 3: Merge with accumulated attributes
        merged_attributes = self._merge_normalized_attributes(
            accumulated_attributes, new_normalized
        )

        # Step 4: Remap attributes to consolidated entity types
        # (e.g., if "author" merged into "person", move author's attributes to person)
        merged_attributes = self._remap_attributes_to_consolidated_types(
            merged_attributes, new_entity_types, consolidated_entity_types
        )

        logger.info(
            f"Cumulative normalization complete: "
            f"{len(consolidated_entity_types)} entity types, "
            f"{sum(len(v) for v in merged_attributes.values())} attributes"
        )

        return consolidated_entity_types, merged_attributes

    def build_type_registry(
        self,
        entity_types: List[EntityType],
        attributes: Dict[str, List[AttributeType]],
        relations: List[RelationType],
    ) -> TypeRegistry:
        """Build a TypeRegistry from discovered types, attributes, and relations.

        This method combines the outputs of Stage 1 (entity type discovery),
        Stage 2a (attribute discovery), and Stage 2b (relation discovery) into
        a unified TypeRegistry for schema generation.

        Args:
            entity_types: List of EntityType objects from discover_all()
            attributes: Dict mapping entity_type name to AttributeType lists
            relations: List of RelationType objects from relation discovery

        Returns:
            TypeRegistry containing all finalized types for schema generation
        """
        logger.info(
            f"Building TypeRegistry from {len(entity_types)} entity types, "
            f"{sum(len(v) for v in attributes.values())} attributes, "
            f"{len(relations)} relations"
        )

        registry = TypeRegistry.empty()

        # Add entity types
        for et in entity_types:
            registry.add_entity_type(et)

        # Add attributes for each entity type
        for entity_type_name, attr_list in attributes.items():
            for attr in attr_list:
                registry.add_attribute_type(entity_type_name, attr)

        # Add relations
        for relation in relations:
            registry.add_relation_type(relation)

        # Ensure natural keys are set for entities without them
        self._ensure_natural_keys(registry)
        apply_role_inference(registry)

        logger.info(f"Built {registry}")

        return registry

    def _ensure_natural_keys(self, registry: TypeRegistry) -> None:
        """Ensure each entity type has a natural key attribute.

        If an entity type doesn't have a natural key, try to find one
        using common naming patterns.

        Args:
            registry: TypeRegistry to update in place
        """
        natural_key_patterns: List[str] = [
            "full_name", "name", "title", "label", "code",
        ]

        for entity_type in registry.entity_types:
            # Skip junction tables
            if entity_type.is_junction:
                continue

            # Check if already has natural key
            existing_natural_key = registry.get_natural_key_for(entity_type.name)
            if existing_natural_key:
                continue

            # Try to find natural key from attributes
            attributes = registry.get_attributes_for(entity_type.name)

            # First, check entity-specific pattern (e.g., person_name for person)
            entity_specific_key = f"{entity_type.name}_name"
            for attr in attributes:
                if attr.name == entity_specific_key:
                    attr.is_natural_key = True
                    attr.is_unique = True
                    logger.debug(
                        f"Set {entity_specific_key} as natural key for {entity_type.name}"
                    )
                    break
            else:
                # Try common patterns
                for pattern in natural_key_patterns:
                    for attr in attributes:
                        if attr.name == pattern:
                            attr.is_natural_key = True
                            attr.is_unique = True
                            logger.debug(
                                f"Set {pattern} as natural key for {entity_type.name}"
                            )
                            break
                    else:
                        continue
                    break

    # =========================================================================
    # Protected Methods - Entity Type Consolidation
    # =========================================================================

    def _consolidate_entity_types(
        self,
        existing: List[Dict[str, str]],
        new: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Consolidate entity types, merging synonyms and duplicates.

        Args:
            existing: Existing canonical entity types
            new: New entity types to merge

        Returns:
            Consolidated list of unique entity types
        """
        if not new:
            return existing

        consolidated: Dict[str, Dict[str, str]] = {}

        # Add existing types
        for et in existing:
            name = et.get("name", "").strip().lower()
            if name:
                canonical = self._canonicalize_entity_type(name)
                consolidated[canonical] = {
                    "name": canonical,
                    "description": et.get("description", ""),
                }

        # Merge new types
        for et in new:
            name = et.get("name", "").strip().lower()
            if not name:
                continue

            canonical = self._canonicalize_entity_type(name)

            if canonical in consolidated:
                # Merge: keep longer description
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
        synonyms: Dict[str, str] = {
            # Person variations
            "people": "person",
            "individual": "person",
            "author": "person",
            "writer": "person",
            "artist": "person",
            "scientist": "person",
            # Work variations
            "works": "work",
            "book": "work",
            "books": "work",
            "novel": "work",
            "publication": "work",
            "article": "work",
            "movie": "work",
            "film": "work",
            # Organization variations
            "organizations": "organization",
            "company": "organization",
            "institution": "organization",
            # Award variations
            "awards": "award",
            "prize": "award",
            # Location variations
            "locations": "location",
            "place": "location",
            "city": "location",
            "country": "location",
            # Event variations
            "events": "event",
        }

        cleaned = name.lower().strip().replace(" ", "_").replace("-", "_")

        if cleaned in synonyms:
            return synonyms[cleaned]

        # Handle plurals
        if cleaned.endswith("s") and len(cleaned) > 3:
            singular = cleaned[:-1]
            if singular in synonyms:
                return synonyms[singular]

        return cleaned

    # =========================================================================
    # Protected Methods - Attribute Merging
    # =========================================================================

    def _merge_normalized_attributes(
        self,
        existing: Dict[str, List[Dict[str, Any]]],
        new: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Merge new normalized attributes with existing ones.

        Handles deduplication by canonical_name within each entity type.

        Args:
            existing: Existing normalized attributes
            new: New normalized attributes to merge

        Returns:
            Merged normalized attributes
        """
        result = {k: list(v) for k, v in existing.items()}

        for entity_type, attrs in new.items():
            if entity_type not in result:
                result[entity_type] = []

            # Get existing canonical names
            existing_canonical = {
                attr.get("canonical_name", "").lower() for attr in result[entity_type]
            }

            # Add new attributes that don't exist
            for attr in attrs:
                canonical_name = attr.get("canonical_name", "").lower()
                if canonical_name and canonical_name not in existing_canonical:
                    result[entity_type].append(attr)
                    existing_canonical.add(canonical_name)
                elif canonical_name in existing_canonical:
                    # Merge variants if same canonical name
                    for existing_attr in result[entity_type]:
                        if (
                            existing_attr.get("canonical_name", "").lower()
                            == canonical_name
                        ):
                            existing_variants = set(existing_attr.get("variants", []))
                            new_variants = set(attr.get("variants", []))
                            existing_attr["variants"] = list(
                                existing_variants | new_variants
                            )
                            break

        return result

    def _remap_attributes_to_consolidated_types(
        self,
        attributes: Dict[str, List[Dict[str, Any]]],
        original_entity_types: List[Dict[str, str]],
        consolidated_entity_types: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Remap attributes when entity types get consolidated.

        For example, if "author" is consolidated into "person", move all
        attributes from "author" to "person".

        Args:
            attributes: Current attribute mappings
            original_entity_types: Entity types before consolidation
            consolidated_entity_types: Entity types after consolidation

        Returns:
            Remapped attributes dictionary
        """
        # Build mapping from original names to canonical names
        type_mapping: Dict[str, str] = {}
        consolidated_names = {et["name"].lower() for et in consolidated_entity_types}

        for et in original_entity_types:
            original_name = et.get("name", "").lower()
            canonical = self._canonicalize_entity_type(original_name)
            if canonical in consolidated_names and original_name != canonical:
                type_mapping[original_name] = canonical

        # Remap attributes
        result: Dict[str, List[Dict[str, Any]]] = {}

        for entity_type, attrs in attributes.items():
            # Determine target entity type
            target_type = type_mapping.get(entity_type.lower(), entity_type.lower())

            if target_type not in result:
                result[target_type] = []

            # Get existing canonical names in target
            existing_canonical = {
                attr.get("canonical_name", "").lower() for attr in result[target_type]
            }

            # Add attributes, avoiding duplicates
            for attr in attrs:
                canonical_name = attr.get("canonical_name", "").lower()
                if canonical_name and canonical_name not in existing_canonical:
                    result[target_type].append(attr)
                    existing_canonical.add(canonical_name)

        return result

    # =========================================================================
    # Protected Methods - Normalization Implementation
    # =========================================================================

    def _normalize_with_llm(
        self,
        raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize attributes using LLM.

        Args:
            raw_attributes: Raw attributes dictionary

        Returns:
            Normalized attributes dictionary
        """
        # Create prompt
        prompt = self.attribute_normalization_prompt(raw_attributes=raw_attributes)

        # Call LLM
        result = self._call_with_fallback(prompt)

        # Parse result
        normalized = self._parse_normalized_attributes(result)

        # Ensure we have entries for all entity types
        for entity_type in raw_attributes:
            if entity_type not in normalized:
                # Use heuristic fallback for this entity type
                normalized[entity_type] = self._normalize_single_type_heuristic(
                    raw_attributes[entity_type]
                )

        return normalized

    def _normalize_with_heuristics(
        self,
        raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize attributes using rule-based heuristics.

        Args:
            raw_attributes: Raw attributes dictionary

        Returns:
            Normalized attributes dictionary
        """
        normalized: Dict[str, List[Dict[str, Any]]] = {}

        for entity_type, attrs in raw_attributes.items():
            normalized[entity_type] = self._normalize_single_type_heuristic(attrs)

        return normalized

    def _normalize_with_embeddings(
        self,
        raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize attributes using embedding-based semantic clustering.

        This is a domain-agnostic approach that uses sentence embeddings
        to cluster semantically similar attribute names.

        Args:
            raw_attributes: Raw attributes dictionary

        Returns:
            Normalized attributes dictionary
        """
        if self.embedding_normalizer is None:
            logger.warning(
                "Embedding normalizer not available, falling back to heuristics"
            )
            return self._normalize_with_heuristics(raw_attributes)

        normalized: Dict[str, List[Dict[str, Any]]] = {}

        for entity_type, attrs in raw_attributes.items():
            normalized_attrs = self.embedding_normalizer.normalize_attributes_for_entity(
                entity_type, attrs
            )
            normalized[entity_type] = normalized_attrs

        return normalized

    def _normalize_hybrid(
        self,
        raw_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize using hybrid approach: embeddings + optional LLM refinement.

        1. First pass: Use embeddings for initial semantic clustering
        2. Second pass: Optionally use LLM to refine canonical names

        Args:
            raw_attributes: Raw attributes dictionary

        Returns:
            Normalized attributes dictionary
        """
        # Step 1: Initial clustering with embeddings
        if self.embedding_normalizer is not None:
            logger.info("Step 1/2: Embedding-based clustering")
            normalized = self._normalize_with_embeddings(raw_attributes)
        else:
            logger.info("Embeddings not available, using heuristics for initial pass")
            normalized = self._normalize_with_heuristics(raw_attributes)

        # Step 2: Optional LLM refinement
        if self.use_llm_refinement and self.use_llm_for_normalization:
            logger.info("Step 2/2: LLM refinement of canonical names")
            try:
                llm_normalized = self._normalize_with_llm(raw_attributes)
                # Merge LLM results with embedding results
                # LLM is better at canonical naming, so prefer its names
                normalized = self._merge_embedding_and_llm_results(
                    normalized, llm_normalized
                )
            except Exception as e:
                logger.warning(f"LLM refinement failed: {e}, using embedding results")
        else:
            logger.debug("Skipping LLM refinement (disabled)")

        return normalized

    def _merge_embedding_and_llm_results(
        self,
        embedding_results: Dict[str, List[Dict[str, Any]]],
        llm_results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Merge embedding and LLM normalization results.

        LLM is generally better at selecting canonical names, while embeddings
        are better at finding semantic similarities. This merges the best of both.

        Args:
            embedding_results: Results from embedding-based normalization
            llm_results: Results from LLM-based normalization

        Returns:
            Merged normalized attributes
        """
        merged: Dict[str, List[Dict[str, Any]]] = {}

        all_entity_types = set(embedding_results.keys()) | set(llm_results.keys())

        for entity_type in all_entity_types:
            emb_attrs = embedding_results.get(entity_type, [])
            llm_attrs = llm_results.get(entity_type, [])

            if not emb_attrs:
                merged[entity_type] = llm_attrs
                continue
            if not llm_attrs:
                merged[entity_type] = emb_attrs
                continue

            # Build maps for quick lookup
            emb_by_variant: Dict[str, Dict] = {}
            for attr in emb_attrs:
                for variant in attr.get("variants", []):
                    emb_by_variant[variant.lower()] = attr

            llm_by_variant: Dict[str, Dict] = {}
            for attr in llm_attrs:
                for variant in attr.get("variants", []):
                    llm_by_variant[variant.lower()] = attr

            # Prefer LLM canonical names but use embedding clusters
            seen_canonical: set = set()
            merged_attrs: List[Dict[str, Any]] = []

            for emb_attr in emb_attrs:
                # Find corresponding LLM attribute by variant overlap
                best_llm_name = emb_attr.get("canonical_name")
                for variant in emb_attr.get("variants", []):
                    if variant.lower() in llm_by_variant:
                        llm_attr = llm_by_variant[variant.lower()]
                        best_llm_name = llm_attr.get("canonical_name", best_llm_name)
                        break

                if best_llm_name not in seen_canonical:
                    merged_attr = dict(emb_attr)
                    merged_attr["canonical_name"] = best_llm_name
                    merged_attrs.append(merged_attr)
                    seen_canonical.add(best_llm_name)

            merged[entity_type] = merged_attrs

        return merged

    def _prune_attributes(
        self,
        normalized: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Prune excess attributes per entity type.

        Args:
            normalized: Normalized attributes dictionary

        Returns:
            Pruned normalized attributes
        """
        max_attrs = self.max_attributes_per_entity

        pruned: Dict[str, List[Dict[str, Any]]] = {}

        for entity_type, attrs in normalized.items():
            if len(attrs) <= max_attrs:
                pruned[entity_type] = attrs
            else:
                # Sort by number of variants (more variants = more important)
                sorted_attrs = sorted(
                    attrs,
                    key=lambda a: len(a.get("variants", [])),
                    reverse=True,
                )
                pruned[entity_type] = sorted_attrs[:max_attrs]
                logger.info(
                    f"  {entity_type}: pruned {len(attrs)} -> {max_attrs} attributes"
                )

        return pruned

    def _normalize_single_type_heuristic(
        self,
        attrs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize attributes for a single entity type using heuristics.

        Args:
            attrs: List of raw attribute dictionaries

        Returns:
            List of normalized attribute dictionaries
        """
        # Group by canonical name
        canonical_groups: Dict[str, Dict[str, Any]] = {}

        for attr in attrs:
            name = attr.get("name", "").lower().strip()
            if not name:
                continue

            # Normalize the name
            canonical = self._canonicalize_attribute_name(name)

            if canonical in canonical_groups:
                # Merge variants
                canonical_groups[canonical]["variants"].append(name)
            else:
                canonical_groups[canonical] = {
                    "canonical_name": canonical,
                    "data_type": attr.get("data_type", "TEXT"),
                    "variants": [name],
                    "description": attr.get("description", ""),
                }

        return list(canonical_groups.values())

    def _canonicalize_attribute_name(self, name: str) -> str:
        """Convert an attribute name to canonical snake_case form.

        This method is domain-agnostic and performs only syntactic normalization.
        Semantic similarity (synonym handling) is delegated to embedding-based
        clustering or LLM-based normalization.

        Args:
            name: Raw attribute name

        Returns:
            Canonical attribute name in snake_case
        """
        # Clean the name - convert to snake_case
        cleaned = name.lower().strip()
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)

        # Remove special characters
        cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)

        # Remove consecutive underscores
        cleaned = re.sub(r"_+", "_", cleaned)

        # Remove common prefixes/suffixes that don't add meaning
        cleaned = re.sub(r"^(the_|a_|an_)", "", cleaned)
        cleaned = re.sub(r"(_name|_value|_text|_info|_data)$", "", cleaned)

        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")

        return cleaned if cleaned else "unknown"

    # =========================================================================
    # Protected Methods - LLM Interaction
    # =========================================================================

    def _call_with_fallback(self, prompt) -> str:
        """Call the LLM with fallback on TooMuchThinkingError.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response string
        """
        try:
            return self.llm_api_caller(
                prompt,
                post_process_fn=None,
                prefix="attribute_normalization",
            )
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking: {e}")
            logger.warning("Using fallback model...")
            return self.fallback_llm_api_caller(
                prompt,
                post_process_fn=None,
                prefix="attribute_normalization_fallback",
            )

    def _parse_normalized_attributes(
        self, response: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Parse normalized attributes from LLM response.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary mapping entity_type to list of normalized attribute dicts
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)

            # Use safe_json_loads with repair capability
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )

            if was_repaired:
                logger.info("JSON was repaired successfully")

            # Handle different response formats
            if isinstance(data, dict):
                if "normalized_attributes" in data:
                    return data["normalized_attributes"]
                # Direct mapping
                return data

            logger.warning(f"Unexpected response format: {type(data)}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse normalized attributes JSON: {e}")
            logger.error(f"Response was: {response}")
            return {}

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        # Check for ```json blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Check for ``` blocks
        code_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find JSON object directly
        json_obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_obj_match:
            return json_obj_match.group(0)

        # Return as-is
        return text
