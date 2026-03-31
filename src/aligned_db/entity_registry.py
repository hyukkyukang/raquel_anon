"""Domain-agnostic entity registry for storing extracted entities.

This module provides a flexible container for entities extracted from QA pairs,
supporting any domain without hardcoded entity types like Author, Book, etc.
It also tracks many-to-many relationships between entities via junction tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from src.aligned_db.qa_extraction import QAExtractionRegistry

logger = logging.getLogger("EntityRegistry")

# Attributes that can have multiple values and should be combined when merging
# These are typically semicolon-separated in extraction
LIST_LIKE_ATTRS: Set[str] = {
    "themes",
    "awards_received",
    "notable_works",
    "genres",
    "influences",
    "collaborations",
    "translations",
    "languages",
    "characters",
    "awards",
    "works",
    "publications",
}


@dataclass
class EntityRegistry:
    """Domain-agnostic registry of extracted entities with relationship tracking.

    Structure: Dict mapping entity_type -> List of entity instances
    Each entity instance is a Dict[str, Any] with flexible attributes.

    Relationships are stored separately for junction tables (many-to-many).

    Example:
        entities = {
            "person": [
                {"name": "John Doe", "birth_date": "1990-01-01"},
                {"name": "Jane Doe", "occupation": "engineer"}
            ],
            "work": [
                {"title": "My Book", "creator": "John Doe"}
            ]
        }
        relationships = {
            "person_work": [
                {"person_id": 1, "work_id": 1, "person_name": "John Doe", "work_title": "My Book"}
            ]
        }

    Attributes:
        entities: Dictionary mapping entity type names to lists of entity instances
        relationships: Dictionary mapping junction table names to lists of relationship records
    """

    entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relationships: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls) -> "EntityRegistry":
        """Create an empty EntityRegistry.

        Returns:
            New EntityRegistry with no entities or relationships
        """
        return cls(entities={}, relationships={})

    @classmethod
    def from_json(cls, json_str: str) -> "EntityRegistry":
        """Create an EntityRegistry from a JSON string.

        Args:
            json_str: JSON string representing entities

        Returns:
            New EntityRegistry populated from JSON
        """
        data = json.loads(json_str)
        if isinstance(data, dict):
            # Handle both {"entities": {...}, "relationships": {...}} and direct {...} formats
            if "entities" in data:
                return cls(
                    entities=data["entities"],
                    relationships=data.get("relationships", {}),
                )
            return cls(entities=data, relationships={})
        return cls.empty()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRegistry":
        """Create an EntityRegistry from a dictionary.

        Args:
            data: Dictionary with entity data

        Returns:
            New EntityRegistry populated from dictionary
        """
        raw_entities = data.get("entities", data)
        relationships = data.get("relationships", {}) if "entities" in data else {}

        # Validate and filter entities: each value must be a list of dicts
        validated_entities: Dict[str, List[Dict[str, Any]]] = {}
        for entity_type, entity_list in raw_entities.items():
            if entity_type == "relationships":
                continue  # Skip if accidentally included
            if not isinstance(entity_list, list):
                logger.warning(
                    f"Skipping entity type '{entity_type}': expected list, got {type(entity_list).__name__}"
                )
                continue
            # Filter to only include dict items
            valid_items = [e for e in entity_list if isinstance(e, dict)]
            if len(valid_items) < len(entity_list):
                logger.warning(
                    f"Filtered {len(entity_list) - len(valid_items)} non-dict items from '{entity_type}'"
                )
            if valid_items:
                validated_entities[entity_type] = valid_items

        return cls(entities=validated_entities, relationships=relationships)

    @classmethod
    def from_qa_extractions(
        cls,
        qa_extractions: "QAExtractionRegistry",
    ) -> "EntityRegistry":
        """Create an EntityRegistry from a QAExtractionRegistry.

        This factory method merges all entities and relations from
        per-QA extractions into a single EntityRegistry.

        Args:
            qa_extractions: QAExtractionRegistry containing per-QA extractions

        Returns:
            New EntityRegistry with all merged entities and relationships
        """
        logger.info(
            f"Creating EntityRegistry from {qa_extractions.count} QA extractions"
        )

        # Merge all entities
        merged_entities: Dict[str, List[Dict[str, Any]]] = qa_extractions.merge_entities()

        # Merge all relations and group by junction table type. Preserve
        # per-relation metadata so endpoint entity types are not lost when the
        # junction-table name is alphabetical or contains multi-word entity
        # types.
        merged_relations: Dict[str, List[Dict[str, Any]]] = {}
        for extraction in qa_extractions:
            for idx, relation in enumerate(extraction.relations):
                rel_type = relation.get("type", "unknown_relation")
                if rel_type not in merged_relations:
                    merged_relations[rel_type] = []

                record = cls._convert_relation_to_named_format(
                    relation,
                    rel_type,
                    relation_metadata=extraction.get_relation_metadata(idx),
                )
                merged_relations[rel_type].append(record)

        registry = cls(entities=merged_entities, relationships=merged_relations)

        logger.info(
            f"Merged EntityRegistry: {len(merged_entities)} entity types, "
            f"{sum(len(v) for v in merged_entities.values())} total entities, "
            f"{len(merged_relations)} relationship types"
        )

        return registry

    @staticmethod
    def _convert_relation_to_named_format(
        relation: Dict[str, Any],
        junction_table: str,
        *,
        relation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert source/target relation format to {entity_type}_name format.

        The per-QA extractor outputs relations as:
            {"type": "person_work", "source": "John", "target": "Book Title", "attr": "..."}

        The junction table upsert logic expects:
            {"person_name": "John", "work_name": "Book Title", ...}

        This method performs that conversion.

        Args:
            relation: The raw relation dict with source/target keys
            junction_table: The junction table name (e.g., "person_work")

        Returns:
            Dict with {entity_type}_name keys instead of source/target
        """
        source_val: Any = relation.get("source")
        target_val: Any = relation.get("target")
        relation_metadata = relation_metadata or {}

        # Parse junction table name to get entity types (alphabetically sorted)
        parts: List[str] = junction_table.split("_")

        # Build the converted record
        record: Dict[str, Any] = {}

        # First, preserve any explicit {entity_type}_name fields.
        for key, val in relation.items():
            if key.endswith("_name") and val:
                record[key] = val

        source_entity_type = relation_metadata.get("source_entity_type")
        target_entity_type = relation_metadata.get("target_entity_type")
        if source_entity_type and source_val and f"{source_entity_type}_name" not in record:
            record[f"{source_entity_type}_name"] = source_val
        if target_entity_type and target_val and f"{target_entity_type}_name" not in record:
            record[f"{target_entity_type}_name"] = target_val

        if len(parts) >= 2 and not (source_entity_type and target_entity_type):
            # Legacy fallback when metadata is unavailable.
            entity1_type, entity2_type = parts[0], parts[1]
            if source_val and f"{entity1_type}_name" not in record:
                record[f"{entity1_type}_name"] = source_val
            if target_val and f"{entity2_type}_name" not in record:
                record[f"{entity2_type}_name"] = target_val

        # Copy over any additional attributes (excluding type, source, target)
        for key, val in relation.items():
            if key not in ("type", "source", "target") and key not in record:
                record[key] = val

        return record

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_entity_types(self) -> List[str]:
        """Return all discovered entity types.

        Returns:
            List of entity type names (e.g., ["person", "work", "award"])
        """
        return list(self.entities.keys())

    def get_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type.

        Args:
            entity_type: The type of entities to retrieve

        Returns:
            List of entity instances, or empty list if type not found
        """
        return self.entities.get(entity_type, [])

    def get_all_attributes(self, entity_type: str) -> Set[str]:
        """Return all unique attributes for an entity type.

        Args:
            entity_type: The entity type to get attributes for

        Returns:
            Set of all attribute names found across all entities of this type
        """
        attrs: Set[str] = set()
        for entity in self.entities.get(entity_type, []):
            attrs.update(entity.keys())
        return attrs

    def get_all_attributes_all_types(self) -> Dict[str, Set[str]]:
        """Return all attributes for all entity types.

        Returns:
            Dictionary mapping entity type to set of attribute names
        """
        return {
            entity_type: self.get_all_attributes(entity_type)
            for entity_type in self.get_entity_types()
        }

    def count_entities(self, entity_type: Optional[str] = None) -> int:
        """Count entities, optionally filtered by type.

        Args:
            entity_type: If provided, count only this type. Otherwise count all.

        Returns:
            Number of entities
        """
        if entity_type:
            return len(self.entities.get(entity_type, []))
        return sum(len(entities) for entities in self.entities.values())

    def count_relationships(self, junction_table: Optional[str] = None) -> int:
        """Count relationships, optionally filtered by junction table.

        Args:
            junction_table: If provided, count only this table. Otherwise count all.

        Returns:
            Number of relationship records
        """
        if junction_table:
            return len(self.relationships.get(junction_table, []))
        return sum(len(rels) for rels in self.relationships.values())

    def get_junction_tables(self) -> List[str]:
        """Return all junction table names.

        Returns:
            List of junction table names (e.g., ["person_work", "award_person"])
        """
        return list(self.relationships.keys())

    def get_relationships(self, junction_table: str) -> List[Dict[str, Any]]:
        """Get all relationship records for a junction table.

        Args:
            junction_table: The junction table name

        Returns:
            List of relationship records, or empty list if not found
        """
        return self.relationships.get(junction_table, [])

    def is_empty(self) -> bool:
        """Check if the registry contains no entities.

        Returns:
            True if no entities exist
        """
        return self.count_entities() == 0

    # =========================================================================
    # Modification Methods
    # =========================================================================

    def add_entity(self, entity_type: str, entity: Dict[str, Any]) -> None:
        """Add a single entity to the registry.

        Args:
            entity_type: The type of the entity
            entity: The entity data as a dictionary
        """
        if entity_type not in self.entities:
            self.entities[entity_type] = []
            logger.info(f"EntityRegistry: Created new entity type '{entity_type}'")
        self.entities[entity_type].append(entity)
        logger.debug(
            f"EntityRegistry: Added entity to '{entity_type}' (total: {len(self.entities[entity_type])})"
        )

    def add_entities(self, entity_type: str, entities: List[Dict[str, Any]]) -> None:
        """Add multiple entities of the same type.

        Args:
            entity_type: The type of the entities
            entities: List of entity dictionaries
        """
        if entity_type not in self.entities:
            self.entities[entity_type] = []
            logger.info(f"EntityRegistry: Created new entity type '{entity_type}'")
        self.entities[entity_type].extend(entities)
        logger.info(
            f"EntityRegistry: Added {len(entities)} entities to '{entity_type}' (total: {len(self.entities[entity_type])})"
        )

    def add_relationship(
        self,
        junction_table: str,
        data: Dict[str, Any],
    ) -> None:
        """Add a relationship record to a junction table.

        Args:
            junction_table: Name of the junction table (e.g., "person_work")
            data: Relationship data containing FK columns and optional metadata
                  e.g., {"person_id": 1, "work_id": 2, "person_name": "John", "work_title": "Book"}
        """
        if junction_table not in self.relationships:
            self.relationships[junction_table] = []
            logger.info(
                f"EntityRegistry: Created new junction table '{junction_table}'"
            )
        self.relationships[junction_table].append(data)
        logger.debug(
            f"EntityRegistry: Added relationship to '{junction_table}' "
            f"(total: {len(self.relationships[junction_table])})"
        )

    def add_relationship_by_names(
        self,
        junction_table: str,
        entity1_type: str,
        entity1_name: str,
        entity2_type: str,
        entity2_name: str,
    ) -> None:
        """Add a relationship using entity names (for later ID resolution).

        This method stores the relationship with both entity names for later
        resolution to actual IDs during upsert generation.

        Args:
            junction_table: Name of the junction table (e.g., "person_work")
            entity1_type: First entity type (e.g., "person")
            entity1_name: Name of first entity (e.g., "John Doe")
            entity2_type: Second entity type (e.g., "work")
            entity2_name: Name of second entity (e.g., "My Book")
        """
        # Sort entity types alphabetically to match junction table naming
        sorted_types = sorted([(entity1_type, entity1_name), (entity2_type, entity2_name)])
        type1, name1 = sorted_types[0]
        type2, name2 = sorted_types[1]

        data = {
            f"{type1}_name": name1,
            f"{type2}_name": name2,
        }

        self.add_relationship(junction_table, data)

    def add_relationships(
        self,
        junction_table: str,
        relationships: List[Dict[str, Any]],
    ) -> None:
        """Add multiple relationship records to a junction table.

        Args:
            junction_table: Name of the junction table
            relationships: List of relationship data dictionaries
        """
        if junction_table not in self.relationships:
            self.relationships[junction_table] = []
            logger.info(
                f"EntityRegistry: Created new junction table '{junction_table}'"
            )
        self.relationships[junction_table].extend(relationships)
        logger.info(
            f"EntityRegistry: Added {len(relationships)} relationships to '{junction_table}' "
            f"(total: {len(self.relationships[junction_table])})"
        )

    def merge(self, other: "EntityRegistry") -> "EntityRegistry":
        """Merge another registry into this one, combining entities and relationships.

        Creates a new EntityRegistry containing entities and relationships from both registries.

        Args:
            other: Another EntityRegistry to merge

        Returns:
            New EntityRegistry with merged entities and relationships
        """
        merged_entities: Dict[str, List[Dict[str, Any]]] = {}
        merged_relationships: Dict[str, List[Dict[str, Any]]] = {}

        # Add all entities from self
        for entity_type, entities in self.entities.items():
            merged_entities[entity_type] = list(entities)

        # Add all entities from other
        for entity_type, entities in other.entities.items():
            if entity_type in merged_entities:
                merged_entities[entity_type].extend(entities)
            else:
                merged_entities[entity_type] = list(entities)

        # Add all relationships from self
        for junction_table, rels in self.relationships.items():
            merged_relationships[junction_table] = list(rels)

        # Add all relationships from other
        for junction_table, rels in other.relationships.items():
            if junction_table in merged_relationships:
                merged_relationships[junction_table].extend(rels)
            else:
                merged_relationships[junction_table] = list(rels)

        result = EntityRegistry(entities=merged_entities, relationships=merged_relationships)
        logger.info(
            f"\nEntityRegistry: Merged registries\n"
            f"  Entities: {self.count_entities()} + {other.count_entities()} -> {result.count_entities()}\n"
            f"  Relationships: {self.count_relationships()} + {other.count_relationships()} -> {result.count_relationships()}"
        )
        return result

    def deduplicate(
        self, key_attributes: Optional[Dict[str, str]] = None
    ) -> "EntityRegistry":
        """Deduplicate entities using specified key attribute per type.

        If no key_attributes provided, uses "name" for all types, falling back
        to "title" if "name" doesn't exist.

        Args:
            key_attributes: Mapping of entity_type -> key attribute name
                           e.g., {"person": "name", "work": "title"}

        Returns:
            New EntityRegistry with deduplicated entities
        """
        if key_attributes is None:
            key_attributes = {}

        deduped_entities: Dict[str, List[Dict[str, Any]]] = {}
        dedup_details: List[str] = []  # Collect per-type dedup info

        for entity_type, entities in self.entities.items():
            original_count = len(entities)

            # Filter out malformed entities (non-dict items)
            valid_entities = [e for e in entities if isinstance(e, dict)]
            if len(valid_entities) < original_count:
                logger.warning(
                    f"Filtered {original_count - len(valid_entities)} malformed "
                    f"(non-dict) entities from '{entity_type}'"
                )
            entities = valid_entities

            # Determine the key attribute for this type
            key_attr = key_attributes.get(entity_type)
            if not key_attr:
                key_attr = self._detect_entity_natural_key(entity_type, entities)

            if not key_attr:
                # Can't deduplicate without a key
                deduped_entities[entity_type] = list(entities)
                dedup_details.append(
                    f"  {entity_type}: {original_count} -> {original_count} (no key)"
                )
                continue

            # Deduplicate by merging entities with same key
            seen: Dict[str, Dict[str, Any]] = {}
            null_key_entities: List[Dict[str, Any]] = []

            # Define fallback key attributes (standard first, then legacy patterns)
            fallback_keys = ["name", "title", "full_name", f"{entity_type}_name"]

            for entity in entities:
                key_value = entity.get(key_attr)

                # Treat None or empty string as "no key", try fallbacks
                if key_value is None or (
                    isinstance(key_value, str) and not key_value.strip()
                ):
                    # Try fallback key attributes
                    for fallback in fallback_keys:
                        fallback_val = entity.get(fallback)
                        if fallback_val and (
                            not isinstance(fallback_val, str)
                            or fallback_val.strip()
                        ):
                            key_value = fallback_val
                            break

                if key_value is None or (
                    isinstance(key_value, str) and not key_value.strip()
                ):
                    # Still no key value, keep as is (can't deduplicate)
                    null_key_entities.append(entity)
                    continue

                # Normalize key for comparison
                key_normalized = str(key_value).strip().lower()

                if key_normalized in seen:
                    # Smart merge: combine attributes intelligently
                    existing = seen[key_normalized]
                    self._smart_merge_entity(existing, entity)
                else:
                    seen[key_normalized] = dict(entity)

            # Combine deduplicated entities with those that had null keys
            deduped_entities[entity_type] = list(seen.values()) + null_key_entities
            final_count = len(deduped_entities[entity_type])
            removed = original_count - final_count
            dedup_details.append(
                f"  {entity_type}: {original_count} -> {final_count} (-{removed}, key='{key_attr}')"
            )

        # Deduplicate relationships as well
        deduped_relationships: Dict[str, List[Dict[str, Any]]] = {}
        for junction_table, rels in self.relationships.items():
            seen_keys: Set[str] = set()
            unique_rels: List[Dict[str, Any]] = []
            for rel in rels:
                # Create a key from sorted items
                key = json.dumps(rel, sort_keys=True, default=str)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_rels.append(rel)
            deduped_relationships[junction_table] = unique_rels

        result = EntityRegistry(entities=deduped_entities, relationships=deduped_relationships)
        details_str = "\n".join(dedup_details) if dedup_details else "  (no entities)"
        logger.info(
            f"\nEntityRegistry: Deduplication\n"
            f"  Entities - Before: {self.count_entities()}, After: {result.count_entities()}\n"
            f"  Relationships - Before: {self.count_relationships()}, After: {result.count_relationships()}\n"
            f"{details_str}"
        )

        # Link relationship entity references to canonical entity names
        result = result._link_relationships_to_entities()

        return result

    def _smart_merge_entity(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any],
    ) -> None:
        """Smart merge of entity attributes from multiple QA pairs.

        Strategy:
        1. For None/missing values: use the new value
        2. For string values: keep the longer/more detailed one
        3. For list-like attributes: combine unique values
        4. For other types: keep existing unless it's None

        Args:
            existing: Existing entity dict to update in place
            new: New entity dict to merge from
        """
        for attr, new_value in new.items():
            if new_value is None:
                continue

            existing_value = existing.get(attr)

            # Case 1: Existing is None - use new value
            if existing_value is None:
                existing[attr] = new_value
                continue

            # Case 2: List-like attributes - combine values
            if attr in LIST_LIKE_ATTRS:
                combined = self._combine_list_like_values(existing_value, new_value)
                if combined != existing_value:
                    existing[attr] = combined
                continue

            # Case 3: Both are strings - keep longer/more detailed
            if isinstance(new_value, str) and isinstance(existing_value, str):
                new_stripped = new_value.strip()
                existing_stripped = existing_value.strip()

                # Keep longer value (more detail)
                if len(new_stripped) > len(existing_stripped):
                    existing[attr] = new_value
                # If same length, prefer one with more information (contains existing)
                elif (
                    len(new_stripped) == len(existing_stripped)
                    and existing_stripped in new_stripped
                ):
                    existing[attr] = new_value
                continue

            # Case 4: Other types - keep existing (already have a value)
            # No change needed

    def _combine_list_like_values(
        self,
        existing: Any,
        new: Any,
    ) -> str:
        """Combine list-like attribute values, removing duplicates.

        Values are typically semicolon-separated strings. Preserves original
        casing while deduplicating based on lowercase comparison.

        Args:
            existing: Existing value
            new: New value to merge

        Returns:
            Combined value string with unique items
        """
        # Convert to strings
        existing_str = str(existing) if existing else ""
        new_str = str(new) if new else ""

        # Split by common separators and track original casing
        separators = [";", ",", "|"]
        # Map lowercase -> original case (first occurrence wins)
        items_map: Dict[str, str] = {}

        def extract_items(text: str) -> None:
            """Extract items from text, preserving first occurrence casing."""
            for sep in separators:
                if sep in text:
                    for item in text.split(sep):
                        stripped = item.strip()
                        if stripped:
                            key = stripped.lower()
                            if key not in items_map:
                                items_map[key] = stripped
                    return
            # No separator found - treat as single item
            stripped = text.strip()
            if stripped:
                key = stripped.lower()
                if key not in items_map:
                    items_map[key] = stripped

        # Extract from existing first (preserves existing casing)
        extract_items(existing_str)
        # Then from new (only adds items not already present)
        extract_items(new_str)

        if not items_map:
            return existing_str

        # Return combined with semicolon separator, sorted for consistency
        return "; ".join(sorted(items_map.values(), key=str.lower))

    # =========================================================================
    # Protected Methods
    # =========================================================================

    @staticmethod
    def _detect_entity_natural_key(
        entity_type: str, entities: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Detect the natural key attribute for an entity type.

        Uses heuristics to find the attribute that serves as the semantic
        identifier for entities. This matches the logic in
        TableSchema._detect_natural_key() for consistency.

        Priority order:
        1. Known natural key names: "name", "title", "label", "award_name"
        2. First non-ID string attribute (attribute not ending with "_id")

        Args:
            entity_type: The entity type name
            entities: List of entity dictionaries

        Returns:
            Attribute name to use as key, or None if no suitable key found
        """
        if not entities:
            return None

        # Get attribute names from first entity
        sample_entity: Dict[str, Any] = entities[0]
        attr_names: List[str] = list(sample_entity.keys())

        # Priority 1: Check known natural key names (same as TableSchema logic)
        known_natural_keys: List[str] = ["name", "title", "label", "award_name"]
        for key_name in known_natural_keys:
            if key_name in attr_names:
                return key_name

        # Priority 2: First non-ID attribute (skip PK and FK columns)
        pk_name: str = f"{entity_type}_id"
        for attr in attr_names:
            # Skip primary key and foreign key columns
            if attr == pk_name or attr.endswith("_id"):
                continue
            # Use the first remaining attribute
            return attr

        return None

    # =========================================================================
    # Entity Linking Methods
    # =========================================================================

    def _link_relationships_to_entities(self) -> "EntityRegistry":
        """Link relationship entity references to canonical entity names.

        This method fixes inconsistencies where relationships reference entities
        using slightly different names than the actual entity names. For example:
        - Entity: "Edgar Allan Poe Award for Best Fact Crime"
        - Relationship reference: "Edgar Allan Poe Award"

        Uses fuzzy matching (substring containment, token overlap) to find the
        best matching canonical entity name.

        Returns:
            New EntityRegistry with linked relationship references
        """
        if not self.relationships:
            return self

        # Build entity name index: {entity_type: {normalized_name: canonical_name}}
        entity_index: Dict[str, Dict[str, str]] = {}
        for entity_type, entity_list in self.entities.items():
            entity_index[entity_type] = {}
            for entity in entity_list:
                # Get canonical name (prefer 'name', fallback to 'title')
                canonical = entity.get("name") or entity.get("title")
                if canonical and isinstance(canonical, str):
                    normalized = self._normalize_entity_name(canonical)
                    entity_index[entity_type][normalized] = canonical

        # Process each relationship type
        linked_relationships: Dict[str, List[Dict[str, Any]]] = {}
        total_refs = 0
        linked_refs = 0
        unlinked_refs: List[str] = []

        for junction_table, rel_list in self.relationships.items():
            linked_rels: List[Dict[str, Any]] = []

            for rel in rel_list:
                linked_rel = dict(rel)  # Copy to avoid mutation

                # Process each key that looks like an entity reference
                for key, value in rel.items():
                    if not isinstance(value, str) or not value.strip():
                        continue

                    # Identify entity type from key pattern
                    entity_type = self._extract_entity_type_from_key(key)
                    if not entity_type or entity_type not in entity_index:
                        continue

                    total_refs += 1

                    # Try to find matching canonical name
                    canonical = self._find_best_entity_match(
                        value, entity_index[entity_type]
                    )

                    if canonical and canonical != value:
                        linked_rel[key] = canonical
                        linked_refs += 1
                        logger.debug(
                            f"Linked '{value}' -> '{canonical}' ({entity_type})"
                        )
                    elif not canonical:
                        # Could not find match - keep original
                        unlinked_refs.append(f"{entity_type}:{value}")

                linked_rels.append(linked_rel)

            linked_relationships[junction_table] = linked_rels

        # Log summary
        if total_refs > 0:
            logger.info(
                f"\nEntityRegistry: Relationship Entity Linking\n"
                f"  Total entity references: {total_refs}\n"
                f"  Successfully linked: {linked_refs}\n"
                f"  Already matched: {total_refs - linked_refs - len(unlinked_refs)}\n"
                f"  Unlinked (no match): {len(unlinked_refs)}"
            )
            if unlinked_refs and len(unlinked_refs) <= 10:
                logger.debug(f"  Unlinked references: {unlinked_refs}")
            elif unlinked_refs:
                logger.debug(
                    f"  Sample unlinked: {unlinked_refs[:5]}... "
                    f"(+{len(unlinked_refs) - 5} more)"
                )

        return EntityRegistry(entities=self.entities, relationships=linked_relationships)

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """Normalize an entity name for comparison.

        Applies lowercase, strips whitespace, and removes common noise.

        Args:
            name: Entity name to normalize

        Returns:
            Normalized name string
        """
        if not name:
            return ""

        # Lowercase and strip
        normalized = name.strip().lower()

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    @staticmethod
    def _extract_entity_type_from_key(key: str) -> Optional[str]:
        """Extract entity type from a relationship key.

        Handles patterns like:
        - "person_name" -> "person"
        - "award_name" -> "award"
        - "work_name" -> "work"
        - "source" -> None (ambiguous)
        - "target" -> None (ambiguous)

        Args:
            key: Relationship dictionary key

        Returns:
            Entity type string or None if not identifiable
        """
        if key.endswith("_name"):
            return key[:-5]  # Remove "_name" suffix
        if key.endswith("_title"):
            return key[:-6]  # Remove "_title" suffix

        # Known entity reference keys without suffix
        # (source/target are ambiguous, so we skip them)
        return None

    def _find_best_entity_match(
        self, query: str, candidates: Dict[str, str]
    ) -> Optional[str]:
        """Find the best matching entity name using fuzzy matching.

        Matching strategies (in order of preference):
        1. Exact match (after normalization)
        2. Substring containment (either direction)
        3. Token overlap (for multi-word names, requires >50% overlap)

        Args:
            query: The entity reference to match
            candidates: Dict mapping normalized names to canonical names

        Returns:
            Canonical entity name if match found, None otherwise
        """
        if not query or not candidates:
            return None

        query_norm = self._normalize_entity_name(query)

        # Strategy 1: Exact match
        if query_norm in candidates:
            return candidates[query_norm]

        # Strategy 2: Substring containment (either direction)
        # Prefer longer matches (more specific)
        substring_matches: List[tuple] = []
        for norm, canonical in candidates.items():
            if query_norm in norm:
                # Query is substring of candidate
                substring_matches.append((len(norm), canonical, "query_in_cand"))
            elif norm in query_norm:
                # Candidate is substring of query
                substring_matches.append((len(norm), canonical, "cand_in_query"))

        if substring_matches:
            # Sort by length descending (prefer longer/more specific matches)
            substring_matches.sort(key=lambda x: x[0], reverse=True)
            return substring_matches[0][1]

        # Strategy 3: Token overlap (for multi-word names)
        query_tokens = set(query_norm.split())
        if len(query_tokens) >= 2:
            best_score = 0.0
            best_match: Optional[str] = None

            for norm, canonical in candidates.items():
                cand_tokens = set(norm.split())
                if len(cand_tokens) < 2:
                    continue

                # Calculate Jaccard-like overlap
                intersection = len(query_tokens & cand_tokens)
                union = len(query_tokens | cand_tokens)
                overlap = intersection / union if union > 0 else 0

                # Require at least 50% overlap and at least 2 matching tokens
                if overlap > best_score and overlap >= 0.5 and intersection >= 2:
                    best_score = overlap
                    best_match = canonical

            if best_match:
                return best_match

        return None

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_json(self, indent: int = 2) -> str:
        """Serialize the registry to a JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation including entities and relationships
        """
        data: Dict[str, Any] = {
            "entities": self.entities,
        }
        if self.relationships:
            data["relationships"] = self.relationships
        return json.dumps(data, indent=indent, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary.

        Returns:
            Dictionary representation with entities and relationships
        """
        data: Dict[str, Any] = {
            "entities": dict(self.entities),
        }
        if self.relationships:
            data["relationships"] = dict(self.relationships)
        return data

    # =========================================================================
    # String Representation
    # =========================================================================

    def __str__(self) -> str:
        """Return a human-readable summary of the registry."""
        lines = ["EntityRegistry:"]
        lines.append("  Entities:")
        for entity_type, entities in self.entities.items():
            lines.append(f"    {entity_type}: {len(entities)} entities")
            attrs = self.get_all_attributes(entity_type)
            if attrs:
                lines.append(f"      attributes: {sorted(attrs)}")
        if self.relationships:
            lines.append("  Relationships:")
            for junction_table, rels in self.relationships.items():
                lines.append(f"    {junction_table}: {len(rels)} records")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return (
            f"EntityRegistry(entity_types={self.get_entity_types()}, "
            f"entities={self.count_entities()}, relationships={self.count_relationships()})"
        )
