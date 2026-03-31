"""Relationship detector for identifying entity relationships from attributes.

This module provides functionality to detect many-to-many and many-to-one
relationships between entity types based on attribute patterns, and to
enhance schemas with junction tables and foreign key constraints.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.aligned_db.attribute_roles import AttributeRole, parse_attribute_role
from src.aligned_db.schema_registry import (
    ColumnInfo,
    ForeignKeyConstraint,
    SchemaRegistry,
    TableSchema,
)

logger = logging.getLogger("RelationshipDetector")

# Suffixes that indicate plural/collection attributes (many-to-many)
PLURAL_SUFFIXES: List[str] = [
    "_list",
    "_received",
    "_created",
    "_authored",
    "_written",
    "_won",
    "_explored",
    "_sources",
    "_works",
]

# Common attribute name patterns that map to entity types
ENTITY_REFERENCE_PATTERNS: Dict[str, str] = {
    "creator": "person",
    "author": "person",
    "writer": "person",
    "birth_place": "location",
    "birth_location": "location",
    "residence": "location",
    "upbringing_location": "location",
    "nationality": "nationality",
    "occupation": "occupation",
    "profession": "profession",
    "genre": "genre",
    "primary_genre": "genre",
    "field": "field",
    "language": "language",
    "series": "series",
    "theme": "theme",
    "gender_identity": "gender_identity",
    "employment_status": "employment_status",
    "age_group": "age",
}

# Attributes that represent many-to-many relationships
MANY_TO_MANY_ATTRIBUTES: Dict[str, Tuple[str, str]] = {
    # attribute_name: (source_entity, target_entity)
    "awards_received": ("person", "award"),
    "works_created": ("person", "work"),
    "authored_works": ("person", "work"),
    "written_works": ("person", "work"),
    "notable_works_list": ("person", "work"),
    "themes_explored_in_works": ("person", "theme"),
    "inspiration_sources": ("person", "person"),  # self-referential
}

# Common patterns mapping attribute substrings to target entity types
# Used by RelationshipDetector for FK relationship detection
RELATIONSHIP_PATTERNS: Dict[str, str] = {
    "awards": "award",
    "works": "work",
    "books": "work",
    "themes": "theme",
    "genres": "genre",
    "occupations": "occupation",
    "languages": "language",
    "characters": "character",
}


@dataclass
class DetectedRelationship:
    """Represents a detected relationship between entity types.

    Attributes:
        source_entity: The entity type where the attribute was found
        target_entity: The entity type being referenced
        attribute_name: The original attribute name
        relationship_type: Either "many_to_many" or "many_to_one"
        junction_table_name: Name of junction table (for many_to_many)
        fk_column_name: Name of FK column (for many_to_one)
    """

    source_entity: str
    target_entity: str
    attribute_name: str
    relationship_type: str  # "many_to_many" or "many_to_one"
    junction_table_name: Optional[str] = None
    fk_column_name: Optional[str] = None


class RelationshipDetector:
    """Detects relationships between entity types from attributes.

    This class analyzes entity types and their attributes to identify:
    1. Many-to-many relationships that need junction tables
    2. Many-to-one relationships that need foreign key columns

    Attributes:
        entity_types: List of entity type dictionaries with name and description
        entity_names: Set of lowercase entity type names for quick lookup
    """

    def __init__(self, entity_types: List[Dict[str, str]]) -> None:
        """Initialize the RelationshipDetector.

        Args:
            entity_types: List of entity type dictionaries with 'name' key
        """
        self.entity_types = entity_types
        self.entity_names: Set[str] = {
            et.get("name", "").lower() for et in entity_types if et.get("name")
        }
        logger.info(
            f"RelationshipDetector initialized with {len(self.entity_names)} entity types: "
            f"{sorted(self.entity_names)}"
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def detect_relationships(
        self,
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> List[DetectedRelationship]:
        """Detect all relationships from normalized attributes.

        Args:
            normalized_attributes: Dict mapping entity_type to list of attribute dicts

        Returns:
            List of DetectedRelationship objects
        """
        relationships: List[DetectedRelationship] = []

        for entity_type, attrs in normalized_attributes.items():
            entity_type_lower = entity_type.lower()

            for attr in attrs:
                attr_name = attr.get("canonical_name", attr.get("name", "")).lower()
                if not attr_name:
                    continue

                # Check for known many-to-many relationships
                if attr_name in MANY_TO_MANY_ATTRIBUTES:
                    source, target = MANY_TO_MANY_ATTRIBUTES[attr_name]
                    # Only add if source matches current entity or is generic
                    if source == entity_type_lower or source in self.entity_names:
                        rel = self._create_many_to_many_relationship(
                            entity_type_lower, target, attr_name
                        )
                        if rel:
                            relationships.append(rel)
                        continue

                # Check if attribute looks like a many-to-many relationship
                if self._is_plural_attribute(attr_name):
                    target_entity = self._detect_referenced_entity(
                        attr_name, attr.get("description", "")
                    )
                    if target_entity and target_entity != entity_type_lower:
                        rel = self._create_many_to_many_relationship(
                            entity_type_lower, target_entity, attr_name
                        )
                        if rel:
                            relationships.append(rel)
                        continue

                # Check for many-to-one relationships
                target_entity = self._detect_fk_reference(
                    entity_type_lower,
                    attr_name,
                    attr,
                )
                if target_entity and target_entity in self.entity_names:
                    rel = self._create_many_to_one_relationship(
                        entity_type_lower, target_entity, attr_name
                    )
                    if rel:
                        relationships.append(rel)

        # Deduplicate relationships
        unique_relationships = self._deduplicate_relationships(relationships)

        logger.info(
            f"Detected {len(unique_relationships)} relationships "
            f"({sum(1 for r in unique_relationships if r.relationship_type == 'many_to_many')} M:N, "
            f"{sum(1 for r in unique_relationships if r.relationship_type == 'many_to_one')} M:1)"
        )

        return unique_relationships

    def enhance_schema(
        self,
        schema_registry: SchemaRegistry,
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> SchemaRegistry:
        """Enhance a schema with junction tables and foreign key columns.

        Args:
            schema_registry: The base schema to enhance
            normalized_attributes: Dict mapping entity_type to attribute dicts

        Returns:
            Enhanced SchemaRegistry with relationships
        """
        logger.info("Enhancing schema with relationships...")

        # Detect all relationships
        relationships = self.detect_relationships(normalized_attributes)

        # Process each relationship
        junction_tables_created: int = 0
        fk_columns_added: int = 0

        for rel in relationships:
            if rel.relationship_type == "many_to_many":
                # Create junction table
                if self._create_junction_table(schema_registry, rel):
                    junction_tables_created += 1
            elif rel.relationship_type == "many_to_one":
                # Add FK column
                if self._add_fk_column(schema_registry, rel):
                    fk_columns_added += 1

        logger.info(
            f"Schema enhancement complete: "
            f"{junction_tables_created} junction tables created, "
            f"{fk_columns_added} FK columns added"
        )

        return schema_registry

    # =========================================================================
    # Protected Methods - Relationship Detection
    # =========================================================================

    def _is_junction_table_name(self, name: str) -> bool:
        """Check if a name looks like a junction table (entity1_entity2 pattern).

        Junction tables are named by combining two entity names with underscore.
        If both parts of a name are known entity types, it's likely a junction
        table rather than a real entity.

        Args:
            name: Name to check

        Returns:
            True if name appears to be a junction table name
        """
        parts = name.split("_")
        if len(parts) != 2:
            return False
        # Both parts must be known entity names
        return parts[0] in self.entity_names and parts[1] in self.entity_names

    def _is_plural_attribute(self, attr_name: str) -> bool:
        """Check if an attribute name indicates a plural/collection.

        Args:
            attr_name: Attribute name to check

        Returns:
            True if the attribute appears to be plural
        """
        # Check known plural suffixes
        for suffix in PLURAL_SUFFIXES:
            if attr_name.endswith(suffix):
                return True

        # Check if ends with 's' and base form might be an entity
        if attr_name.endswith("s") and len(attr_name) > 2:
            base = attr_name[:-1]  # Remove 's'
            if base in self.entity_names:
                return True

        return False

    def _detect_referenced_entity(
        self,
        attr_name: str,
        description: str,
    ) -> Optional[str]:
        """Detect which entity type an attribute references.

        Args:
            attr_name: Attribute name to analyze
            description: Attribute description for additional context

        Returns:
            Referenced entity type name, or None if not detected
        """
        # Clean attribute name
        cleaned = attr_name.lower()
        for suffix in PLURAL_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break

        # Remove trailing 's' if present
        if cleaned.endswith("s") and len(cleaned) > 2:
            cleaned = cleaned[:-1]

        # Check if cleaned name matches an entity type
        if cleaned in self.entity_names:
            return cleaned

        # Check description for entity mentions
        desc_lower = description.lower()
        for entity in self.entity_names:
            if entity in desc_lower:
                return entity

        # Common mappings
        mappings: Dict[str, str] = {
            "award": "award",
            "work": "work",
            "book": "work",
            "theme": "theme",
            "genre": "genre",
            "occupation": "occupation",
            "parent": "person",
            "collaborat": "person",
        }

        for key, entity in mappings.items():
            if key in cleaned and entity in self.entity_names:
                return entity

        return None

    def _detect_fk_reference(
        self,
        source_entity: str,
        attr_name: str,
        attr_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Detect if an attribute should be a foreign key reference.

        Args:
            source_entity: Entity type that owns this attribute
            attr_name: Attribute name to check
            attr_meta: Optional metadata for explicit role hints

        Returns:
            Referenced entity type, or None if not a FK reference
        """
        role = parse_attribute_role(
            (attr_meta or {}).get("predicted_role") or (attr_meta or {}).get("role_hint")
        )
        hinted_target = (attr_meta or {}).get("target_table")
        hinted_confidence = float((attr_meta or {}).get("role_confidence", 0.0) or 0.0)

        if (
            role in (AttributeRole.ENTITY_REFERENCE, AttributeRole.SELF_REFERENCE)
            and hinted_target in self.entity_names
            and hinted_confidence >= 0.6
        ):
            return hinted_target

        if role in (
            AttributeRole.SCALAR,
            AttributeRole.CONTROLLED_VALUE,
            AttributeRole.RELATION_ATTRIBUTE,
        ) and hinted_confidence >= 0.6:
            return None

        # Check known patterns first
        explicit_target = ENTITY_REFERENCE_PATTERNS.get(attr_name)
        if explicit_target and explicit_target in self.entity_names:
            if explicit_target == source_entity:
                return None
            return explicit_target

        # Check if attribute name matches an entity type directly
        if attr_name in self.entity_names and attr_name != source_entity:
            return attr_name

        # Check if attribute name with common suffixes removed matches entity
        for suffix in ["_name", "_type", "_id"]:
            if attr_name.endswith(suffix):
                base = attr_name[: -len(suffix)]
                if base in self.entity_names:
                    if base == source_entity:
                        continue
                    if suffix == "_type":
                        continue
                    return base

        return None

    # =========================================================================
    # Protected Methods - Relationship Creation
    # =========================================================================

    def _create_many_to_many_relationship(
        self,
        source_entity: str,
        target_entity: str,
        attr_name: str,
    ) -> Optional[DetectedRelationship]:
        """Create a many-to-many relationship definition.

        Args:
            source_entity: Source entity type
            target_entity: Target entity type
            attr_name: Original attribute name

        Returns:
            DetectedRelationship or None if invalid
        """
        if target_entity not in self.entity_names:
            logger.debug(
                f"Skipping M:N relationship {source_entity}.{attr_name} -> {target_entity}: "
                f"target entity not found"
            )
            return None

        # Create junction table name (alphabetically sorted)
        sorted_entities = sorted([source_entity, target_entity])
        junction_name = f"{sorted_entities[0]}_{sorted_entities[1]}"

        return DetectedRelationship(
            source_entity=source_entity,
            target_entity=target_entity,
            attribute_name=attr_name,
            relationship_type="many_to_many",
            junction_table_name=junction_name,
        )

    def _create_many_to_one_relationship(
        self,
        source_entity: str,
        target_entity: str,
        attr_name: str,
    ) -> Optional[DetectedRelationship]:
        """Create a many-to-one relationship definition.

        Args:
            source_entity: Source entity type
            target_entity: Target entity type
            attr_name: Original attribute name

        Returns:
            DetectedRelationship or None if invalid
        """
        if target_entity not in self.entity_names:
            return None

        # Create FK column name
        fk_col_name = f"{attr_name}_id" if not attr_name.endswith("_id") else attr_name

        return DetectedRelationship(
            source_entity=source_entity,
            target_entity=target_entity,
            attribute_name=attr_name,
            relationship_type="many_to_one",
            fk_column_name=fk_col_name,
        )

    def _deduplicate_relationships(
        self,
        relationships: List[DetectedRelationship],
    ) -> List[DetectedRelationship]:
        """Remove duplicate relationships.

        Args:
            relationships: List of relationships to deduplicate

        Returns:
            Deduplicated list
        """
        seen: Set[str] = set()
        unique: List[DetectedRelationship] = []

        for rel in relationships:
            # Create unique key based on relationship type
            if rel.relationship_type == "many_to_many":
                # Junction table name is already normalized (alphabetically sorted)
                key = f"m2m:{rel.junction_table_name}"
            else:
                key = f"m2o:{rel.source_entity}:{rel.fk_column_name}"

            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    # =========================================================================
    # Protected Methods - Schema Enhancement
    # =========================================================================

    def _create_junction_table(
        self,
        schema_registry: SchemaRegistry,
        rel: DetectedRelationship,
    ) -> bool:
        """Create a junction table for a many-to-many relationship.

        Args:
            schema_registry: Schema to add table to
            rel: Relationship definition

        Returns:
            True if table was created, False if already exists or skipped
        """
        if not rel.junction_table_name:
            return False

        # Check if table already exists
        if schema_registry.has_table(rel.junction_table_name):
            logger.debug(f"Junction table {rel.junction_table_name} already exists")
            return False

        # Skip if source or target entity is itself a junction table
        # (Junction tables have composite PKs, not single <entity>_id columns)
        if self._is_junction_table_name(rel.source_entity):
            logger.warning(
                f"Skipping junction table {rel.junction_table_name}: "
                f"source '{rel.source_entity}' appears to be a junction table"
            )
            return False
        if self._is_junction_table_name(rel.target_entity):
            logger.warning(
                f"Skipping junction table {rel.junction_table_name}: "
                f"target '{rel.target_entity}' appears to be a junction table"
            )
            return False

        # Get the two entity names from junction table name
        sorted_entities = sorted([rel.source_entity, rel.target_entity])
        entity1, entity2 = sorted_entities[0], sorted_entities[1]

        # Handle self-referential relationships (e.g., person_person)
        is_self_referential: bool = entity1 == entity2
        if is_self_referential:
            col1_name = f"source_{entity1}_id"
            col2_name = f"target_{entity1}_id"
        else:
            col1_name = f"{entity1}_id"
            col2_name = f"{entity2}_id"

        # Create columns
        columns: List[ColumnInfo] = [
            ColumnInfo(
                name=col1_name,
                data_type="INTEGER",
                is_foreign_key=True,
            ),
            ColumnInfo(
                name=col2_name,
                data_type="INTEGER",
                is_foreign_key=True,
            ),
        ]

        # Create foreign key constraints
        foreign_keys: List[ForeignKeyConstraint] = [
            ForeignKeyConstraint(
                column_name=col1_name,
                references_table=entity1,
                references_column=f"{entity1}_id",
            ),
            ForeignKeyConstraint(
                column_name=col2_name,
                references_table=entity2,
                references_column=f"{entity2}_id",
            ),
        ]

        # Create table with composite primary key
        table = TableSchema(
            name=rel.junction_table_name,
            columns=columns,
            foreign_keys=foreign_keys,
            primary_key_columns=[col1_name, col2_name],
        )

        schema_registry.add_table(table)
        logger.info(
            f"Created junction table: {rel.junction_table_name} "
            f"({col1_name}, {col2_name})"
        )

        return True

    def _add_fk_column(
        self,
        schema_registry: SchemaRegistry,
        rel: DetectedRelationship,
    ) -> bool:
        """Add a foreign key column to a table.

        Args:
            schema_registry: Schema containing the table
            rel: Relationship definition

        Returns:
            True if column was added, False otherwise
        """
        if not rel.fk_column_name:
            return False

        # Get source table
        table = schema_registry.get_table(rel.source_entity)
        if not table:
            logger.warning(f"Cannot add FK column: table {rel.source_entity} not found")
            return False

        # Check if FK column already exists
        if table.has_column(rel.fk_column_name):
            logger.debug(
                f"FK column {rel.source_entity}.{rel.fk_column_name} already exists"
            )
            return False

        # Add FK column
        fk_column = ColumnInfo(
            name=rel.fk_column_name,
            data_type="INTEGER",
            is_foreign_key=True,
            references=f"{rel.target_entity}({rel.target_entity}_id)",
        )
        table.add_column(fk_column)

        # Add FK constraint
        fk_constraint = ForeignKeyConstraint(
            column_name=rel.fk_column_name,
            references_table=rel.target_entity,
            references_column=f"{rel.target_entity}_id",
        )
        table.add_foreign_key(fk_constraint)

        logger.info(
            f"Added FK column: {rel.source_entity}.{rel.fk_column_name} -> "
            f"{rel.target_entity}({rel.target_entity}_id)"
        )

        return True


def get_junction_table_name(entity1: str, entity2: str) -> str:
    """Get the standardized junction table name for two entity types.

    Args:
        entity1: First entity type
        entity2: Second entity type

    Returns:
        Junction table name (alphabetically sorted)
    """
    sorted_entities = sorted([entity1.lower(), entity2.lower()])
    return f"{sorted_entities[0]}_{sorted_entities[1]}"
