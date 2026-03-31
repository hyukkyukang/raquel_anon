"""Type registry for finalized entity, attribute, and relation types.

This module provides data structures for storing normalized type definitions
discovered during the type discovery phases (Stages 1 and 2), before
schema generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.aligned_db.attribute_roles import AttributeRole, parse_attribute_role
from src.utils.string import sanitize_sql_identifier

logger = logging.getLogger("src.aligned_db.type_registry")

# Priority attributes that should be extracted first and marked in prompts
# These are high-value attributes that commonly appear in verification questions
PRIORITY_ATTRIBUTES: Dict[str, List[str]] = {
    "person": [
        "name",
        "birth_date",
        "birth_place",
        "occupation",
        "nationality",
        "father_name",
        "father_occupation",
        "mother_name",
        "mother_occupation",
        "gender_identity",
        "awards_received",
    ],
    "work": [
        "title",
        "author",
        "creator",
        "publication_date",
        "genre",
        "themes",
        "characters",
    ],
    "award": [
        "name",
        "year_received",
        "recipient",
        "category",
        "field",
    ],
    "character": [
        "name",
        "appears_in",
        "role",
        "description",
    ],
    "location": [
        "name",
        "country",
        "region",
        "type",
    ],
}


@dataclass
class EntityType:
    """Definition of an entity type (maps to a database table).

    Attributes:
        name: The normalized entity type name (e.g., "person", "work")
        description: Human-readable description of this entity type
        aliases: Alternative names that were normalized to this type
        examples: Example entity names from QA pairs
        is_junction: Whether this represents a junction (many-to-many) table
    """

    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    is_junction: bool = False

    def __post_init__(self) -> None:
        self.name = sanitize_sql_identifier(self.name, default="entity")
        self.aliases = [alias for alias in self.aliases if alias]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "aliases": self.aliases,
            "examples": self.examples,
            "is_junction": self.is_junction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityType":
        """Create EntityType from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            examples=data.get("examples", []),
            is_junction=data.get("is_junction", False),
        )


@dataclass
class AttributeType:
    """Definition of an attribute type (maps to a database column).

    Attributes:
        name: The normalized attribute name (e.g., "full_name", "birth_date")
        data_type: SQL data type (e.g., "TEXT", "INTEGER", "DATE")
        description: Human-readable description
        is_required: Whether this attribute is required (NOT NULL)
        is_unique: Whether this attribute should have a UNIQUE constraint
        is_natural_key: Whether this is the natural key for deduplication
        default_value: Default value for this attribute
        examples: Example values from QA pairs
    """

    name: str
    data_type: str = "TEXT"
    description: str = ""
    is_required: bool = False
    is_unique: bool = False
    is_natural_key: bool = False
    default_value: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    predicted_role: Optional[AttributeRole] = None
    target_table: Optional[str] = None
    role_confidence: Optional[float] = None
    role_evidence: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = sanitize_sql_identifier(self.name, default="attribute")
        self.predicted_role = parse_attribute_role(self.predicted_role)
        if self.target_table:
            self.target_table = sanitize_sql_identifier(
                self.target_table, default="entity"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "is_required": self.is_required,
            "is_unique": self.is_unique,
            "is_natural_key": self.is_natural_key,
            "default_value": self.default_value,
            "examples": self.examples,
            "predicted_role": self.predicted_role.value if self.predicted_role else None,
            "target_table": self.target_table,
            "role_confidence": self.role_confidence,
            "role_evidence": self.role_evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeType":
        """Create AttributeType from dictionary."""
        return cls(
            name=data.get("name", ""),
            data_type=data.get("data_type", "TEXT"),
            description=data.get("description", ""),
            is_required=data.get("is_required", False),
            is_unique=data.get("is_unique", False),
            is_natural_key=data.get("is_natural_key", False),
            default_value=data.get("default_value"),
            examples=data.get("examples", []),
            predicted_role=data.get("predicted_role"),
            target_table=data.get("target_table"),
            role_confidence=data.get("role_confidence"),
            role_evidence=data.get("role_evidence", []),
        )


@dataclass
class RelationType:
    """Definition of a many-to-many relationship (maps to junction table).

    Attributes:
        name: Junction table name (e.g., "person_work")
        source_entity: Name of the source entity type
        target_entity: Name of the target entity type
        description: Human-readable description
        attributes: Additional attributes on the junction table
        examples: Example relationship instances from QA pairs
    """

    name: str
    source_entity: str
    target_entity: str
    description: str = ""
    attributes: List[AttributeType] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = sanitize_sql_identifier(self.name, default="relation")
        self.source_entity = sanitize_sql_identifier(
            self.source_entity, default="source"
        )
        self.target_entity = sanitize_sql_identifier(
            self.target_entity, default="target"
        )

    @property
    def source_fk_name(self) -> str:
        """Get the foreign key column name for source entity."""
        return f"{self.source_entity}_id"

    @property
    def target_fk_name(self) -> str:
        """Get the foreign key column name for target entity."""
        return f"{self.target_entity}_id"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "description": self.description,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationType":
        """Create RelationType from dictionary."""
        return cls(
            name=data.get("name", ""),
            source_entity=data.get("source_entity", ""),
            target_entity=data.get("target_entity", ""),
            description=data.get("description", ""),
            attributes=[
                AttributeType.from_dict(attr) for attr in data.get("attributes", [])
            ],
            examples=data.get("examples", []),
        )


@dataclass
class TypeRegistry:
    """Registry of all finalized types for schema generation.

    This class holds the normalized entity types, their attributes,
    and relationship definitions discovered during Stages 1 and 2.
    It serves as input to Stage 3 (Schema Generation).

    Attributes:
        entity_types: List of normalized entity types
        attribute_types: Dict mapping entity_type name to list of attributes
        relation_types: List of many-to-many relationship definitions
    """

    entity_types: List[EntityType] = field(default_factory=list)
    attribute_types: Dict[str, List[AttributeType]] = field(default_factory=dict)
    relation_types: List[RelationType] = field(default_factory=list)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls) -> "TypeRegistry":
        """Create an empty TypeRegistry."""
        return cls(entity_types=[], attribute_types={}, relation_types=[])

    # =========================================================================
    # Property Methods
    # =========================================================================

    @property
    def entity_type_names(self) -> Set[str]:
        """Get all entity type names."""
        return {et.name for et in self.entity_types}

    @property
    def relation_type_names(self) -> Set[str]:
        """Get all relation type (junction table) names."""
        return {rt.name for rt in self.relation_types}

    @property
    def all_table_names(self) -> Set[str]:
        """Get all table names (entities + junction tables)."""
        return self.entity_type_names | self.relation_type_names

    # =========================================================================
    # Public Methods - Access
    # =========================================================================

    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Get entity type by name.

        Args:
            name: Entity type name

        Returns:
            EntityType if found, None otherwise
        """
        normalized_name = sanitize_sql_identifier(name, default="")
        for et in self.entity_types:
            if et.name == normalized_name:
                return et
        return None

    def get_attributes_for(self, entity_type: str) -> List[AttributeType]:
        """Get all attributes for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            List of AttributeType, empty if entity type not found
        """
        normalized_entity_type = sanitize_sql_identifier(entity_type, default="")
        return self.attribute_types.get(normalized_entity_type, [])

    def get_attribute_type(
        self,
        entity_type: str,
        attribute_name: str,
    ) -> Optional[AttributeType]:
        """Get a specific attribute definition for an entity type."""
        normalized_attribute_name = sanitize_sql_identifier(attribute_name, default="")
        for attr in self.get_attributes_for(entity_type):
            if attr.name == normalized_attribute_name:
                return attr
        return None

    def get_natural_key_for(self, entity_type: str) -> Optional[str]:
        """Get the natural key attribute name for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Natural key attribute name, or None if not found
        """
        for attr in self.get_attributes_for(entity_type):
            if attr.is_natural_key:
                return attr.name
        return None

    def get_relation_type(self, name: str) -> Optional[RelationType]:
        """Get relation type by name.

        Args:
            name: Relation type (junction table) name

        Returns:
            RelationType if found, None otherwise
        """
        normalized_name = sanitize_sql_identifier(name, default="")
        for rt in self.relation_types:
            if rt.name == normalized_name:
                return rt
        return None

    def get_relations_for(self, entity_type: str) -> List[RelationType]:
        """Get all relations involving an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            List of RelationType where entity_type is source or target
        """
        normalized_entity_type = sanitize_sql_identifier(entity_type, default="")
        relations: List[RelationType] = []
        for rt in self.relation_types:
            if (
                rt.source_entity == normalized_entity_type
                or rt.target_entity == normalized_entity_type
            ):
                relations.append(rt)
        return relations

    # =========================================================================
    # Public Methods - Modification
    # =========================================================================

    def add_entity_type(self, entity_type: EntityType) -> None:
        """Add an entity type to the registry.

        Args:
            entity_type: The EntityType to add
        """
        # Check for duplicates
        existing = self.get_entity_type(entity_type.name)
        if existing:
            # Merge aliases and examples
            existing.aliases = list(set(existing.aliases + entity_type.aliases))
            existing.examples = list(set(existing.examples + entity_type.examples))
        else:
            self.entity_types.append(entity_type)

    def add_attribute_type(self, entity_type: str, attribute: AttributeType) -> None:
        """Add an attribute type for an entity.

        Args:
            entity_type: The entity type name
            attribute: The AttributeType to add
        """
        normalized_entity_type = sanitize_sql_identifier(entity_type, default="entity")
        if normalized_entity_type not in self.attribute_types:
            self.attribute_types[normalized_entity_type] = []

        # Check for duplicates
        for existing in self.attribute_types[normalized_entity_type]:
            if existing.name == attribute.name:
                # Merge examples
                existing.examples = list(set(existing.examples + attribute.examples))
                if existing.predicted_role is None and attribute.predicted_role is not None:
                    existing.predicted_role = attribute.predicted_role
                if not existing.target_table and attribute.target_table:
                    existing.target_table = attribute.target_table
                if existing.role_confidence is None and attribute.role_confidence is not None:
                    existing.role_confidence = attribute.role_confidence
                if attribute.role_evidence:
                    existing.role_evidence = list(
                        dict.fromkeys(existing.role_evidence + attribute.role_evidence)
                    )
                return

        self.attribute_types[normalized_entity_type].append(attribute)

    def add_relation_type(self, relation: RelationType) -> None:
        """Add a relation type to the registry.

        Args:
            relation: The RelationType to add
        """
        # Check for duplicates
        existing = self.get_relation_type(relation.name)
        if existing:
            # Merge examples
            existing.examples = existing.examples + relation.examples
        else:
            self.relation_types.append(relation)

    def set_natural_key(self, entity_type: str, attribute_name: str) -> bool:
        """Set an attribute as the natural key for an entity type.

        Args:
            entity_type: Entity type name
            attribute_name: Attribute name to set as natural key

        Returns:
            True if successful, False if attribute not found
        """
        attributes = self.get_attributes_for(entity_type)
        for attr in attributes:
            if attr.name == attribute_name:
                attr.is_natural_key = True
                attr.is_unique = True
                return True
        return False

    # =========================================================================
    # Public Methods - Schema Generation Support
    # =========================================================================

    def to_schema_input(self) -> Dict[str, Any]:
        """Convert to format suitable for schema generator.

        Returns:
            Dictionary with entity_types, attributes, and relations
        """
        return {
            "entity_types": [et.to_dict() for et in self.entity_types],
            "attributes": {
                entity_type: [attr.to_dict() for attr in attrs]
                for entity_type, attrs in self.attribute_types.items()
            },
            "relations": [rt.to_dict() for rt in self.relation_types],
        }

    def get_schema_hints(self) -> Dict[str, Dict[str, Any]]:
        """Get schema hints for each table.

        Returns hints like natural keys, unique constraints, and
        required fields for the schema generator.

        Returns:
            Dict mapping table_name to hints dict
        """
        hints: Dict[str, Dict[str, Any]] = {}

        for entity_type in self.entity_types:
            table_hints: Dict[str, Any] = {
                "unique_columns": [],
                "required_columns": [],
                "natural_key": None,
            }

            for attr in self.get_attributes_for(entity_type.name):
                if attr.is_unique:
                    table_hints["unique_columns"].append(attr.name)
                if attr.is_required:
                    table_hints["required_columns"].append(attr.name)
                if attr.is_natural_key:
                    table_hints["natural_key"] = attr.name

            hints[entity_type.name] = table_hints

        return hints

    # =========================================================================
    # Public Methods - Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entity_types": [et.to_dict() for et in self.entity_types],
            "attribute_types": {
                entity_type: [attr.to_dict() for attr in attrs]
                for entity_type, attrs in self.attribute_types.items()
            },
            "relation_types": [rt.to_dict() for rt in self.relation_types],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypeRegistry":
        """Create TypeRegistry from dictionary."""
        registry = cls.empty()

        # Load entity types
        for et_data in data.get("entity_types", []):
            registry.entity_types.append(EntityType.from_dict(et_data))

        # Load attribute types
        for entity_type, attrs_data in data.get("attribute_types", {}).items():
            registry.attribute_types[entity_type] = [
                AttributeType.from_dict(attr) for attr in attrs_data
            ]

        # Load relation types
        for rt_data in data.get("relation_types", []):
            registry.relation_types.append(RelationType.from_dict(rt_data))

        return registry

    # =========================================================================
    # Public Methods - Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the type registry."""
        total_attributes: int = sum(
            len(attrs) for attrs in self.attribute_types.values()
        )

        return {
            "entity_type_count": len(self.entity_types),
            "relation_type_count": len(self.relation_types),
            "total_attributes": total_attributes,
            "entity_types": [et.name for et in self.entity_types],
            "relation_types": [rt.name for rt in self.relation_types],
        }

    def get_role_inference_summary(self) -> Dict[str, Any]:
        """Summarize inferred roles across the registry."""
        by_entity: Dict[str, Dict[str, Any]] = {}
        role_counts: Dict[str, int] = {}

        for entity_type, attrs in self.attribute_types.items():
            entity_summary: Dict[str, Any] = {}
            for attr in attrs:
                role_name = attr.predicted_role.value if attr.predicted_role else None
                if role_name:
                    role_counts[role_name] = role_counts.get(role_name, 0) + 1
                entity_summary[attr.name] = {
                    "role": role_name,
                    "target_table": attr.target_table,
                    "confidence": attr.role_confidence,
                    "evidence": list(attr.role_evidence),
                }
            if entity_summary:
                by_entity[entity_type] = entity_summary

        return {
            "role_counts": role_counts,
            "by_entity": by_entity,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        stats = self.get_statistics()
        return (
            f"TypeRegistry("
            f"entities={stats['entity_type_count']}, "
            f"relations={stats['relation_type_count']}, "
            f"attributes={stats['total_attributes']})"
        )
