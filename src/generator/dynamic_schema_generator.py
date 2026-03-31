"""Dynamic schema generator for creating tables from entity types and attributes.

This module provides functionality to generate PostgreSQL CREATE TABLE statements
programmatically from discovered entity types, attributes, and relations.
No LLM is used - schema is generated deterministically.
"""

import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Set

from omegaconf import DictConfig

from src.aligned_db.relationship_detector import RelationshipDetector
from src.aligned_db.schema_registry import (
    ColumnInfo,
    ForeignKeyConstraint,
    SchemaRegistry,
    TableSchema,
)
from src.aligned_db.type_registry import TypeRegistry

logger = logging.getLogger("DynamicSchemaGenerator")


class DynamicSchemaGenerator:
    """Generates schema programmatically from discovered entity types and attributes.

    Creates normalized PostgreSQL CREATE TABLE statements based on discovered
    entity types, attributes (with data types), and relation types (for junction
    tables). All generation is deterministic - no LLM calls.

    Attributes:
        global_cfg: Global configuration
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the DynamicSchemaGenerator.

        Args:
            api_cfg: LLM API configuration (kept for backward compatibility, unused)
            global_cfg: Global configuration
        """
        self.global_cfg = global_cfg

    # =========================================================================
    # Public Methods
    # =========================================================================

    def generate(
        self,
        entity_types: List[Dict[str, str]],
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> SchemaRegistry:
        """Generate normalized schema for discovered entities.

        Creates tables programmatically from entity types and attributes,
        then enhances with junction tables and foreign key relationships.

        Args:
            entity_types: List of entity type dictionaries with name and description
            normalized_attributes: Dict mapping entity_type to list of attribute dicts

        Returns:
            SchemaRegistry containing the generated schema with relationships
        """
        # Filter out junction-table-like entity types (entity1_entity2 pattern)
        entity_name_set: Set[str] = {
            et.get("name", "").lower() for et in entity_types if et.get("name")
        }
        filtered_entity_types, filtered_attributes = self._filter_junction_entities(
            entity_types, normalized_attributes, entity_name_set
        )

        logger.info(f"Generating schema for {len(filtered_entity_types)} entity types")

        # Generate schema programmatically
        schema_registry = self._generate_tables(filtered_entity_types, filtered_attributes)

        logger.info(
            f"Base schema: {len(schema_registry.get_table_names())} tables"
        )

        # Enhance with relationships (junction tables and FK columns)
        schema_registry = self._enhance_with_relationships(
            schema_registry, entity_types, normalized_attributes
        )

        logger.info(
            f"Schema with relationships: {len(schema_registry.get_table_names())} tables"
        )

        # Post-process: standardize naming and constraints
        self._post_process_schema(schema_registry)

        return schema_registry

    def generate_from_registry(
        self,
        type_registry: TypeRegistry,
    ) -> SchemaRegistry:
        """Generate schema from a TypeRegistry (Stage 3 of pipeline).

        Creates schema directly from TypeRegistry which contains:
        - Entity types → Tables
        - Attribute types (with data types) → Columns
        - Relation types → Junction tables

        Args:
            type_registry: TypeRegistry from Stages 1 and 2

        Returns:
            SchemaRegistry containing the generated schema
        """
        logger.info(f"Generating schema from TypeRegistry: {type_registry}")

        # Convert to internal format
        entity_types: List[Dict[str, str]] = [
            {"name": et.name, "description": et.description}
            for et in type_registry.entity_types
        ]

        normalized_attributes: Dict[str, List[Dict[str, Any]]] = {}
        for entity_name, attr_list in type_registry.attribute_types.items():
            normalized_attributes[entity_name] = [
                {
                    "canonical_name": attr.name,
                    "data_type": attr.data_type,
                    "is_unique": attr.is_unique,
                    "predicted_role": attr.predicted_role.value if attr.predicted_role else None,
                    "target_table": attr.target_table,
                    "role_confidence": attr.role_confidence,
                }
                for attr in attr_list
            ]

        # Generate base schema
        schema_registry = self.generate(entity_types, normalized_attributes)

        # Add explicit junction tables from relation_types
        for relation in type_registry.relation_types:
            if not schema_registry.get_table(relation.name):
                self._create_junction_table(
                    schema_registry,
                    relation.name,
                    relation.source_entity,
                    relation.target_entity,
                    relation.attributes,
                )

        # Apply schema hints (natural keys, unique constraints)
        self._apply_schema_hints(schema_registry, type_registry.get_schema_hints())

        logger.info(
            f"Schema generation complete: {len(schema_registry.get_table_names())} tables"
        )

        return schema_registry

    def generate_sql_list(
        self,
        entity_types: List[Dict[str, str]],
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        """Generate schema as a list of SQL statements.

        Args:
            entity_types: List of entity type dictionaries
            normalized_attributes: Dict mapping entity_type to attributes

        Returns:
            List of CREATE TABLE SQL statements
        """
        schema_registry = self.generate(entity_types, normalized_attributes)
        return schema_registry.to_sql_list()

    # =========================================================================
    # Protected Methods - Schema Generation
    # =========================================================================

    def _filter_junction_entities(
        self,
        entity_types: List[Dict[str, str]],
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
        entity_name_set: Set[str],
    ) -> tuple[List[Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
        """Filter out junction-table-like entity types.

        Entity types matching 'entity1_entity2' pattern where both parts are
        known entity types are filtered out (handled as junction tables).

        Args:
            entity_types: List of entity type dictionaries
            normalized_attributes: Dict mapping entity_type to attributes
            entity_name_set: Set of all entity type names

        Returns:
            Tuple of (filtered_entity_types, filtered_attributes)
        """
        filtered_types: List[Dict[str, str]] = []
        filtered_attrs: Dict[str, List[Dict[str, Any]]] = {}

        for et in entity_types:
            name: str = et.get("name", "").lower()
            if self._is_junction_pattern(name, entity_name_set):
                logger.warning(
                    f"Filtering junction-like entity: '{name}' → junction table"
                )
                continue
            filtered_types.append(et)
            if name in normalized_attributes:
                filtered_attrs[name] = normalized_attributes[name]

        return filtered_types, filtered_attrs

    def _is_junction_pattern(self, name: str, entity_names: Set[str]) -> bool:
        """Check if name matches entity1_entity2 junction pattern."""
        parts = name.split("_")
        return (
            len(parts) == 2
            and parts[0] in entity_names
            and parts[1] in entity_names
        )

    def _generate_tables(
        self,
        entity_types: List[Dict[str, str]],
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> SchemaRegistry:
        """Generate tables for all entity types.

        Args:
            entity_types: List of entity type dictionaries
            normalized_attributes: Dict mapping entity_type to attributes

        Returns:
            SchemaRegistry with all entity tables
        """
        schema_registry = SchemaRegistry()

        for entity_info in entity_types:
            entity_type: str = entity_info.get("name", "").lower()
            if not entity_type:
                continue

            table = self._create_table(
                entity_type,
                normalized_attributes.get(entity_type, []),
            )
            schema_registry.add_table(table)
            logger.debug(f"  Created table: {entity_type} ({len(table.columns)} cols)")

        return schema_registry

    def _create_table(
        self,
        entity_type: str,
        attributes: List[Dict[str, Any]],
    ) -> TableSchema:
        """Create a table with standard naming conventions.

        Structure:
        - Primary key: {entity_type}_id SERIAL PRIMARY KEY
        - Natural key: 'name' TEXT UNIQUE (or 'title' for work)
        - Attribute columns with inferred types

        Args:
            entity_type: Name of the entity type
            attributes: List of attribute dictionaries

        Returns:
            TableSchema for the entity type
        """
        columns: List[ColumnInfo] = []
        seen_columns: Set[str] = set()

        # Primary key
        pk_name = f"{entity_type}_id"
        columns.append(
            ColumnInfo(name=pk_name, data_type="SERIAL", is_primary_key=True)
        )
        seen_columns.add(pk_name)

        # Natural key column (name for most, title for work)
        natural_key = "title" if entity_type == "work" else "name"
        columns.append(
            ColumnInfo(name=natural_key, data_type="TEXT", is_unique=True)
        )
        seen_columns.add(natural_key)

        # Attribute columns
        for attr in attributes:
            col_name = attr.get("canonical_name", attr.get("name", "")).lower()
            if not col_name or col_name in seen_columns:
                continue

            # Skip redundant name variants
            if self._is_redundant_name_column(col_name, entity_type):
                continue

            data_type = self._infer_column_type(col_name, attr.get("data_type", ""))
            columns.append(ColumnInfo(name=col_name, data_type=data_type))
            seen_columns.add(col_name)

        return TableSchema(name=entity_type, columns=columns)

    def _is_redundant_name_column(self, col_name: str, entity_type: str) -> bool:
        """Check if column name is redundant with standard natural key."""
        if entity_type == "work":
            return False  # work uses 'title', not 'name'
        # Skip {entity}_name, full_name, title (redundant with 'name')
        return col_name in (f"{entity_type}_name", "full_name", "title")

    def _infer_column_type(self, column_name: str, provided_type: str = "") -> str:
        """Infer PostgreSQL column type from column name.

        Args:
            column_name: Name of the column
            provided_type: Optional type hint from attribute discovery

        Returns:
            PostgreSQL data type string
        """
        # Use provided type if valid
        if provided_type:
            ptype = provided_type.upper()
            if ptype in ("TEXT", "INTEGER", "BOOLEAN", "DATE", "FLOAT", "SERIAL"):
                return ptype

        col = column_name.lower()

        # Date patterns
        if any(p in col for p in ("_date", "date_", "birth_date", "death_date", "published")):
            return "DATE"

        # Year patterns
        if col.endswith("_year") or col in ("year", "birth_year", "death_year"):
            return "INTEGER"

        # Boolean patterns
        if col.startswith("is_") or col.startswith("has_"):
            return "BOOLEAN"

        # Count patterns
        if col.endswith("_count") or col in ("count", "quantity", "number"):
            return "INTEGER"

        return "TEXT"

    # =========================================================================
    # Protected Methods - Relationships
    # =========================================================================

    def _enhance_with_relationships(
        self,
        schema_registry: SchemaRegistry,
        entity_types: List[Dict[str, str]],
        normalized_attributes: Dict[str, List[Dict[str, Any]]],
    ) -> SchemaRegistry:
        """Enhance schema with junction tables and foreign key columns.

        Uses RelationshipDetector to analyze attributes and detect:
        - Many-to-many relationships → junction tables
        - Many-to-one relationships → FK columns

        Args:
            schema_registry: Base schema to enhance
            entity_types: List of entity type dictionaries
            normalized_attributes: Dict mapping entity_type to attributes

        Returns:
            Enhanced SchemaRegistry
        """
        enable_detection: bool = self.global_cfg.model.aligned_db.get(
            "enable_relationship_detection", True
        )
        if not enable_detection:
            logger.info("Relationship detection disabled")
            return schema_registry

        logger.info("Detecting and creating relationships...")

        # Filter junction-like entity types
        entity_name_set: Set[str] = {
            et.get("name", "").lower() for et in entity_types if et.get("name")
        }
        filtered_types = [
            et for et in entity_types
            if not self._is_junction_pattern(et.get("name", "").lower(), entity_name_set)
        ]

        # Detect and add relationships
        detector = RelationshipDetector(filtered_types)
        return detector.enhance_schema(schema_registry, normalized_attributes)

    def _create_junction_table(
        self,
        schema_registry: SchemaRegistry,
        table_name: str,
        source_entity: str,
        target_entity: str,
        additional_attributes: List[Any],
    ) -> None:
        """Create a junction table for a many-to-many relationship.

        Creates table with:
        - Composite PRIMARY KEY (source_id, target_id)
        - FOREIGN KEY constraints to both entity tables

        Args:
            schema_registry: SchemaRegistry to add table to
            table_name: Name of the junction table
            source_entity: Source entity type name
            target_entity: Target entity type name
            additional_attributes: Additional AttributeType objects
        """
        source_col = f"{source_entity}_id"
        target_col = f"{target_entity}_id"

        # Columns
        columns: List[ColumnInfo] = [
            ColumnInfo(name=source_col, data_type="INTEGER", is_nullable=False),
            ColumnInfo(name=target_col, data_type="INTEGER", is_nullable=False),
        ]

        # Additional attributes
        for attr in additional_attributes:
            attr_name = attr.name if hasattr(attr, "name") else str(attr)
            attr_type = attr.data_type if hasattr(attr, "data_type") else "TEXT"
            columns.append(
                ColumnInfo(name=attr_name, data_type=attr_type, is_nullable=True)
            )

        # Foreign key constraints
        foreign_keys: List[ForeignKeyConstraint] = [
            ForeignKeyConstraint(
                column_name=source_col,
                references_table=source_entity,
                references_column=source_col,
            ),
            ForeignKeyConstraint(
                column_name=target_col,
                references_table=target_entity,
                references_column=target_col,
            ),
        ]

        table = TableSchema(
            name=table_name,
            columns=columns,
            foreign_keys=foreign_keys,
            primary_key_columns=[source_col, target_col],
        )
        schema_registry.add_table(table)
        logger.debug(f"Created junction table: {table_name}")

    # =========================================================================
    # Protected Methods - Post-processing
    # =========================================================================

    def _post_process_schema(self, schema_registry: SchemaRegistry) -> None:
        """Apply post-processing to standardize schema.

        Args:
            schema_registry: SchemaRegistry to process
        """
        # Standardize {entity}_name columns to 'name'
        renamed = schema_registry.standardize_name_columns()
        if renamed > 0:
            logger.info(f"Post-process: renamed {renamed} columns to 'name'")

        # Ensure natural keys have UNIQUE constraint
        unique_added = schema_registry.ensure_natural_keys_unique()
        if unique_added > 0:
            logger.info(f"Post-process: marked {unique_added} natural keys UNIQUE")

        # Enforce single UNIQUE per table (for UPSERT ON CONFLICT)
        fixed = schema_registry.enforce_single_unique_constraint()
        if fixed > 0:
            logger.info(f"Post-process: fixed {fixed} tables with multiple UNIQUEs")

    def _apply_schema_hints(
        self,
        schema_registry: SchemaRegistry,
        hints: Dict[str, Dict[str, Any]],
    ) -> None:
        """Apply schema hints from TypeRegistry.

        Args:
            schema_registry: SchemaRegistry to update
            hints: Dict mapping table name to hints dict
        """
        for table_name, table_hints in hints.items():
            table: Optional[TableSchema] = schema_registry.get_table(table_name)
            if not table:
                continue

            # Apply natural key
            natural_key = table_hints.get("natural_key")
            if natural_key:
                for col in table.columns:
                    if col.name == natural_key:
                        col.is_unique = True
                        break

            # Apply unique constraints
            for col_name in table_hints.get("unique_columns", []):
                for col in table.columns:
                    if col.name == col_name:
                        col.is_unique = True
                        break
