"""Runtime services used during aligned DB upsert generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import hkkang_utils.pg as pg_utils

from src.aligned_db.entity_lookup import (
    auto_create_missing_referenced_entities,
    build_fk_subquery,
    get_entity_lookup_column,
    get_lookup_column_from_db,
)
from src.aligned_db.entity_grounding import (
    ground_entity_references as ground_registry_entity_references,
)
from src.aligned_db.entity_registry import EntityRegistry
from src.aligned_db.grounding_resolver import GroundingResolver
from src.aligned_db.junction_upserts import (
    create_missing_junction_tables,
    generate_junction_table_upserts,
)
from src.aligned_db.schema_registry import ForeignKeyConstraint, SchemaRegistry, TableSchema
from src.aligned_db.sql_values import (
    build_self_ref_fk_update,
    escape_value,
    get_self_referential_fk_columns,
    normalize_entity_attributes,
)
from src.aligned_db.upsert_support import get_conflict_key, resolve_conflict_key


@dataclass
class AlignedDBUpsertRuntime:
    """Stateful helper object for entity and junction upsert generation."""

    pg_client: pg_utils.PostgresConnector
    lookup_column_cache: Dict[str, Optional[str]] = field(default_factory=dict)
    grounding_summary: Dict[str, Any] = field(default_factory=dict)

    def clear_lookup_column_cache(self) -> None:
        """Clear cached lookup-column metadata after schema changes."""
        self.lookup_column_cache.clear()

    def get_entity_lookup_column(
        self,
        entity_type: str,
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> str:
        """Resolve the best lookup column for an entity table."""
        return get_entity_lookup_column(
            pg_client=self.pg_client,
            entity_type=entity_type,
            schema_registry=schema_registry,
            cache=self.lookup_column_cache,
        )

    def get_lookup_column_from_db(self, table_name: str) -> Optional[str]:
        """Query the live DB schema for the preferred lookup column."""
        return get_lookup_column_from_db(
            pg_client=self.pg_client,
            table_name=table_name,
            cache=self.lookup_column_cache,
        )

    def build_fk_subquery(
        self,
        fk_constraint: ForeignKeyConstraint,
        value: str,
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> str:
        """Build a FK-resolution subquery for a string entity reference."""
        return build_fk_subquery(
            fk_constraint=fk_constraint,
            value=value,
            schema_registry=schema_registry,
            get_entity_lookup_column_fn=self.get_entity_lookup_column,
            escape_value_fn=self.escape_value,
        )

    def auto_create_missing_referenced_entities(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
    ) -> None:
        """Create stub referenced entities needed to satisfy FK-bearing tables."""
        auto_create_missing_referenced_entities(
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            get_entity_lookup_column_fn=self.get_entity_lookup_column,
        )

    def ground_entity_references(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
    ) -> None:
        """Ground FK-bearing entity attributes onto canonical lookup values."""
        diagnostics = ground_registry_entity_references(
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            get_entity_lookup_column_fn=self.get_entity_lookup_column,
        )
        self.grounding_summary = diagnostics.to_dict()

    def build_grounding_resolver(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
    ) -> GroundingResolver:
        """Build a reusable grounding resolver for the current registry state."""
        return GroundingResolver(
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            get_entity_lookup_column_fn=self.get_entity_lookup_column,
        )

    def generate_junction_table_upserts(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
    ) -> tuple[List[str], Set[str]]:
        """Generate INSERT statements for junction-table relationships."""
        return generate_junction_table_upserts(
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            runtime=self,
        )

    def create_missing_junction_tables(
        self,
        schema_registry: SchemaRegistry,
        entity_registry: EntityRegistry,
        missing_tables: Set[str],
    ) -> int:
        """Create missing junction tables in the DB and schema registry."""
        return create_missing_junction_tables(
            pg_client=self.pg_client,
            schema_registry=schema_registry,
            entity_registry=entity_registry,
            missing_tables=missing_tables,
        )

    def get_conflict_key(
        self,
        entity_type: str,
        table: Optional[TableSchema],
    ) -> Optional[str]:
        """Return the preferred conflict key for an entity table."""
        return get_conflict_key(entity_type, table)

    def get_self_referential_fk_columns(self, table: TableSchema) -> Set[str]:
        """Return FK columns that point back to the same table."""
        return get_self_referential_fk_columns(table)

    def normalize_entity_attributes(
        self,
        entity: Dict[str, Any],
        entity_type: str,
        valid_columns: List[str],
    ) -> Dict[str, Any]:
        """Normalize extracted entity fields to match schema columns."""
        return normalize_entity_attributes(
            entity=entity,
            entity_type=entity_type,
            valid_columns=valid_columns,
        )

    def escape_value(self, value: Any, column_type: Optional[str] = None) -> str:
        """Escape a value for SQL insertion."""
        return escape_value(value, column_type)

    def resolve_conflict_key(
        self,
        conflict_key: Optional[str],
        columns: List[str],
        entity_type: str,
    ) -> Optional[str]:
        """Map a requested conflict key onto actual inserted columns."""
        return resolve_conflict_key(conflict_key, columns, entity_type)

    def build_self_ref_fk_update(
        self,
        entity_type: str,
        entity: Dict[str, Any],
        self_ref_fk_values: Dict[str, str],
        fk_column_map: Dict[str, ForeignKeyConstraint],
        conflict_key: Optional[str],
        schema_registry: SchemaRegistry,
    ) -> Optional[str]:
        """Build an UPDATE for deferred self-referential FK resolution."""
        return build_self_ref_fk_update(
            entity_type=entity_type,
            entity=entity,
            self_ref_fk_values=self_ref_fk_values,
            fk_column_map=fk_column_map,
            conflict_key=conflict_key,
            schema_registry=schema_registry,
            build_fk_subquery=self.build_fk_subquery,
            escape_value=self.escape_value,
        )
