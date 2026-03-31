"""
Utility functions for schema format conversion and validation.

This module provides utilities to convert between different schema representations
and validate schema consistency.
"""

import logging
from typing import Dict, List, Optional, Set

from .table_schema import SchemaRegistry, TableSchema, ColumnDefinition

logger = logging.getLogger("SchemaUtils")


def sql_list_to_registry(sql_statements: List[str]) -> SchemaRegistry:
    """Convert list of SQL statements to SchemaRegistry."""
    return SchemaRegistry.from_sql_list(sql_statements)


def registry_to_sql_list(registry: SchemaRegistry) -> List[str]:
    """Convert SchemaRegistry to list of SQL statements."""
    return registry.to_sql_list()


def validate_schema_consistency(registry: SchemaRegistry) -> Dict[str, List[str]]:
    """
    Validate schema consistency and return any issues found.

    Returns:
        Dict with issue categories as keys and lists of issues as values
    """
    issues = {
        "missing_primary_keys": [],
        "invalid_foreign_keys": [],
        "missing_referenced_tables": [],
        "duplicate_column_names": [],
        "data_type_issues": []
    }

    table_names = registry.get_table_names()

    for table_name, table in registry.tables.items():
        # Check for primary key
        if not table.primary_keys:
            issues["missing_primary_keys"].append(f"Table {table_name} has no primary key")

        # Check foreign key references
        for col_name, ref_table, ref_column in table.foreign_keys:
            if ref_table not in table_names:
                issues["missing_referenced_tables"].append(
                    f"Table {table_name}.{col_name} references non-existent table {ref_table}"
                )
            else:
                ref_table_obj = registry.get_table(ref_table)
                if ref_table_obj and ref_column not in ref_table_obj.columns:
                    issues["invalid_foreign_keys"].append(
                        f"Table {table_name}.{col_name} references non-existent column {ref_table}.{ref_column}"
                    )

        # Check for duplicate columns (shouldn't happen but good to verify)
        col_names = list(table.columns.keys())
        if len(col_names) != len(set(col_names)):
            duplicates = [name for name in col_names if col_names.count(name) > 1]
            issues["duplicate_column_names"].append(
                f"Table {table_name} has duplicate columns: {duplicates}"
            )

        # Basic data type validation
        for col_name, col_def in table.columns.items():
            if not col_def.data_type:
                issues["data_type_issues"].append(
                    f"Table {table_name}.{col_name} has no data type"
                )

    return issues


def compare_schemas(old_registry: SchemaRegistry, new_registry: SchemaRegistry) -> Dict[str, any]:
    """
    Compare two schema registries and return differences.

    Returns:
        Dict with added_tables, removed_tables, modified_tables
    """
    old_tables = old_registry.get_table_names()
    new_tables = new_registry.get_table_names()

    added_tables = new_tables - old_tables
    removed_tables = old_tables - new_tables
    common_tables = old_tables & new_tables

    modified_tables = []
    for table_name in common_tables:
        old_table = old_registry.get_table(table_name)
        new_table = new_registry.get_table(table_name)

        if old_table and new_table:
            # Compare table schemas
            if old_table.to_sql() != new_table.to_sql():
                modified_tables.append(table_name)

    return {
        "added_tables": list(added_tables),
        "removed_tables": list(removed_tables),
        "modified_tables": modified_tables,
        "summary": {
            "total_changes": len(added_tables) + len(removed_tables) + len(modified_tables),
            "tables_added": len(added_tables),
            "tables_removed": len(removed_tables),
            "tables_modified": len(modified_tables)
        }
    }


def merge_registries(base_registry: SchemaRegistry, overlay_registry: SchemaRegistry) -> SchemaRegistry:
    """
    Merge two registries, with overlay taking precedence for conflicts.

    Args:
        base_registry: Base schema registry
        overlay_registry: Registry with changes to apply

    Returns:
        New merged registry
    """
    merged = SchemaRegistry()

    # Add all tables from base
    for table in base_registry.tables.values():
        merged.add_table(table)

    # Add/override with tables from overlay
    for table in overlay_registry.tables.values():
        merged.add_table(table)

    return merged


def extract_table_dependencies(registry: SchemaRegistry) -> Dict[str, Set[str]]:
    """
    Extract dependency graph showing which tables depend on which other tables.

    Returns:
        Dict mapping table names to sets of tables they depend on via foreign keys
    """
    dependencies = {}

    for table_name, table in registry.tables.items():
        deps = set()
        for _, ref_table, _ in table.foreign_keys:
            if ref_table != table_name:  # Avoid self-references
                deps.add(ref_table)
        dependencies[table_name] = deps

    return dependencies


def sort_tables_by_dependencies(registry: SchemaRegistry) -> List[str]:
    """
    Sort table names by dependency order (referenced tables first).

    This is useful for CREATE TABLE statement ordering to avoid foreign key errors.

    Returns:
        List of table names in dependency order
    """
    dependencies = extract_table_dependencies(registry)
    sorted_tables = []
    remaining_tables = set(registry.get_table_names())

    # Iterative dependency resolution
    while remaining_tables:
        # Find tables with no unresolved dependencies
        ready_tables = []
        for table_name in remaining_tables:
            table_deps = dependencies.get(table_name, set())
            unresolved_deps = table_deps & remaining_tables
            if not unresolved_deps:
                ready_tables.append(table_name)

        if not ready_tables:
            # Circular dependency or missing reference - just add remaining tables
            logger.warning(f"Circular dependencies detected in tables: {remaining_tables}")
            ready_tables = list(remaining_tables)

        # Add ready tables and remove from remaining
        for table_name in ready_tables:
            sorted_tables.append(table_name)
            remaining_tables.discard(table_name)

    return sorted_tables


def create_ordered_schema(registry: SchemaRegistry) -> List[str]:
    """
    Create schema SQL statements in dependency order.

    Returns:
        List of CREATE TABLE statements ordered by dependencies
    """
    ordered_table_names = sort_tables_by_dependencies(registry)
    ordered_statements = []

    for table_name in ordered_table_names:
        table = registry.get_table(table_name)
        if table:
            ordered_statements.append(table.to_sql())

    return ordered_statements