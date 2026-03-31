"""Fast heuristic checks for schema post-processing decisions.

This module provides fast O(n) heuristic checks to determine if schema
modifications are needed, allowing expensive LLM calls to be skipped when
modifications are clearly not needed.
"""

import re
from collections import Counter
from typing import List

# Columns that indicate denormalized parent/occupation data in author table
DENORMALIZED_PARENT_COLUMNS: List[str] = [
    "father_name",
    "mother_name",
    "father_occupation",
    "mother_occupation",
]


class SchemaHeuristicChecker:
    """Fast heuristic checks to determine if schema modifications are needed.

    These checks run in O(n) time on the schema string and avoid LLM calls
    when modifications are clearly not needed. Each check returns True if
    the modification step should be run, False if it can be skipped.
    """

    @staticmethod
    def needs_primary_key_modification(schema: List[str]) -> bool:
        """Check if any table lacks a PRIMARY KEY constraint.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if at least one table lacks a PRIMARY KEY constraint
        """
        schema_str: str = "\n".join(schema).upper()
        table_count: int = schema_str.count("CREATE TABLE")
        pk_count: int = schema_str.count("PRIMARY KEY")
        return pk_count < table_count

    @staticmethod
    def needs_foreign_key_modification(schema: List[str]) -> bool:
        """Check if schema might need additional FK relationships.

        Heuristic: If there are more _id columns than the sum of
        REFERENCES clauses and PRIMARY KEYs, some might be missing FK definitions.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if schema might benefit from foreign key modifications
        """
        schema_str: str = "\n".join(schema)
        # Count _id columns that could be foreign keys
        id_columns: List[str] = re.findall(
            r"(\w+_id)\s+INTEGER", schema_str, re.IGNORECASE
        )
        fk_refs: List[str] = re.findall(r"REFERENCES\s+(\w+)", schema_str, re.IGNORECASE)
        pk_count: int = len(re.findall(r"PRIMARY KEY", schema_str, re.IGNORECASE))
        # More ID columns than FKs + PKs suggests missing relationships
        return len(id_columns) > len(fk_refs) + pk_count

    @staticmethod
    def needs_order_modification(schema: List[str]) -> bool:
        """Check if tables might be in wrong order (referenced before defined).

        Ensures tables are ordered so that a table's FK references are defined
        before that table appears.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if any table references a table defined later in the schema
        """
        tables_order: List[str] = []
        references: dict = {}

        for stmt in schema:
            match = re.search(r"CREATE TABLE\s+(\w+)", stmt, re.IGNORECASE)
            if match:
                table_name: str = match.group(1).lower()
                tables_order.append(table_name)
                refs: List[str] = [
                    r.lower()
                    for r in re.findall(r"REFERENCES\s+(\w+)", stmt, re.IGNORECASE)
                ]
                references[table_name] = refs

        # Check if any table references a table defined later
        for i, table in enumerate(tables_order):
            for ref in references.get(table, []):
                if ref in tables_order[i + 1 :]:
                    return True
        return False

    @staticmethod
    def needs_unique_key_modification(schema: List[str]) -> bool:
        """Check if schema might benefit from UNIQUE constraints.

        Heuristic: If no UNIQUE constraints exist, the schema might need them.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if schema has no UNIQUE constraints
        """
        schema_str: str = "\n".join(schema).upper()
        return "UNIQUE" not in schema_str

    @staticmethod
    def needs_normalization(schema: List[str]) -> bool:
        """Check if schema might have redundancy issues.

        Heuristic: Non-id columns appearing in multiple tables suggest
        data redundancy that could be normalized. Also checks for
        denormalized parent/occupation columns in the author table.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if normalization is needed (redundant columns or denormalized parent data)
        """
        schema_str: str = "\n".join(schema).lower()

        # Check 1: Look for denormalized parent/occupation columns in author table
        # These columns should be in separate parent/occupation tables
        for col in DENORMALIZED_PARENT_COLUMNS:
            if col in schema_str:
                return True

        # Check 2: Verify parent and occupation tables exist (if author table exists)
        has_author_table: bool = "create table author" in schema_str
        has_parent_table: bool = "create table parent" in schema_str
        has_occupation_table: bool = "create table occupation" in schema_str

        if has_author_table and (not has_parent_table or not has_occupation_table):
            # Author table exists but missing normalized parent/occupation tables
            return True

        # Check 3: Original heuristic - repeated non-id columns across tables
        all_columns: List[str] = []
        for stmt in schema:
            columns: List[str] = re.findall(
                r"(\w+)\s+(?:VARCHAR|TEXT|INTEGER|DATE|BOOLEAN)", stmt, re.IGNORECASE
            )
            all_columns.extend([c.lower() for c in columns])

        # Same non-id column appearing in multiple tables suggests redundancy
        column_counts: Counter = Counter(all_columns)
        repeated: List[str] = [
            c for c, count in column_counts.items() if count > 1 and not c.endswith("_id")
        ]
        return len(repeated) > 2

    @staticmethod
    def needs_column_deduplication(schema: List[str]) -> bool:
        """Check if schema has potentially duplicate columns.

        Heuristic: Multiple columns related to 'name' or 'title' might be
        semantically equivalent and could be deduplicated.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            True if there are many name-related or title-related columns
        """
        all_columns: List[str] = []
        for stmt in schema:
            columns: List[str] = re.findall(
                r"(\w+)\s+(?:VARCHAR|TEXT|INTEGER|DATE|BOOLEAN)", stmt, re.IGNORECASE
            )
            all_columns.extend([c.lower() for c in columns])

        # Look for similar column names that might be duplicates
        name_related: List[str] = [c for c in all_columns if "name" in c]
        title_related: List[str] = [c for c in all_columns if "title" in c]
        return len(name_related) > 2 or len(title_related) > 2

