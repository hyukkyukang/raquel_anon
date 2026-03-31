"""Schema registry for tracking and evolving database schemas.

This module provides data structures for managing database schemas dynamically,
supporting schema evolution via ALTER TABLE when new attributes are discovered.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.string import sanitize_sql_identifier

logger = logging.getLogger("SchemaRegistry")

# SQL reserved words that must be quoted when used as identifiers
SQL_RESERVED_WORDS: Set[str] = {
    "select",
    "from",
    "where",
    "insert",
    "update",
    "delete",
    "set",
    "values",
    "into",
    "create",
    "drop",
    "alter",
    "table",
    "index",
    "and",
    "or",
    "not",
    "null",
    "true",
    "false",
    "join",
    "on",
    "order",
    "by",
    "group",
    "having",
    "limit",
    "offset",
    "union",
    "references",
    "foreign",
    "key",
    "primary",
    "constraint",
    "unique",
    "check",
    "default",
    "as",
    "is",
    "in",
    "like",
    "between",
    "exists",
    "case",
    "when",
    "then",
    "else",
    "end",
    "all",
    "any",
    "some",
    "distinct",
    "asc",
    "desc",
    "with",
    "grant",
    "revoke",
    "user",
    "role",
    "schema",
    "database",
    "trigger",
    "function",
    "procedure",
    "view",
}


def _quote_identifier(name: str) -> str:
    """Quote an identifier if it's a SQL reserved word.

    Args:
        name: Column or table name

    Returns:
        Quoted name if reserved word, otherwise original name
    """
    if name.lower() in SQL_RESERVED_WORDS:
        return f'"{name}"'
    return name


@dataclass
class ColumnInfo:
    """Information about a table column.

    Attributes:
        name: Column name
        data_type: PostgreSQL data type (TEXT, INTEGER, DATE, etc.)
        is_primary_key: Whether this is the primary key
        is_foreign_key: Whether this references another table
        is_unique: Whether this column has a UNIQUE constraint
        references: Foreign key reference in format "table_name(column_name)"
        is_nullable: Whether NULL values are allowed
        default: Default value expression
    """

    name: str
    data_type: str = "TEXT"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    references: Optional[str] = None
    is_nullable: bool = True
    default: Optional[str] = None

    def __post_init__(self) -> None:
        """Sanitize column name after initialization.

        Ensures column names are valid SQL identifiers by:
        - Converting to lowercase
        - Replacing spaces/hyphens with underscores
        - Removing invalid characters
        """
        if self.name:
            sanitized = sanitize_sql_identifier(self.name, default="unnamed_column")
            if sanitized != self.name:
                logger.debug(f"Sanitized column name: '{self.name}' -> '{sanitized}'")
            self.name = sanitized

    def to_sql_definition(self) -> str:
        """Generate SQL column definition.

        Returns:
            SQL fragment for column definition
        """
        # Quote column name if it's a SQL reserved word
        quoted_name = _quote_identifier(self.name)
        parts = [quoted_name, self.data_type]

        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        else:
            if self.is_unique:
                parts.append("UNIQUE")
            if not self.is_nullable:
                parts.append("NOT NULL")

        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "is_unique": self.is_unique,
            "references": self.references,
            "is_nullable": self.is_nullable,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnInfo":
        """Create ColumnInfo from dictionary."""
        return cls(
            name=data.get("name", ""),
            data_type=data.get("data_type", "TEXT"),
            is_primary_key=data.get("is_primary_key", False),
            is_foreign_key=data.get("is_foreign_key", False),
            is_unique=data.get("is_unique", False),
            references=data.get("references"),
            is_nullable=data.get("is_nullable", True),
            default=data.get("default"),
        )


@dataclass
class ForeignKeyConstraint:
    """Foreign key constraint information.

    Attributes:
        column_name: Local column name
        references_table: Referenced table name
        references_column: Referenced column name
    """

    column_name: str
    references_table: str
    references_column: str

    def __post_init__(self) -> None:
        self.column_name = sanitize_sql_identifier(
            self.column_name, default="unnamed_column"
        )
        self.references_table = sanitize_sql_identifier(
            self.references_table, default="unnamed_table"
        )
        self.references_column = sanitize_sql_identifier(
            self.references_column, default="unnamed_column"
        )

    def to_sql(self) -> str:
        """Generate SQL foreign key constraint.

        Returns:
            SQL fragment for foreign key constraint
        """
        quoted_col = _quote_identifier(self.column_name)
        quoted_ref_table = _quote_identifier(self.references_table)
        quoted_ref_col = _quote_identifier(self.references_column)
        return (
            f"FOREIGN KEY ({quoted_col}) "
            f"REFERENCES {quoted_ref_table}({quoted_ref_col})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "column_name": self.column_name,
            "references_table": self.references_table,
            "references_column": self.references_column,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForeignKeyConstraint":
        """Create ForeignKeyConstraint from dictionary."""
        references_table = data.get("references_table", "")
        references_column = data.get("references_column", "")
        column_name = data.get("column_name", "")

        if (
            not references_table
            and not references_column
            and not column_name
            and data.get("column")
            and data.get("references")
        ):
            column_name = data.get("column", "")
            references = str(data.get("references", ""))
            match = re.match(r"^\s*([^(]+)\(([^)]+)\)\s*$", references)
            if match:
                references_table = match.group(1).strip()
                references_column = match.group(2).strip()

        return cls(
            column_name=column_name,
            references_table=references_table,
            references_column=references_column,
        )


@dataclass
class TableSchema:
    """Schema for a single database table.

    Attributes:
        name: Table name
        columns: List of column definitions
        foreign_keys: List of foreign key constraints
        primary_key_columns: List of primary key column names (for composite PKs)
    """

    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyConstraint] = field(default_factory=list)
    primary_key_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Deduplicate columns after initialization.

        This ensures no duplicate columns exist even when columns list is
        passed directly to the constructor (bypassing add_column).
        """
        self.name = sanitize_sql_identifier(self.name, default="unnamed_table")
        self.primary_key_columns = [
            sanitize_sql_identifier(col, default="unnamed_column")
            for col in self.primary_key_columns
        ]
        self.foreign_keys = [
            ForeignKeyConstraint(
                column_name=fk.column_name,
                references_table=fk.references_table,
                references_column=fk.references_column,
            )
            for fk in self.foreign_keys
        ]

        if self.columns:
            seen: Set[str] = set()
            deduplicated: List[ColumnInfo] = []
            for col in self.columns:
                col_name_lower = col.name.lower()
                if col_name_lower not in seen:
                    # Normalize column name to lowercase
                    col.name = col_name_lower
                    deduplicated.append(col)
                    seen.add(col_name_lower)
                else:
                    logger.warning(
                        f"TableSchema.__post_init__: Removing duplicate column "
                        f"'{col.name}' from table '{self.name}'"
                    )
            self.columns = deduplicated

    # =========================================================================
    # Query Methods
    # =========================================================================

    def has_column(self, column_name: str) -> bool:
        """Check if a column exists in this table.

        Args:
            column_name: Column name to check

        Returns:
            True if column exists
        """
        # Case-insensitive comparison to prevent duplicates with different casing
        column_name_lower = column_name.lower()
        return any(col.name.lower() == column_name_lower for col in self.columns)

    def get_column(self, column_name: str) -> Optional[ColumnInfo]:
        """Get a column by name.

        Args:
            column_name: Column name to find

        Returns:
            ColumnInfo if found, None otherwise
        """
        # Case-insensitive lookup for consistency
        column_name_lower = column_name.lower()
        for col in self.columns:
            if col.name.lower() == column_name_lower:
                return col
        return None

    def get_column_names(self) -> Set[str]:
        """Get all column names.

        Returns:
            Set of column names
        """
        return {col.name for col in self.columns}

    def get_primary_key(self) -> Optional[str]:
        """Get the primary key column name.

        Returns:
            Primary key column name, or None if no single PK
        """
        for col in self.columns:
            if col.is_primary_key:
                return col.name
        if len(self.primary_key_columns) == 1:
            return self.primary_key_columns[0]
        return None

    def get_conflict_key(self) -> Optional[str]:
        """Get the column to use for ON CONFLICT clause in UPSERT operations.

        Priority order:
        1. First column with UNIQUE constraint (excluding primary key)
        2. Heuristic detection (first non-PK, non-FK TEXT column)
        3. Primary key (fallback for junction tables)

        Returns:
            Column name for conflict resolution, or None if no suitable key
        """
        # Priority 1: Explicit unique column (non-PK)
        for col in self.columns:
            if col.is_unique and not col.is_primary_key:
                return col.name

        # Priority 2: Heuristic detection (first non-PK, non-FK TEXT column)
        natural_key = self._detect_natural_key()
        if natural_key:
            return natural_key

        # Priority 3: Primary key (fallback for junction tables)
        return self.get_primary_key()

    def _detect_natural_key(self) -> Optional[str]:
        """Detect the natural key column using heuristics.

        The natural key is the first non-PK, non-FK TEXT/VARCHAR column,
        which typically represents the semantic identifier of the entity
        (e.g., "name", "title", "label").

        Returns:
            Column name if a natural key is detected, None otherwise
        """
        pk_name: Optional[str] = self.get_primary_key()
        fk_columns: Set[str] = {fk.column_name for fk in self.foreign_keys}

        # Get non-PK, non-FK columns (entity's own attributes)
        non_pk_fk_columns: List[ColumnInfo] = [
            col
            for col in self.columns
            if col.name != pk_name and col.name not in fk_columns
        ]

        # Skip TRUE junction tables: tables where ALL non-PK columns are FKs
        # (e.g., person_occupation with only person_id and occupation_id)
        # But allow tables with substantial non-FK columns (e.g., work with title,
        # publication_year, etc. even though it has 3 FKs)
        if len(non_pk_fk_columns) == 0:
            return None

        # For tables with mostly FK columns (junction-like), skip if ratio is high
        # A table like work with 3 FKs and 10+ non-FK columns is NOT junction-like
        total_non_pk: int = len(self.columns) - (1 if pk_name else 0)
        if total_non_pk > 0:
            fk_ratio: float = len(fk_columns) / total_non_pk
            # Only skip if >70% of columns are FKs (true junction tables)
            if fk_ratio > 0.7 and len(non_pk_fk_columns) <= 1:
                return None

        # Find the first TEXT/VARCHAR column among non-FK columns
        for col in non_pk_fk_columns:
            if col.data_type.upper() in ("TEXT", "VARCHAR"):
                return col.name

        return None

    def is_junction_table(self) -> bool:
        """Check if this table is a junction table (many-to-many relationship).

        Junction tables typically have:
        - Composite primary key (two or more columns)
        - Only foreign key columns (no natural key columns)

        Returns:
            True if this appears to be a junction table
        """
        # Junction tables have composite primary keys
        if len(self.primary_key_columns) >= 2:
            return True

        # Or if all non-PK columns are foreign keys
        pk_name = self.get_primary_key()
        non_pk_cols = [col for col in self.columns if col.name != pk_name]
        fk_columns = {fk.column_name for fk in self.foreign_keys}

        if non_pk_cols and all(col.name in fk_columns for col in non_pk_cols):
            return True

        return False

    def is_junction_like_table(self) -> bool:
        """Check if this table is junction-like (2+ FK columns linking entities).

        Junction-like tables are tables that link two or more entities, possibly
        with additional attribute columns (e.g., work_character with role_in_work).
        These should use composite FK keys for conflict resolution, not attribute
        columns.

        Returns:
            True if this table has 2 or more foreign key columns
        """
        return len(self.foreign_keys) >= 2

    def get_composite_fk_columns(self) -> Optional[List[str]]:
        """Get the FK columns for composite conflict key in junction-like tables.

        For junction-like tables, the FK columns together form the natural
        unique identifier (e.g., work_id + character_id in work_character).

        Returns:
            List of FK column names if this is a junction-like table, None otherwise
        """
        if not self.is_junction_like_table():
            return None
        return [fk.column_name for fk in self.foreign_keys]

    # =========================================================================
    # Modification Methods
    # =========================================================================

    def add_column(self, column: ColumnInfo) -> None:
        """Add a new column to the table.

        Args:
            column: Column definition to add
        """
        # Normalize column name to lowercase for consistency
        column.name = column.name.lower()
        if not self.has_column(column.name):
            self.columns.append(column)
            logger.debug(f"Added column {column.name} to table {self.name}")

    def add_foreign_key(self, fk: ForeignKeyConstraint) -> None:
        """Add a foreign key constraint.

        Args:
            fk: Foreign key constraint to add
        """
        self.foreign_keys.append(fk)

    # =========================================================================
    # SQL Generation Methods
    # =========================================================================

    def to_create_sql(
        self,
        exclude_fks: Optional[Set[str]] = None,
        valid_tables: Optional[Set[str]] = None,
    ) -> str:
        """Generate CREATE TABLE SQL statement.

        Args:
            exclude_fks: Optional set of referenced table names whose FK constraints
                        should be excluded (for handling circular dependencies)
            valid_tables: Optional set of valid table names. If provided, FKs to
                         tables not in this set will be skipped.

        Returns:
            Complete CREATE TABLE statement
        """
        lines: List[str] = []
        exclude_fks = exclude_fks or set()

        # Column definitions (deduplicate by lowercase name as safety check)
        seen_columns: Set[str] = set()
        for col in self.columns:
            col_name_lower = col.name.lower()
            if col_name_lower in seen_columns:
                logger.warning(
                    f"Skipping duplicate column '{col.name}' in table '{self.name}'"
                )
                continue
            seen_columns.add(col_name_lower)
            lines.append(f"    {col.to_sql_definition()}")

        # Composite primary key (if any)
        if self.primary_key_columns and len(self.primary_key_columns) > 1:
            pk_cols = ", ".join(
                _quote_identifier(col) for col in self.primary_key_columns
            )
            lines.append(f"    PRIMARY KEY ({pk_cols})")

        # Foreign key constraints (excluding deferred ones and invalid refs)
        for fk in self.foreign_keys:
            ref_table_lower = fk.references_table.lower()
            # Skip if deferred
            if ref_table_lower in exclude_fks:
                continue
            # Skip if references non-existent table (but allow self-references)
            if valid_tables is not None:
                if ref_table_lower not in valid_tables and ref_table_lower != self.name.lower():
                    logger.debug(
                        f"Skipping FK {self.name}.{fk.column_name} -> "
                        f"{fk.references_table} (table does not exist)"
                    )
                    continue
            lines.append(f"    {fk.to_sql()}")

        columns_sql = ",\n".join(lines)
        quoted_table = _quote_identifier(self.name)
        return f"CREATE TABLE {quoted_table} (\n{columns_sql}\n);"

    def get_deferred_fk_statements(
        self, deferred_refs: Set[str]
    ) -> List[str]:
        """Generate ALTER TABLE statements for deferred FK constraints.

        Args:
            deferred_refs: Set of referenced table names that were deferred

        Returns:
            List of ALTER TABLE ADD CONSTRAINT statements
        """
        statements: List[str] = []
        for fk in self.foreign_keys:
            if fk.references_table.lower() in deferred_refs:
                statements.append(self.to_alter_add_foreign_key_sql(fk))
        return statements

    def to_alter_add_column_sql(self, column: ColumnInfo) -> str:
        """Generate ALTER TABLE ADD COLUMN SQL statement.

        Args:
            column: Column to add

        Returns:
            ALTER TABLE statement
        """
        quoted_table = _quote_identifier(self.name)
        col_def = column.to_sql_definition()
        return f"ALTER TABLE {quoted_table} ADD COLUMN {col_def};"

    def to_alter_add_foreign_key_sql(self, fk: ForeignKeyConstraint) -> str:
        """Generate ALTER TABLE ADD FOREIGN KEY SQL statement.

        Args:
            fk: Foreign key constraint to add

        Returns:
            ALTER TABLE statement
        """
        quoted_table = _quote_identifier(self.name)
        quoted_col = _quote_identifier(fk.column_name)
        quoted_ref_table = _quote_identifier(fk.references_table)
        quoted_ref_col = _quote_identifier(fk.references_column)
        constraint_name = f"fk_{self.name}_{fk.column_name}"
        return (
            f"ALTER TABLE {quoted_table} ADD CONSTRAINT {constraint_name} "
            f"FOREIGN KEY ({quoted_col}) "
            f"REFERENCES {quoted_ref_table}({quoted_ref_col});"
        )

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "columns": [col.to_dict() for col in self.columns],
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "primary_key_columns": self.primary_key_columns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSchema":
        """Create TableSchema from dictionary."""
        return cls(
            name=data.get("name", ""),
            columns=[ColumnInfo.from_dict(c) for c in data.get("columns", [])],
            foreign_keys=[
                ForeignKeyConstraint.from_dict(fk)
                for fk in data.get("foreign_keys", [])
            ],
            primary_key_columns=data.get("primary_key_columns", []),
        )


@dataclass
class SchemaRegistry:
    """Registry of all tables and their schemas.

    Tracks current schema state and supports evolution via ALTER TABLE.

    Attributes:
        tables: Dictionary mapping table name to TableSchema
    """

    tables: Dict[str, TableSchema] = field(default_factory=dict)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def empty(cls) -> "SchemaRegistry":
        """Create an empty SchemaRegistry.

        Returns:
            New empty SchemaRegistry
        """
        return cls(tables={})

    @classmethod
    def from_sql_list(cls, statements: List[str]) -> "SchemaRegistry":
        """Parse CREATE TABLE statements into a SchemaRegistry.

        Args:
            statements: List of CREATE TABLE SQL statements

        Returns:
            SchemaRegistry populated from SQL
        """
        registry = cls.empty()
        for stmt in statements:
            table = cls._parse_create_table(stmt)
            if table:
                registry.add_table(table)
        return registry

    @staticmethod
    def _parse_create_table(sql: str) -> Optional[TableSchema]:
        """Parse a CREATE TABLE statement into a TableSchema.

        Args:
            sql: CREATE TABLE SQL statement

        Returns:
            TableSchema if parsing successful, None otherwise
        """
        # Extract table name
        table_match = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\(", sql, re.IGNORECASE)
        if not table_match:
            return None

        table_name = table_match.group(1).lower()
        table = TableSchema(name=table_name)

        # Extract content between parentheses
        content_match = re.search(r"\((.*)\)", sql, re.DOTALL)
        if not content_match:
            return table

        content = content_match.group(1)

        # Split by comma, but not within parentheses
        parts: List[str] = []
        current_part = ""
        paren_depth = 0
        for char in content:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        if current_part.strip():
            parts.append(current_part.strip())

        # Parse each part
        for part in parts:
            part_upper = part.upper()

            # Skip constraints (FOREIGN KEY, PRIMARY KEY as standalone)
            if part_upper.startswith("FOREIGN KEY"):
                # Parse foreign key constraint
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)",
                    part,
                    re.IGNORECASE,
                )
                if fk_match:
                    table.add_foreign_key(
                        ForeignKeyConstraint(
                            column_name=fk_match.group(1).lower(),
                            references_table=fk_match.group(2).lower(),
                            references_column=fk_match.group(3).lower(),
                        )
                    )
                continue

            if part_upper.startswith("PRIMARY KEY"):
                # Parse composite primary key
                pk_match = re.search(
                    r"PRIMARY\s+KEY\s*\(([^)]+)\)", part, re.IGNORECASE
                )
                if pk_match:
                    pk_cols = [
                        col.strip().lower() for col in pk_match.group(1).split(",")
                    ]
                    table.primary_key_columns = pk_cols
                continue

            # Skip standalone UNIQUE constraints (we use inline UNIQUE in column defs)
            # Handles: UNIQUE (col), CONSTRAINT name UNIQUE (col), or malformed variants
            if part_upper.startswith("UNIQUE") or part_upper.startswith("CONSTRAINT"):
                # Extract column name from UNIQUE (col_name) pattern if present
                unique_match = re.search(r"UNIQUE\s*\((\w+)\)", part, re.IGNORECASE)
                if unique_match:
                    unique_col = unique_match.group(1).lower()
                    # Mark the column as unique if it exists
                    for col in table.columns:
                        if col.name == unique_col:
                            col.is_unique = True
                            break
                continue

            # Parse column definition
            col_parts = part.split()
            if len(col_parts) >= 2:
                col_name = col_parts[0].lower()
                col_type = col_parts[1].upper()
                is_pk = "PRIMARY KEY" in part_upper
                is_unique = "UNIQUE" in part_upper and not is_pk
                is_nullable = "NOT NULL" not in part_upper

                column = ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    is_primary_key=is_pk,
                    is_unique=is_unique,
                    is_nullable=is_nullable,
                )
                table.add_column(column)

        return table

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_table(self, name: str) -> Optional[TableSchema]:
        """Get a table by name.

        Args:
            name: Table name

        Returns:
            TableSchema if found, None otherwise
        """
        return self.tables.get(name.lower())

    def get_table_names(self) -> List[str]:
        """Get all table names.

        Returns:
            List of table names
        """
        return list(self.tables.keys())

    def has_table(self, name: str) -> bool:
        """Check if a table exists.

        Args:
            name: Table name

        Returns:
            True if table exists
        """
        return name.lower() in self.tables

    def get_all_columns(self, table_name: str) -> Set[str]:
        """Get all column names for a table.

        Args:
            table_name: Table name

        Returns:
            Set of column names, or empty set if table not found
        """
        table = self.get_table(table_name)
        if table:
            return table.get_column_names()
        return set()

    def find_new_attributes(
        self, entity_type: str, extracted_attrs: Set[str]
    ) -> Set[str]:
        """Find attributes in extraction that aren't in schema yet.

        Args:
            entity_type: Entity type (maps to table name)
            extracted_attrs: Set of attribute names from extraction

        Returns:
            Set of new attribute names not in schema
        """
        existing = self.get_all_columns(entity_type)
        # Filter out id columns and other system columns
        new_attrs = extracted_attrs - existing - {"id", f"{entity_type}_id"}
        return new_attrs

    # =========================================================================
    # Modification Methods
    # =========================================================================

    def add_table(self, table: TableSchema) -> None:
        """Add a table to the registry.

        Args:
            table: TableSchema to add
        """
        self.tables[table.name.lower()] = table
        col_names = [c.name for c in table.columns]
        logger.debug(
            f"SchemaRegistry: Added table '{table.name}' with columns: {col_names}"
        )

    def add_column_to_table(self, table_name: str, column: ColumnInfo) -> Optional[str]:
        """Add a column to an existing table.

        Args:
            table_name: Table to modify
            column: Column to add

        Returns:
            ALTER TABLE SQL statement, or None if table not found
        """
        table = self.get_table(table_name)
        if table:
            table.add_column(column)
            alter_sql = table.to_alter_add_column_sql(column)
            logger.info(
                f"SchemaRegistry: Added column '{column.name}' ({column.data_type}) to table '{table_name}'"
            )
            return alter_sql
        logger.warning(
            f"SchemaRegistry: Cannot add column '{column.name}' - table '{table_name}' not found"
        )
        return None

    def ensure_natural_keys_unique(self) -> int:
        """Ensure natural key columns have UNIQUE constraint for deduplication.

        This is a post-processing step that marks natural key columns as unique
        for non-junction tables that don't already have a unique column defined.
        This enables proper ON CONFLICT handling in UPSERT operations.

        Returns:
            Number of tables that had their natural key marked as unique
        """
        updated_count: int = 0

        for table_name, table in self.tables.items():
            # Skip junction tables (they use composite PK for conflict resolution)
            if table.is_junction_table():
                logger.debug(
                    f"  Skipping junction table '{table_name}' (uses composite PK)"
                )
                continue

            # Check if table already has a unique column (non-PK)
            has_unique: bool = any(
                col.is_unique and not col.is_primary_key for col in table.columns
            )
            if has_unique:
                logger.debug(f"  Table '{table_name}' already has UNIQUE constraint")
                continue

            # Detect and mark natural key as unique
            natural_key: Optional[str] = table._detect_natural_key()
            if natural_key:
                col = table.get_column(natural_key)
                if col:
                    col.is_unique = True
                    updated_count += 1
                    logger.info(
                        f"  Marked '{table_name}.{natural_key}' as UNIQUE (natural key)"
                    )
            else:
                logger.debug(f"  Table '{table_name}' has no detectable natural key")

        if updated_count > 0:
            logger.info(
                f"SchemaRegistry: Marked {updated_count} natural key columns as UNIQUE"
            )

        return updated_count

    def enforce_single_unique_constraint(self) -> int:
        """Ensure each non-junction table has at most one UNIQUE constraint.

        PostgreSQL's ON CONFLICT clause can only target one constraint per INSERT.
        Having multiple UNIQUE columns causes upserts to fail when conflicts occur
        on non-targeted unique columns. This method removes extra UNIQUE constraints,
        keeping only the natural key column (first non-PK TEXT column).

        Returns:
            Number of tables that had extra UNIQUE constraints removed
        """
        fixed_count: int = 0

        for table_name, table in self.tables.items():
            # Skip junction tables (they use composite PK)
            if table.is_junction_table():
                continue

            # Find all UNIQUE columns (non-PK)
            unique_cols: List[ColumnInfo] = [
                col for col in table.columns if col.is_unique and not col.is_primary_key
            ]

            if len(unique_cols) <= 1:
                continue  # Already has 0 or 1 UNIQUE, no fix needed

            # Determine which column should keep UNIQUE (natural key)
            natural_key: Optional[str] = table._detect_natural_key()

            # Remove UNIQUE from all except the natural key
            removed_cols: List[str] = []
            for col in unique_cols:
                if col.name != natural_key:
                    col.is_unique = False
                    removed_cols.append(col.name)

            if removed_cols:
                fixed_count += 1
                logger.info(
                    f"  Table '{table_name}': removed UNIQUE from {removed_cols}, "
                    f"kept UNIQUE on '{natural_key}'"
                )

        if fixed_count > 0:
            logger.info(
                f"SchemaRegistry: Fixed {fixed_count} tables with multiple UNIQUE constraints"
            )

        return fixed_count

    def standardize_name_columns(self) -> int:
        """Rename various name columns to 'name' for consistency.

        Standardizes schemas to use 'name' as the natural key column:
        - '{entity_type}_name' → 'name' (e.g., 'genre_name' → 'name')
        - 'full_name' → 'name' (for person table)
        Exception: 'work' tables keep 'title' as the natural key.

        Returns:
            Number of columns renamed
        """
        renamed_count: int = 0

        for table_name, table in self.tables.items():
            if table.is_junction_table():
                continue

            # Skip 'work' table (uses 'title')
            if table_name == "work":
                continue

            # Already has 'name' column - skip
            if table.has_column("name"):
                continue

            # Try various name column patterns to rename
            name_candidates = [
                f"{table_name}_name",  # genre_name, location_name, etc.
                "full_name",  # person.full_name
            ]

            for candidate in name_candidates:
                old_col = table.get_column(candidate)
                if old_col:
                    # Rename column → name
                    old_col.name = "name"
                    renamed_count += 1
                    logger.info(f"  Renamed '{table_name}.{candidate}' → 'name'")
                    break  # Only rename one column per table

        if renamed_count > 0:
            logger.info(f"SchemaRegistry: Renamed {renamed_count} columns to 'name'")

        return renamed_count

    # =========================================================================
    # Schema Enrichment
    # =========================================================================

    def enrich_from_entities(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        infer_types: bool = True,
    ) -> Tuple[int, List[str]]:
        """Enrich schema with columns discovered from extracted entities.

        Scans all extracted entities to find attribute keys that don't have
        corresponding columns in the schema, and adds them.

        Args:
            entities: Dict mapping entity_type -> List of entity dicts
            infer_types: Whether to infer column types from values

        Returns:
            Tuple of (columns_added_count, list of ALTER TABLE statements)
        """
        columns_added = 0
        alter_statements: List[str] = []

        # Columns to skip (handled differently or redundant)
        skip_columns = {
            "name",
            "title",
            "id",  # Natural keys
        }

        for entity_type, entity_list in entities.items():
            table = self.get_table(entity_type.lower())
            if not table:
                logger.debug(
                    f"Schema enrichment: table '{entity_type}' not found, skipping"
                )
                continue

            # Collect all unique keys across all entities of this type
            existing_columns = {col.name.lower() for col in table.columns}
            discovered_keys: Dict[str, Any] = {}  # key -> sample_value
            entity_name_col = f"{entity_type.lower()}_name"

            for entity in entity_list:
                if not isinstance(entity, dict):
                    continue
                for key, value in entity.items():
                    key_lower = key.lower()
                    if key_lower == "full_name" and (
                        "name" in existing_columns or "name" in discovered_keys
                    ):
                        continue
                    if key_lower == entity_name_col and (
                        "name" in existing_columns or "name" in discovered_keys
                    ):
                        continue
                    if key_lower == "name" and entity_type.lower() == "work" and (
                        "title" in existing_columns or "title" in discovered_keys
                    ):
                        continue
                    if (
                        key_lower not in existing_columns
                        and key_lower not in skip_columns
                    ):
                        if key_lower not in discovered_keys:
                            discovered_keys[key_lower] = value

            # Add missing columns
            for col_name, sample_value in discovered_keys.items():
                # Skip if it looks like a FK reference we shouldn't add as raw column
                if col_name.endswith("_id") and isinstance(sample_value, int):
                    continue

                # Infer data type from sample value
                data_type = (
                    self._infer_type_from_value(col_name, sample_value)
                    if infer_types
                    else "TEXT"
                )

                # Create and add column
                new_col = ColumnInfo(name=col_name, data_type=data_type)
                alter_sql = self.add_column_to_table(entity_type.lower(), new_col)
                if alter_sql:
                    alter_statements.append(alter_sql)
                    columns_added += 1

        if columns_added > 0:
            logger.info(
                f"SchemaRegistry: Enriched schema with {columns_added} new columns "
                f"discovered from extracted entities"
            )

        return columns_added, alter_statements

    def _infer_type_from_value(self, column_name: str, value: Any) -> str:
        """Infer PostgreSQL type from column name and sample value.

        Args:
            column_name: Name of the column
            value: Sample value from extracted entity

        Returns:
            PostgreSQL data type string
        """
        col = column_name.lower()

        if any(
            token in col
            for token in (
                "_note",
                "_notes",
                "_description",
                "_comment",
                "_comments",
                "_summary",
            )
        ):
            return "TEXT"
        if "precision" in col or "approx" in col:
            return "TEXT"

        # Pattern-based inference first
        if any(p in col for p in ("_date", "date_", "birth_date", "death_date")):
            return "DATE"
        if col.endswith("_year") or col in ("year", "birth_year", "death_year"):
            return "INTEGER"
        if col.startswith("is_") or col.startswith("has_"):
            return "BOOLEAN"

        # Value-based inference
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "FLOAT"

        return "TEXT"

    # =========================================================================
    # SQL Generation Methods
    # =========================================================================

    def to_sql_list(self) -> List[str]:
        """Generate all CREATE TABLE and ALTER TABLE statements.

        Handles FK dependencies by:
        1. Creating tables with only self-referencing FKs inline
        2. Deferring all inter-table FK constraints to ALTER TABLE statements
        3. Skipping FKs that reference non-existent tables

        This approach is robust against circular dependencies and forward references.

        Returns:
            List of SQL statements (CREATE TABLE + ALTER TABLE for deferred FKs)
        """
        # Sort tables (order doesn't matter much since FKs are deferred)
        sorted_tables = self._topological_sort()
        table_names = [t.name for t in sorted_tables]
        logger.debug(
            f"SchemaRegistry: Generating SQL for {len(self.tables)} tables "
            f"(order: {table_names})"
        )

        # Collect all FK references for each table
        all_fk_refs: Dict[str, Set[str]] = {}
        all_table_names = {t.name.lower() for t in sorted_tables}

        for table in sorted_tables:
            refs = set()
            for fk in table.foreign_keys:
                ref_table = fk.references_table.lower()
                # Only defer FKs to OTHER tables that exist in the schema
                # (self-references are OK inline, non-existent tables are skipped)
                if ref_table != table.name.lower() and ref_table in all_table_names:
                    refs.add(ref_table)
            if refs:
                all_fk_refs[table.name.lower()] = refs

        if all_fk_refs:
            logger.info(
                f"SchemaRegistry: Deferring {sum(len(v) for v in all_fk_refs.values())} "
                f"FK constraints from {len(all_fk_refs)} tables to ALTER TABLE statements"
            )

        # Generate CREATE TABLE statements (without inter-table FKs)
        statements: List[str] = []
        for table in sorted_tables:
            exclude_fks = all_fk_refs.get(table.name.lower(), set())
            statements.append(
                table.to_create_sql(
                    exclude_fks=exclude_fks,
                    valid_tables=all_table_names,
                )
            )

        # Add ALTER TABLE statements for all deferred FK constraints
        for table in sorted_tables:
            deferred_refs = all_fk_refs.get(table.name.lower(), set())
            if deferred_refs:
                alter_statements = table.get_deferred_fk_statements(deferred_refs)
                statements.extend(alter_statements)

        return statements

    def to_sql_with_relationships(self) -> str:
        """Generate SQL schema with relationship documentation.

        This method produces a schema string that includes:
        1. Comments explaining junction tables and M:N relationships
        2. All CREATE TABLE statements

        Returns:
            Complete SQL schema with relationship documentation
        """
        lines = []

        # Identify junction tables (tables with 2+ FKs and a composite name)
        junction_tables = self._identify_junction_tables()

        if junction_tables:
            lines.append("-- =========================================")
            lines.append("-- JUNCTION TABLES (Many-to-Many Relationships)")
            lines.append("-- =========================================")
            lines.append("-- These tables link two entity tables together.")
            lines.append("-- To find related entities, query the junction table")
            lines.append("-- using one entity's ID to find the other entity's IDs.")
            lines.append("--")

            for jt_name, (entity1, entity2) in junction_tables.items():
                lines.append(f"-- {jt_name}: links {entity1} <-> {entity2}")

            lines.append("--")
            lines.append("")

        # Add all CREATE TABLE statements
        for sql in self.to_sql_list():
            # Add a comment marker for junction tables
            table_name = self._extract_table_name(sql)
            if table_name and table_name.lower() in junction_tables:
                lines.append(f"-- Junction table: {table_name}")
            lines.append(sql)
            lines.append("")

        return "\n".join(lines)

    def _identify_junction_tables(self) -> Dict[str, Tuple[str, str]]:
        """Identify junction tables in the schema.

        Junction tables are identified by:
        1. Having 2+ foreign key constraints
        2. Having a composite name with underscore (e.g., person_work)

        Returns:
            Dict mapping junction table name to (entity1, entity2) tuple
        """
        junction_tables: Dict[str, Tuple[str, str]] = {}

        for name, table in self.tables.items():
            # Check if it looks like a junction table
            if "_" in name and len(table.foreign_keys) >= 2:
                # Extract the two entity types from foreign keys
                referenced_tables = [
                    fk.references_table for fk in table.foreign_keys[:2]
                ]
                if len(referenced_tables) >= 2:
                    junction_tables[name] = (
                        referenced_tables[0],
                        referenced_tables[1],
                    )

        return junction_tables

    def _extract_table_name(self, create_sql: str) -> Optional[str]:
        """Extract table name from CREATE TABLE statement.

        Args:
            create_sql: CREATE TABLE SQL statement

        Returns:
            Table name or None if not found
        """
        match = re.search(r"CREATE TABLE\s+(\w+)", create_sql, re.IGNORECASE)
        return match.group(1) if match else None

    def _topological_sort(self) -> List[TableSchema]:
        """Sort tables by foreign key dependencies.

        Returns:
            List of TableSchema in dependency order
        """
        # Build dependency graph
        deps: Dict[str, Set[str]] = {}
        for name, table in self.tables.items():
            deps[name] = set()
            for fk in table.foreign_keys:
                if fk.references_table.lower() in self.tables:
                    deps[name].add(fk.references_table.lower())

        # Topological sort
        result: List[TableSchema] = []
        visited: Set[str] = set()
        temp_visited: Set[str] = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                # Cycle detected, just continue
                return
            if name in visited:
                return
            temp_visited.add(name)
            for dep in deps.get(name, set()):
                visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            if name in self.tables:
                result.append(self.tables[name])

        for name in self.tables:
            visit(name)

        return result

    # =========================================================================
    # String Representation
    # =========================================================================

    def __str__(self) -> str:
        """Return a human-readable summary."""
        lines = ["SchemaRegistry:"]
        for name, table in self.tables.items():
            cols = ", ".join(c.name for c in table.columns)
            lines.append(f"  {name}: [{cols}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return f"SchemaRegistry(tables={list(self.tables.keys())})"

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for persistence.

        Returns:
            Dictionary with all schema data including FK relationships
        """
        return {
            "tables": {name: table.to_dict() for name, table in self.tables.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaRegistry":
        """Create SchemaRegistry from dictionary.

        Args:
            data: Dictionary with schema data

        Returns:
            New SchemaRegistry instance
        """
        registry = cls.empty()
        tables_data = data.get("tables", {})
        for name, table_data in tables_data.items():
            table = TableSchema.from_dict(table_data)
            registry.tables[name] = table
        return registry
