"""
Table-based schema representation and management.

This module provides a more structured approach to schema management using
table-level abstractions instead of raw SQL strings.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import sqlparse
from sqlparse.sql import Statement, Parenthesis, Identifier
from sqlparse.tokens import Keyword


@dataclass
class ColumnDefinition:
    """Represents a single column definition."""
    name: str
    data_type: str
    constraints: List[str] = field(default_factory=list)
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_table: Optional[str] = None
    foreign_column: Optional[str] = None

    def to_sql(self) -> str:
        """Convert column definition to SQL string."""
        parts = [self.name, self.data_type]
        if self.constraints:
            parts.extend(self.constraints)
        return " ".join(parts)


@dataclass
class TableSchema:
    """Represents a complete table schema."""
    name: str
    columns: Dict[str, ColumnDefinition] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)  # (column, ref_table, ref_column)
    unique_constraints: List[List[str]] = field(default_factory=list)
    check_constraints: List[str] = field(default_factory=list)
    table_constraints: List[str] = field(default_factory=list)

    def add_column(self, column: ColumnDefinition) -> None:
        """Add a column to the table."""
        self.columns[column.name] = column
        if column.is_primary_key and column.name not in self.primary_keys:
            self.primary_keys.append(column.name)
        if column.is_foreign_key and column.foreign_table and column.foreign_column:
            fk_tuple = (column.name, column.foreign_table, column.foreign_column)
            if fk_tuple not in self.foreign_keys:
                self.foreign_keys.append(fk_tuple)

    def get_column_names(self) -> List[str]:
        """Get list of column names in order."""
        return list(self.columns.keys())

    def to_sql(self) -> str:
        """Convert table schema to CREATE TABLE SQL statement."""
        if not self.columns:
            return f"-- Empty table: {self.name}"

        lines = [f"CREATE TABLE {self.name} ("]

        # Add column definitions
        column_lines = []
        for column in self.columns.values():
            column_lines.append(f"    {column.to_sql()}")

        # Add table-level constraints
        constraints = []

        # Primary key constraint
        if self.primary_keys:
            if len(self.primary_keys) == 1:
                # Single column primary key - might already be in column definition
                pk_col = self.columns.get(self.primary_keys[0])
                if pk_col and not pk_col.is_primary_key:
                    constraints.append(f"    PRIMARY KEY ({self.primary_keys[0]})")
            else:
                # Composite primary key
                pk_cols = ", ".join(self.primary_keys)
                constraints.append(f"    PRIMARY KEY ({pk_cols})")

        # Foreign key constraints
        for col_name, ref_table, ref_column in self.foreign_keys:
            constraints.append(f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_column})")

        # Unique constraints
        for unique_cols in self.unique_constraints:
            if len(unique_cols) > 1:
                cols_str = ", ".join(unique_cols)
                constraints.append(f"    UNIQUE ({cols_str})")

        # Check constraints
        for check in self.check_constraints:
            constraints.append(f"    CHECK ({check})")

        # Custom table constraints
        constraints.extend(f"    {constraint}" for constraint in self.table_constraints)

        # Combine columns and constraints
        all_lines = column_lines + constraints
        lines.append(",\n".join(all_lines))
        lines.append(");")

        return "\n".join(lines)


class SchemaRegistry:
    """Manages a collection of table schemas."""

    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}

    def add_table(self, table: TableSchema) -> None:
        """Add or update a table schema."""
        self.tables[table.name] = table

    def get_table(self, name: str) -> Optional[TableSchema]:
        """Get a table schema by name."""
        return self.tables.get(name)

    def remove_table(self, name: str) -> bool:
        """Remove a table schema. Returns True if table existed."""
        return self.tables.pop(name, None) is not None

    def get_table_names(self) -> Set[str]:
        """Get all table names."""
        return set(self.tables.keys())

    def to_sql_list(self) -> List[str]:
        """Convert all tables to list of SQL CREATE TABLE statements."""
        return [table.to_sql() for table in self.tables.values()]

    def to_sql_string(self, separator: str = "\n\n") -> str:
        """Convert all tables to a single SQL string."""
        return separator.join(self.to_sql_list())

    @classmethod
    def from_sql_list(cls, sql_statements: List[str]) -> "SchemaRegistry":
        """Create SchemaRegistry from list of SQL CREATE TABLE statements."""
        registry = cls()

        for sql in sql_statements:
            if not sql.strip():
                continue

            # Parse the SQL statement
            parsed = sqlparse.parse(sql.strip())
            if not parsed:
                continue

            stmt = parsed[0]
            if stmt.get_type() != "CREATE":
                continue

            table = cls._parse_create_table_statement(stmt)
            if table:
                registry.add_table(table)

        return registry

    @staticmethod
    def _parse_create_table_statement(stmt: Statement) -> Optional[TableSchema]:
        """Parse a CREATE TABLE statement into a TableSchema."""
        # Extract table name
        table_name = None
        for tok in stmt.tokens:
            if tok.ttype is Keyword and tok.value.upper() == "TABLE":
                _, nxt = stmt.token_next(stmt.token_index(tok), skip_ws=True)
                if isinstance(nxt, Identifier):
                    table_name = nxt.get_real_name()
                else:
                    table_name = nxt.value
                break

        if not table_name:
            return None

        table = TableSchema(name=table_name)

        # Extract column definitions from parenthesis
        for tok in stmt.tokens:
            if isinstance(tok, Parenthesis):
                body = tok.value[1:-1]  # Remove outer parentheses
                SchemaRegistry._parse_table_body(body, table)
                break

        return table

    @staticmethod
    def _parse_table_body(body: str, table: TableSchema) -> None:
        """Parse the body of a CREATE TABLE statement (inside parentheses)."""
        # Split by commas, but respect nested parentheses
        definitions = SchemaRegistry._split_top_level(body)

        for defn in definitions:
            defn = defn.strip()
            if not defn:
                continue

            # Check if this is a table-level constraint
            upper_defn = defn.upper()
            if any(upper_defn.startswith(kw) for kw in ["PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "CONSTRAINT"]):
                SchemaRegistry._parse_table_constraint(defn, table)
            else:
                # This should be a column definition
                column = SchemaRegistry._parse_column_definition(defn)
                if column:
                    table.add_column(column)

    @staticmethod
    def _split_top_level(text: str) -> List[str]:
        """Split text by commas, respecting nested parentheses."""
        parts = []
        current = ""
        paren_count = 0

        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
                continue

            current += char

        if current.strip():
            parts.append(current.strip())

        return parts

    @staticmethod
    def _parse_column_definition(defn: str) -> Optional[ColumnDefinition]:
        """Parse a single column definition."""
        parts = defn.strip().split()
        if len(parts) < 2:
            return None

        name = parts[0].strip('"`')
        data_type = parts[1]

        column = ColumnDefinition(name=name, data_type=data_type)

        # Check for constraints in the remaining parts
        remaining = " ".join(parts[2:]).upper()

        if "PRIMARY KEY" in remaining:
            column.is_primary_key = True
            column.constraints.append("PRIMARY KEY")

        if "NOT NULL" in remaining:
            column.constraints.append("NOT NULL")

        if "UNIQUE" in remaining:
            column.constraints.append("UNIQUE")

        # Check for foreign key reference
        fk_match = re.search(r"REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)", remaining)
        if fk_match:
            column.is_foreign_key = True
            column.foreign_table = fk_match.group(1)
            column.foreign_column = fk_match.group(2)
            column.constraints.append(f"REFERENCES {column.foreign_table}({column.foreign_column})")

        return column

    @staticmethod
    def _parse_table_constraint(defn: str, table: TableSchema) -> None:
        """Parse a table-level constraint."""
        upper_defn = defn.upper()

        if upper_defn.startswith("PRIMARY KEY"):
            # Extract column names from PRIMARY KEY (col1, col2, ...)
            match = re.search(r"PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)", upper_defn)
            if match:
                cols = [col.strip().strip('"`') for col in match.group(1).split(',')]
                table.primary_keys.extend(col for col in cols if col not in table.primary_keys)

        elif upper_defn.startswith("FOREIGN KEY"):
            # Extract FOREIGN KEY (col) REFERENCES table(ref_col)
            match = re.search(r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s+REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)", upper_defn)
            if match:
                col_name = match.group(1)
                ref_table = match.group(2)
                ref_column = match.group(3)
                fk_tuple = (col_name, ref_table, ref_column)
                if fk_tuple not in table.foreign_keys:
                    table.foreign_keys.append(fk_tuple)

        elif upper_defn.startswith("UNIQUE"):
            # Extract UNIQUE (col1, col2, ...)
            match = re.search(r"UNIQUE\s*\(\s*([^)]+)\s*\)", upper_defn)
            if match:
                cols = [col.strip().strip('"`') for col in match.group(1).split(',')]
                table.unique_constraints.append(cols)

        elif upper_defn.startswith("CHECK"):
            # Extract CHECK (condition)
            match = re.search(r"CHECK\s*\(\s*(.+)\s*\)", defn, re.IGNORECASE)
            if match:
                table.check_constraints.append(match.group(1))

        else:
            # Generic table constraint
            table.table_constraints.append(defn)