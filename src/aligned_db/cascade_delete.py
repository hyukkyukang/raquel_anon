"""Cascade delete functionality for PostgreSQL databases.

This module provides utilities for performing cascade deletes that respect
foreign key relationships, including cleanup of rows with empty values.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("src.aligned_db.cascade_delete")


def _quote_identifier(name: str) -> str:
    """Quote an identifier used in dynamic cleanup queries."""
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


class CascadeDeleteHandler:
    """Handles cascade delete operations respecting foreign key relationships.

    This class provides methods for recursively deleting rows and their
    dependents, as well as cleaning up rows with empty/null values.

    Attributes:
        pg_client: PostgreSQL database client
    """

    def __init__(self, pg_client: Any) -> None:
        """Initialize the CascadeDeleteHandler.

        Args:
            pg_client: PostgreSQL client with a connection
        """
        self._pg_client = pg_client
        self._primary_key_cache: Dict[str, Optional[str]] = {}
        self._identifier_column_cache: Dict[str, Set[str]] = {}

    _TEXT_LIKE_TYPES = {
        "character",
        "character varying",
        "citext",
        "name",
        "text",
    }

    @property
    def pg_client(self) -> Any:
        """Get the PostgreSQL client."""
        return self._pg_client

    def _get_primary_key_column(self, table_name: str) -> Optional[str]:
        """Return the primary key column name for a table."""
        if table_name in self._primary_key_cache:
            return self._primary_key_cache[table_name]

        cursor = self._pg_client.conn.cursor()
        cursor.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = 'public'
              AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            """,
            (table_name,),
        )

        pk_columns = [row[0] for row in cursor.fetchall()]
        if not pk_columns:
            self._primary_key_cache[table_name] = None
            return None

        if len(pk_columns) > 1:
            logger.warning(
                "Table '%s' has a composite primary key; using '%s' for cascade cleanup.",
                table_name,
                pk_columns[0],
            )

        self._primary_key_cache[table_name] = pk_columns[0]
        return pk_columns[0]

    def _get_identifier_columns(self, table_name: str) -> Set[str]:
        """Return columns that act as identifiers (primary or foreign keys)."""
        if table_name in self._identifier_column_cache:
            return self._identifier_column_cache[table_name]

        cursor = self._pg_client.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.table_name = %s
              AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
            """,
            (table_name,),
        )
        identifier_columns = {row[0] for row in cursor.fetchall()}
        self._identifier_column_cache[table_name] = identifier_columns
        return identifier_columns

    def _empty_condition(self, column_name: str, data_type: str) -> str:
        """Return a type-aware SQL predicate for empty content columns."""
        quoted_column = _quote_identifier(column_name)
        if data_type in self._TEXT_LIKE_TYPES:
            return (
                f"({quoted_column} IS NULL OR BTRIM({quoted_column}) = '' "
                f"OR UPPER(BTRIM({quoted_column})) = 'NULL')"
            )
        return f"{quoted_column} IS NULL"

    def get_foreign_key_relationships(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Get foreign key relationships between tables.

        Returns:
            Dict mapping referenced_table -> List of (referencing_table,
            referencing_column, referenced_column) tuples
        """
        cursor = self._pg_client.conn.cursor()

        cursor.execute("""
            SELECT
                tc.table_name AS referencing_table,
                kcu.column_name AS referencing_column,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
        """)

        relationships: Dict[str, List[Tuple[str, str, str]]] = {}
        for row in cursor.fetchall():
            referencing_table, referencing_column, referenced_table, referenced_column = row

            if referenced_table not in relationships:
                relationships[referenced_table] = []

            relationships[referenced_table].append(
                (referencing_table, referencing_column, referenced_column)
            )

        return relationships

    def find_dependent_rows(
        self,
        table_name: str,
        id_column: str,
        id_value: Any,
        foreign_key_map: Dict[str, List[Tuple[str, str, str]]],
    ) -> List[Tuple[str, str, Any]]:
        """Find all rows in other tables that reference the given row.

        Args:
            table_name: Name of the table containing the referenced row
            id_column: Name of the ID column
            id_value: Value of the ID to look for dependencies
            foreign_key_map: Foreign key relationship mapping

        Returns:
            List of (table_name, id_column, id_value) tuples for dependent rows
        """
        dependent_rows: List[Tuple[str, str, Any]] = []

        if table_name not in foreign_key_map:
            return dependent_rows

        cursor = self._pg_client.conn.cursor()

        for ref_table, ref_column, referenced_column in foreign_key_map[table_name]:
            if referenced_column != id_column:
                continue

            ref_id_column = self._get_primary_key_column(ref_table)
            if ref_id_column is None:
                logger.warning(
                    "Skipping dependency lookup for table '%s' because no primary key was found.",
                    ref_table,
                )
                continue

            try:
                cursor.execute(
                    f"SELECT {_quote_identifier(ref_id_column)} "
                    f"FROM {_quote_identifier(ref_table)} "
                    f"WHERE {_quote_identifier(ref_column)} = %s",
                    (id_value,),
                )

                for (ref_id,) in cursor.fetchall():
                    dependent_rows.append((ref_table, ref_id_column, ref_id))

            except Exception as e:
                logger.warning(
                    f"Error finding dependencies for {table_name}.{id_column}={id_value}: {e}"
                )

        return dependent_rows

    def cascade_delete_row(
        self,
        table_name: str,
        id_column: str,
        id_value: Any,
        foreign_key_map: Dict[str, List[Tuple[str, str, str]]],
        visited: Optional[Set[Tuple[str, Any]]] = None,
    ) -> int:
        """Recursively delete a row and all its dependent rows.

        Args:
            table_name: Name of the table containing the row to delete
            id_column: Name of the ID column
            id_value: Value of the ID to delete
            foreign_key_map: Foreign key relationship mapping
            visited: Set of already visited (table, id) pairs to prevent cycles

        Returns:
            Number of rows deleted
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        key = (table_name, id_value)
        if key in visited:
            return 0
        visited.add(key)

        deleted_count = 0
        cursor = self._pg_client.conn.cursor()

        try:
            # First, recursively delete all dependent rows
            dependent_rows = self.find_dependent_rows(
                table_name, id_column, id_value, foreign_key_map
            )

            for dep_table, dep_id_column, dep_id_value in dependent_rows:
                deleted_count += self.cascade_delete_row(
                    dep_table, dep_id_column, dep_id_value, foreign_key_map, visited
                )

            # Then delete the original row
            cursor.execute(
                f"DELETE FROM {_quote_identifier(table_name)} "
                f"WHERE {_quote_identifier(id_column)} = %s",
                (id_value,),
            )
            if cursor.rowcount > 0:
                deleted_count += cursor.rowcount
                logger.debug(
                    f"Cascade deleted row from {table_name} where {id_column}={id_value}"
                )

        except Exception as e:
            logger.error(
                f"Error in cascade delete for {table_name}.{id_column}={id_value}: {e}"
            )

        return deleted_count

    def get_table_column_info(
        self, table_name: str
    ) -> Tuple[Optional[str], List[Tuple[str, str]]]:
        """Get the primary key and content columns for a table.

        Args:
            table_name: Name of the table

        Returns:
            Tuple of (id_column, list of non-ID (name, data_type) columns)
        """
        cursor = self._pg_client.conn.cursor()
        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            AND table_schema = 'public'
            ORDER BY ordinal_position
            """,
            (table_name,),
        )

        columns: List[Tuple[str, str]] = cursor.fetchall()
        if not columns:
            return None, []

        id_column = self._get_primary_key_column(table_name)
        identifier_columns = self._get_identifier_columns(table_name)
        non_id_columns = [
            (col_name, data_type)
            for col_name, data_type in columns
            if col_name not in identifier_columns
        ]

        if id_column is None:
            for col_name, data_type in columns:
                if col_name.endswith("_id") and data_type in ("integer", "bigint"):
                    logger.warning(
                        "Falling back to identifier-like column '%s' as the row key for table '%s'.",
                        col_name,
                        table_name,
                    )
                    id_column = col_name
                    break

        if id_column is not None:
            non_id_columns = [
                (col_name, data_type)
                for col_name, data_type in non_id_columns
                if col_name != id_column
            ]

        return id_column, non_id_columns

    def find_empty_rows(
        self, table_name: str
    ) -> List[Tuple[str, Any]]:
        """Find rows where all non-ID fields are null/empty.

        Args:
            table_name: Name of the table to check

        Returns:
            List of (id_column, id_value) tuples for empty rows
        """
        id_column, non_id_columns = self.get_table_column_info(table_name)

        # Skip tables that contain only id fields
        if not non_id_columns or not id_column:
            return []

        # Build condition for empty/null check
        non_id_conditions: List[str] = [
            self._empty_condition(col_name, data_type)
            for col_name, data_type in non_id_columns
        ]

        cursor = self._pg_client.conn.cursor()
        select_query = f"""
            SELECT {_quote_identifier(id_column)}
            FROM {_quote_identifier(table_name)}
            WHERE {' AND '.join(non_id_conditions)}
        """

        cursor.execute(select_query)
        return [(id_column, row[0]) for row in cursor.fetchall()]

    def cleanup_empty_rows(self) -> int:
        """Delete rows where all fields except ID are empty/null.

        Performs cascade deletion to remove related rows in other tables.

        Returns:
            Total number of rows deleted
        """
        logger.info(
            "Starting cleanup of rows with empty values (with cascade delete)"
        )

        # Get foreign key relationships
        foreign_key_map = self.get_foreign_key_relationships()
        logger.info(f"Found foreign key relationships for {len(foreign_key_map)} tables")

        # Get all table names
        cursor = self._pg_client.conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        tables: List[str] = [row[0] for row in cursor.fetchall()]

        # First pass: identify all empty rows
        rows_to_delete: List[Tuple[str, str, Any]] = []

        for table_name in tables:
            try:
                empty_rows = self.find_empty_rows(table_name)
                if empty_rows:
                    logger.info(
                        f"Found {len(empty_rows)} empty rows in table '{table_name}'"
                    )
                    for id_column, id_value in empty_rows:
                        rows_to_delete.append((table_name, id_column, id_value))
            except Exception as e:
                logger.error(f"Error identifying empty rows in table '{table_name}': {e}")

        # Second pass: cascade delete
        logger.info(f"Performing cascade delete for {len(rows_to_delete)} empty rows")
        total_deleted = 0

        for table_name, id_column, id_value in rows_to_delete:
            try:
                deleted_count = self.cascade_delete_row(
                    table_name, id_column, id_value, foreign_key_map
                )
                total_deleted += deleted_count

                if deleted_count > 0:
                    logger.debug(
                        f"Cascade deleted {deleted_count} rows starting from "
                        f"{table_name}.{id_column}={id_value}"
                    )
            except Exception as e:
                logger.error(
                    f"Error cascade deleting {table_name}.{id_column}={id_value}: {e}"
                )

        logger.info(f"Cleanup completed. Total rows deleted: {total_deleted}")
        return total_deleted
