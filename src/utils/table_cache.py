"""In-memory cache for table data with incremental updates.

This module provides a caching mechanism for database table data that avoids
expensive full table fetches for every QA pair during upsert operations.
"""

import logging
import re
from typing import Any, Dict, List, Set

from src.utils.table_data import extract_table_names_from_schema, fetch_table_data

logger = logging.getLogger("src.utils.table_cache")


class TableDataCache:
    """In-memory cache for table data with incremental updates.

    Instead of fetching all table data for every QA pair, this cache
    maintains the current state and only refreshes tables that were
    modified by the last upsert operation.

    Attributes:
        _pg_client: PostgreSQL client with a connection
        _cache: Dictionary mapping table names to their row data
        _initialized: Whether the cache has been initialized
        _table_names: List of table names in the schema
    """

    def __init__(self, pg_client: Any) -> None:
        """Initialize the cache.

        Args:
            pg_client: PostgreSQL client with a connection
        """
        self._pg_client = pg_client
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized: bool = False
        self._table_names: List[str] = []

    @property
    def initialized(self) -> bool:
        """Check if cache has been initialized.

        Returns:
            True if cache has been initialized, False otherwise
        """
        return self._initialized

    def initialize(self, schema: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform initial full fetch of all table data.

        This should be called once at the start of the upsert process
        to populate the cache with all existing table data.

        Args:
            schema: List of CREATE TABLE statements

        Returns:
            Dictionary mapping table names to their row data
        """
        self._table_names = extract_table_names_from_schema(schema)
        self._cache = fetch_table_data(self._pg_client, self._table_names)
        self._initialized = True
        logger.info(f"Cache initialized with {len(self._table_names)} tables")
        return self._cache

    def get_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get current cached data.

        Returns:
            Dictionary mapping table names to their row data

        Raises:
            RuntimeError: If cache has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        return self._cache

    def update_after_upsert(self, upsert_statements: List[str]) -> None:
        """Incrementally update cache after upsert execution.

        Only refetches tables that were affected by the upsert statements,
        avoiding expensive full database reads.

        Args:
            upsert_statements: List of executed INSERT/UPDATE statements
        """
        if not self._initialized:
            return

        affected_tables: Set[str] = self._extract_affected_tables(upsert_statements)

        if not affected_tables:
            return

        logger.debug(
            f"Refreshing {len(affected_tables)} affected tables: {affected_tables}"
        )

        for table in affected_tables:
            if table in self._table_names:
                refreshed: Dict[str, List[Dict[str, Any]]] = fetch_table_data(
                    self._pg_client, [table]
                )
                self._cache[table] = refreshed.get(table, [])

    def _extract_affected_tables(self, statements: List[str]) -> Set[str]:
        """Extract table names from INSERT/UPDATE statements.

        Parses SQL statements to identify which tables were modified,
        supporting both INSERT INTO and UPDATE syntax.

        Args:
            statements: List of SQL statements

        Returns:
            Set of affected table names (lowercase)
        """
        tables: Set[str] = set()

        for stmt in statements:
            # Match INSERT INTO table_name
            insert_match = re.search(r"INSERT\s+INTO\s+(\w+)", stmt, re.IGNORECASE)
            if insert_match:
                tables.add(insert_match.group(1).lower())

            # Match UPDATE table_name
            update_match = re.search(r"UPDATE\s+(\w+)", stmt, re.IGNORECASE)
            if update_match:
                tables.add(update_match.group(1).lower())

        return tables

    def clear(self) -> None:
        """Clear the cache and reset state.

        This should be called when starting a new upsert process
        or when the schema changes.
        """
        self._cache.clear()
        self._initialized = False
        self._table_names.clear()
        logger.debug("Cache cleared")

