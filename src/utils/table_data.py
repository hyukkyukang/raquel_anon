"""Shared utilities for table data retrieval and formatting.

This module provides common functions for extracting data from PostgreSQL tables,
parsing schema statements, and formatting data for prompts. These utilities are
used across multiple components including query synthesis and upsert generation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("src.utils.table_data")


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier for simple table/column helper queries."""
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def extract_table_names_from_schema(schema: List[str]) -> List[str]:
    """Extract table names from a list of CREATE TABLE statements.

    Args:
        schema: List of SQL CREATE TABLE statements

    Returns:
        List of table names extracted from the schema
    """
    table_names: List[str] = []

    for schema_statement in schema:
        if "CREATE TABLE" not in schema_statement.upper():
            continue

        lines = schema_statement.strip().split("\n")
        for line in lines:
            if not line.strip().upper().startswith("CREATE TABLE"):
                continue

            parts = line.strip().split()
            # Parse "CREATE TABLE table_name" or "CREATE TABLE IF NOT EXISTS table_name"
            if (
                "IF" in line.upper()
                and "NOT" in line.upper()
                and "EXISTS" in line.upper()
            ):
                # CREATE TABLE IF NOT EXISTS table_name
                table_name = parts[5]
            else:
                # CREATE TABLE table_name
                table_name = parts[2]

            # Remove any schema prefix and parentheses
            table_name = table_name.split(".")[-1].rstrip("(").strip('"').strip("'")
            table_names.append(table_name)
            break

    return table_names


def extract_table_names_from_schema_str(schema: str) -> List[str]:
    """Extract table names from a schema string.

    Args:
        schema: SQL schema as a single string

    Returns:
        List of table names extracted from the schema
    """
    table_names: List[str] = []
    schema_lines = schema.strip().split("\n")

    for line in schema_lines:
        if not line.strip().upper().startswith("CREATE TABLE"):
            continue

        parts = line.strip().split()
        # Parse "CREATE TABLE table_name" or "CREATE TABLE IF NOT EXISTS table_name"
        if "IF" in line.upper() and "NOT" in line.upper() and "EXISTS" in line.upper():
            # CREATE TABLE IF NOT EXISTS table_name
            table_name = parts[5]
        else:
            # CREATE TABLE table_name
            table_name = parts[2]

        # Remove any schema prefix and parentheses
        table_name = table_name.split(".")[-1].rstrip("(").strip('"').strip("'")
        table_names.append(table_name)

    return table_names


def extract_columns_from_schema(schema: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract table -> column mapping from CREATE TABLE statements.

    Parses SQL schema to extract column names and types for each table.

    Args:
        schema: SQL schema as a single string with CREATE TABLE statements

    Returns:
        Dictionary mapping table names to list of (column_name, column_type) tuples
    """
    result: Dict[str, List[Tuple[str, str]]] = {}

    # Split schema into individual CREATE TABLE statements
    # Match CREATE TABLE ... ; patterns (handles quoted table names like "group")
    create_table_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?(\w+)["\']?\s*\((.*?)\);',
        re.IGNORECASE | re.DOTALL,
    )

    for match in create_table_pattern.finditer(schema):
        table_name = match.group(1).strip('"').strip("'")
        columns_section = match.group(2)

        columns: List[Tuple[str, str]] = []

        # Split by comma, but be careful about parentheses (for constraints)
        # Simple approach: split by comma and filter out constraints
        lines = columns_section.split(",")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip constraint definitions
            upper_line = line.upper()
            if any(
                kw in upper_line
                for kw in [
                    "PRIMARY KEY",
                    "FOREIGN KEY",
                    "UNIQUE",
                    "CHECK",
                    "CONSTRAINT",
                    "REFERENCES",
                ]
            ):
                # But allow "column_name TYPE PRIMARY KEY" format
                if not upper_line.startswith(
                    ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")
                ):
                    pass  # This is a column definition with inline constraint
                else:
                    continue

            # Parse column definition: column_name TYPE [constraints...]
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('"').strip("'")
                col_type = parts[1].upper()

                # Skip if this looks like a constraint keyword
                if col_name.upper() in (
                    "PRIMARY",
                    "FOREIGN",
                    "UNIQUE",
                    "CHECK",
                    "CONSTRAINT",
                ):
                    continue

                # Normalize common types
                if col_type in ("INT", "INT4", "INT8", "BIGINT", "SMALLINT"):
                    col_type = "INTEGER"
                elif col_type in ("VARCHAR", "CHARACTER", "CHAR"):
                    col_type = "TEXT"
                elif col_type == "BOOL":
                    col_type = "BOOLEAN"

                columns.append((col_name, col_type))

        if columns:
            result[table_name] = columns

    return result


def get_column_non_null_counts(
    pg_client: Any,
    table_name: str,
    columns: List[str],
) -> Dict[str, int]:
    """Query database to get non-null counts for each column.

    Args:
        pg_client: PostgreSQL client with a connection
        table_name: Name of the table to query
        columns: List of column names to check

    Returns:
        Dictionary mapping column names to their non-null counts
    """
    result: Dict[str, int] = {}

    if not columns:
        return result

    try:
        with pg_client.conn.cursor() as cursor:
            # Build query to count non-null values for each column
            count_expressions = [f'COUNT("{col}") AS "{col}"' for col in columns]
            query = f'SELECT {", ".join(count_expressions)} FROM "{table_name}"'

            cursor.execute(query)
            row = cursor.fetchone()

            if row:
                for i, col in enumerate(columns):
                    result[col] = row[i] if row[i] is not None else 0

    except Exception as e:
        logger.warning(f"Failed to get column stats for {table_name}: {e}")
        # Return zeros for all columns on failure
        result = {col: 0 for col in columns}

    return result


def prioritize_columns(
    columns: List[Tuple[str, str]],
    table_name: str,
    max_columns: int = 15,
    non_null_counts: Optional[Dict[str, int]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Prioritize columns by importance for LLM context.

    Returns columns grouped into: primary keys, foreign keys, and data columns.
    Data columns are prioritized by:
    1. Name/title columns
    2. Date columns
    3. Common descriptive columns
    4. Remaining columns by non-null count (if provided) or alphabetically

    Args:
        columns: List of (column_name, column_type) tuples
        table_name: Name of the table (used to identify primary key)
        max_columns: Maximum total columns to return
        non_null_counts: Optional dict of column -> non-null count for ranking

    Returns:
        Tuple of (primary_keys, foreign_keys, data_columns)
    """
    primary_keys: List[Tuple[str, str]] = []
    foreign_keys: List[Tuple[str, str]] = []
    data_columns: List[Tuple[str, str]] = []

    # Priority patterns for data columns
    name_patterns = {"name", "title", "label"}
    date_patterns = {"date", "year"}
    common_patterns = {"description", "type", "status", "category"}

    # Expected primary key name
    expected_pk = f"{table_name}_id"

    for col_name, col_type in columns:
        col_lower = col_name.lower()

        # Check if primary key (first column ending with _id matching table name, or 'id')
        if col_lower == expected_pk or (col_lower == "id" and not primary_keys):
            primary_keys.append((col_name, col_type))
        # Check if foreign key (ends with _id but not the primary key)
        elif col_lower.endswith("_id") and col_lower != expected_pk:
            foreign_keys.append((col_name, col_type))
        else:
            data_columns.append((col_name, col_type))

    # Sort data columns by priority
    def data_column_priority(col: Tuple[str, str]) -> Tuple[int, int, str]:
        col_name = col[0].lower()

        # Priority 1: Name/title columns
        if col_name in name_patterns or any(p in col_name for p in name_patterns):
            priority = 0
        # Priority 2: Date columns
        elif any(p in col_name for p in date_patterns):
            priority = 1
        # Priority 3: Common descriptive columns
        elif col_name in common_patterns:
            priority = 2
        # Priority 4: Other columns - sort by non-null count (descending)
        else:
            priority = 3

        # Secondary sort: by non-null count (descending, so negate)
        non_null = 0
        if non_null_counts and col[0] in non_null_counts:
            non_null = -non_null_counts[col[0]]  # Negative for descending

        return (priority, non_null, col_name)

    data_columns.sort(key=data_column_priority)

    # Calculate how many data columns we can include
    pk_fk_count = len(primary_keys) + len(foreign_keys)
    max_data_columns = max(0, max_columns - pk_fk_count)
    data_columns = data_columns[:max_data_columns]

    return primary_keys, foreign_keys, data_columns


def extract_valid_joins(
    columns_by_table: Dict[str, List[Tuple[str, str]]],
) -> Tuple[List[str], List[str]]:
    """Extract valid join relationships from schema.

    Identifies two types of joins:
    1. Direct FK joins: table.fk_id -> referenced_table.pk_id
    2. Junction table joins: table1 <-> junction_table <-> table2

    Args:
        columns_by_table: Dictionary mapping table names to (column_name, type) tuples

    Returns:
        Tuple of (direct_joins, junction_joins) where each is a list of join descriptions
    """
    direct_joins: List[str] = []
    junction_joins: List[str] = []

    all_tables = set(columns_by_table.keys())

    for table_name, columns in columns_by_table.items():
        col_names = [c[0] for c in columns]
        fk_cols = [
            c for c in col_names if c.endswith("_id") and c != f"{table_name}_id"
        ]

        # Check if this is a junction table (only has FK columns, typically 2)
        non_id_cols = [c for c in col_names if not c.endswith("_id")]
        is_junction = len(fk_cols) >= 2 and len(non_id_cols) == 0

        if is_junction and len(fk_cols) == 2:
            # Junction table - extract the two tables it connects
            table1 = fk_cols[0].rsplit("_id", 1)[0]
            table2 = fk_cols[1].rsplit("_id", 1)[0]
            if table1 in all_tables and table2 in all_tables:
                junction_joins.append(
                    f"{table1} <-> {table_name} <-> {table2} "
                    f"(JOIN {table_name} ON {table1}.{table1}_id = {table_name}.{fk_cols[0]} "
                    f"JOIN {table2} ON {table_name}.{fk_cols[1]} = {table2}.{table2}_id)"
                )
        else:
            # Regular table - check for direct FK relationships
            for fk_col in fk_cols:
                target_table = fk_col.rsplit("_id", 1)[0]
                if target_table in all_tables:
                    direct_joins.append(
                        f"{table_name}.{fk_col} -> {target_table}.{target_table}_id"
                    )

    return direct_joins, junction_joins


def format_schema_with_columns(
    schema: str,
    pg_client: Any = None,
    max_columns_per_table: int = 15,
) -> str:
    """Format schema with explicit column listing for each table.

    Creates a clear, readable format showing available columns per table
    to help prevent LLM hallucination of non-existent columns.

    Args:
        schema: SQL schema as a single string
        pg_client: Optional PostgreSQL client for getting column statistics
        max_columns_per_table: Maximum columns to show per table (default 15)

    Returns:
        Formatted string with explicit column listings
    """
    columns_by_table = extract_columns_from_schema(schema)

    if not columns_by_table:
        return schema  # Return original if parsing failed

    formatted_parts: List[str] = []
    formatted_parts.append("=== AVAILABLE TABLES AND COLUMNS ===")
    formatted_parts.append(
        "(Use ONLY these columns in your queries - do NOT invent columns)"
    )
    formatted_parts.append("")

    # Extract and show valid join paths first
    direct_joins, junction_joins = extract_valid_joins(columns_by_table)

    if direct_joins or junction_joins:
        formatted_parts.append("=== VALID JOIN RELATIONSHIPS ===")
        formatted_parts.append(
            "(Use ONLY these joins - tables can only be joined as shown below)"
        )
        formatted_parts.append("")

        if direct_joins:
            formatted_parts.append("Direct Foreign Key Joins:")
            for join in sorted(set(direct_joins))[:20]:  # Limit to 20 most important
                formatted_parts.append(f"  - {join}")
            formatted_parts.append("")

        if junction_joins:
            formatted_parts.append("Many-to-Many Joins (via junction tables):")
            for join in sorted(set(junction_joins)):
                formatted_parts.append(f"  - {join}")
            formatted_parts.append("")

        formatted_parts.append("=" * 50)
        formatted_parts.append("")

    for table_name, columns in columns_by_table.items():
        total_columns = len(columns)

        # Get non-null counts if pg_client is available
        non_null_counts = None
        if pg_client:
            col_names = [col[0] for col in columns]
            non_null_counts = get_column_non_null_counts(
                pg_client, table_name, col_names
            )

        # Prioritize and limit columns
        primary_keys, foreign_keys, data_columns = prioritize_columns(
            columns=columns,
            table_name=table_name,
            max_columns=max_columns_per_table,
            non_null_counts=non_null_counts,
        )

        shown_count = len(primary_keys) + len(foreign_keys) + len(data_columns)
        hidden_count = total_columns - shown_count

        # Format table header
        if hidden_count > 0:
            formatted_parts.append(
                f"Table: {table_name} ({total_columns} columns, showing {shown_count} most relevant)"
            )
        else:
            formatted_parts.append(f"Table: {table_name}")

        # Format primary keys
        if primary_keys:
            formatted_parts.append("  Primary Key:")
            for col_name, col_type in primary_keys:
                formatted_parts.append(f"    - {col_name} ({col_type})")

        # Format foreign keys with relationship hints
        if foreign_keys:
            formatted_parts.append("  Foreign Keys:")
            for col_name, col_type in foreign_keys:
                # Extract target table from FK name (e.g., genre_id -> genre)
                target_table = (
                    col_name.rsplit("_id", 1)[0] if col_name.endswith("_id") else "?"
                )
                formatted_parts.append(
                    f"    - {col_name} ({col_type}) -> {target_table}"
                )

        # Format data columns
        if data_columns:
            formatted_parts.append("  Data Columns:")
            for col_name, col_type in data_columns:
                formatted_parts.append(f"    - {col_name} ({col_type})")

        formatted_parts.append("")

    return "\n".join(formatted_parts)


def get_all_column_statistics(
    pg_client: Any,
    table_names: List[str],
) -> Dict[str, Dict[str, int]]:
    """Get non-null counts for all columns in all tables.

    Args:
        pg_client: PostgreSQL client with a connection
        table_names: List of table names to analyze

    Returns:
        Dictionary mapping table names to column non-null counts
    """
    result: Dict[str, Dict[str, int]] = {}

    for table_name in table_names:
        try:
            with pg_client.conn.cursor() as cursor:
                # Get column names first
                cursor.execute(f"SELECT * FROM {_quote_identifier(table_name)} LIMIT 0")
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )

                if columns:
                    # Get non-null counts
                    count_exprs = [f'COUNT("{col}") AS "{col}"' for col in columns]
                    cursor.execute(
                        f'SELECT {", ".join(count_exprs)} FROM "{table_name}"'
                    )
                    row = cursor.fetchone()
                    if row:
                        result[table_name] = {
                            col: (row[i] if row[i] is not None else 0)
                            for i, col in enumerate(columns)
                        }
        except Exception as e:
            logger.warning(f"Failed to get column stats for {table_name}: {e}")
            result[table_name] = {}

    return result


def get_columns_with_data(
    column_stats: Dict[str, Dict[str, int]],
    min_non_null: int = 1,
) -> Dict[str, List[str]]:
    """Get list of columns that have actual data (non-null values).

    Args:
        column_stats: Dictionary from get_all_column_statistics
        min_non_null: Minimum non-null count to consider column as having data

    Returns:
        Dictionary mapping table names to lists of columns with data
    """
    result: Dict[str, List[str]] = {}

    for table_name, col_counts in column_stats.items():
        cols_with_data = [
            col for col, count in col_counts.items() if count >= min_non_null
        ]
        result[table_name] = cols_with_data

    return result


def get_valid_join_pairs(
    pg_client: Any,
    table_names: List[str],
    max_pairs: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get actual matching foreign key values that can be used in JOINs.

    This helps the LLM understand which IDs actually exist across tables
    for successful JOIN queries.

    Args:
        pg_client: PostgreSQL client
        table_names: List of table names to analyze
        max_pairs: Maximum pairs to show per relationship

    Returns:
        Dictionary of join relationships with sample matching values
    """
    result: Dict[str, List[Dict[str, Any]]] = {}

    # Common junction table patterns
    junction_patterns = [
        ("person_work", "person", "person_id", "work", "work_id"),
        ("work_genre", "work", "work_id", "genre", "genre_id"),
        ("work_theme", "work", "work_id", "theme", "theme_id"),
        ("language_work", "language", "language_id", "work", "work_id"),
        ("person_award", "person", "person_id", "award", "award_id"),
        ("location_work", "location", "location_id", "work", "work_id"),
        ("genre_theme", "genre", "genre_id", "theme", "theme_id"),
        ("award_field", "award", "award_id", "field", "field_id"),
    ]

    # Direct foreign key patterns (table, fk_column, referenced_table, ref_pk)
    direct_fk_patterns = [
        ("person", "field_id", "field", "field_id"),
        ("person", "occupation_id", "occupation", "occupation_id"),
        ("person", "nationality_id", "nationality", "nationality_id"),
        ("work", "series_id", "series", "series_id"),
        ("award", "organization_id", "organization", "organization_id"),
        ("award", "work_id", "work", "work_id"),
    ]

    # Check junction tables
    for junction, t1, fk1, t2, fk2 in junction_patterns:
        if junction not in table_names:
            continue
        try:
            query = f"""
                SELECT DISTINCT j.{fk1}, j.{fk2}
                FROM "{junction}" j
                LIMIT {max_pairs}
            """
            rows = pg_client.execute_and_fetchall_with_col_names(query)
            if rows:
                key = f"{t1} <-> {t2} (via {junction})"
                result[key] = [{fk1: r[0], fk2: r[1]} for r in rows]
        except Exception:
            pass

    # Check direct foreign keys
    for table, fk_col, ref_table, ref_pk in direct_fk_patterns:
        if table not in table_names or ref_table not in table_names:
            continue
        try:
            query = f"""
                SELECT DISTINCT t.{fk_col}
                FROM "{table}" t
                WHERE t.{fk_col} IS NOT NULL
                LIMIT {max_pairs}
            """
            rows = pg_client.execute_and_fetchall_with_col_names(query)
            if rows:
                key = f"{table}.{fk_col} -> {ref_table}.{ref_pk}"
                result[key] = [r[0] for r in rows if r[0] is not None]
        except Exception:
            pass

    return result


def format_join_hints_for_prompt(
    join_pairs: Dict[str, List[Any]],
) -> str:
    """Format valid JOIN relationships as prompt hints.

    Args:
        join_pairs: Output from get_valid_join_pairs

    Returns:
        Formatted string for prompt
    """
    if not join_pairs:
        return ""

    parts: List[str] = []
    parts.append("=" * 60)
    parts.append("VALID JOIN RELATIONSHIPS (use these for JOIN queries)")
    parts.append("=" * 60)
    parts.append("These relationships have actual matching data:")
    parts.append("")

    for relationship, values in join_pairs.items():
        parts.append(f"  {relationship}")
        if isinstance(values, list) and values:
            if isinstance(values[0], dict):
                # Junction table with both IDs
                for pair in values[:3]:
                    pair_str = ", ".join(f"{k}={v}" for k, v in pair.items())
                    parts.append(f"    - {pair_str}")
            else:
                # Direct FK values
                vals_str = ", ".join(str(v) for v in values[:5])
                parts.append(f"    values: [{vals_str}]")
        parts.append("")

    parts.append("IMPORTANT: JOINs using other relationships may return empty results!")
    parts.append("")

    return "\n".join(parts)


def get_column_sample_values(
    pg_client: Any,
    table_names: List[str],
    max_distinct_values: int = 10,
    priority_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[Any]]]:
    """Get sample distinct values for columns to help LLM generate valid queries.

    For categorical columns (low cardinality), returns all distinct values.
    For numeric/date columns, returns min, max, and some sample values.

    Args:
        pg_client: PostgreSQL client with a connection
        table_names: List of table names to analyze
        max_distinct_values: Maximum distinct values to fetch per column
        priority_columns: Optional list of column name patterns to prioritize
            (e.g., ['name', 'title', 'type', 'country'])

    Returns:
        Dictionary mapping table_name -> column_name -> list of sample values
    """
    if priority_columns is None:
        priority_columns = [
            "name",
            "title",
            "type",
            "country",
            "genre",
            "theme",
            "category",
            "status",
            "year",
            "date",
            "location",
        ]

    result: Dict[str, Dict[str, List[Any]]] = {}

    for table_name in table_names:
        result[table_name] = {}
        try:
            with pg_client.conn.cursor() as cursor:
                # Get column info
                cursor.execute(f"SELECT * FROM {_quote_identifier(table_name)} LIMIT 0")
                if not cursor.description:
                    continue

                columns = [desc[0] for desc in cursor.description]

                # Prioritize columns that are likely categorical or useful for queries
                cols_to_check = []
                for col in columns:
                    col_lower = col.lower()
                    # Skip ID columns and very long text columns
                    if col_lower.endswith("_id") and col_lower != f"{table_name}_id":
                        continue
                    # Prioritize columns matching patterns
                    if any(p in col_lower for p in priority_columns):
                        cols_to_check.insert(0, col)
                    elif not col_lower.endswith("_id"):
                        cols_to_check.append(col)

                # Limit to top columns to avoid too many queries
                cols_to_check = cols_to_check[:8]

                for col in cols_to_check:
                    try:
                        # Get distinct values count first
                        cursor.execute(
                            f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}" '
                            f'WHERE "{col}" IS NOT NULL'
                        )
                        distinct_count = cursor.fetchone()[0]

                        if distinct_count == 0:
                            continue

                        if distinct_count <= max_distinct_values:
                            # Categorical - get all distinct values
                            cursor.execute(
                                f'SELECT DISTINCT "{col}" FROM "{table_name}" '
                                f'WHERE "{col}" IS NOT NULL '
                                f'ORDER BY "{col}" LIMIT {max_distinct_values}'
                            )
                            values = [row[0] for row in cursor.fetchall()]
                        else:
                            # High cardinality - get sample values
                            cursor.execute(
                                f'SELECT MIN("{col}"), MAX("{col}") FROM "{table_name}" '
                                f'WHERE "{col}" IS NOT NULL'
                            )
                            min_val, max_val = cursor.fetchone()

                            # Also get a few sample values
                            cursor.execute(
                                f'SELECT DISTINCT "{col}" FROM "{table_name}" '
                                f'WHERE "{col}" IS NOT NULL '
                                f'ORDER BY "{col}" LIMIT 5'
                            )
                            samples = [row[0] for row in cursor.fetchall()]

                            values = {
                                "min": min_val,
                                "max": max_val,
                                "samples": samples,
                                "total_distinct": distinct_count,
                            }

                        if values:
                            result[table_name][col] = values

                    except Exception as e:
                        logger.debug(
                            f"Error getting values for {table_name}.{col}: {e}"
                        )
                        continue

        except Exception as e:
            logger.warning(f"Failed to get sample values for {table_name}: {e}")

    return result


def estimate_column_statistics_from_rows(
    table_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """Estimate non-null counts directly from already-fetched sample rows."""
    result: Dict[str, Dict[str, int]] = {}

    for table_name, rows in table_data.items():
        stats: Dict[str, int] = {}
        columns = {column for row in rows for column in row.keys()}
        for column in columns:
            stats[column] = sum(1 for row in rows if row.get(column) is not None)
        result[table_name] = stats

    return result


def extract_sample_values_from_rows(
    table_data: Dict[str, List[Dict[str, Any]]],
    max_distinct_values: int = 10,
    priority_columns: Optional[List[str]] = None,
    max_columns_per_table: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Build prompt-oriented sample values from already fetched table samples."""
    if priority_columns is None:
        priority_columns = [
            "name",
            "title",
            "type",
            "country",
            "genre",
            "theme",
            "category",
            "status",
            "year",
            "date",
            "location",
        ]

    result: Dict[str, Dict[str, Any]] = {}

    for table_name, rows in table_data.items():
        if not rows:
            result[table_name] = {}
            continue

        columns = list(dict.fromkeys(col for row in rows for col in row.keys()))
        prioritized = sorted(
            columns,
            key=lambda col: (
                not any(pattern in col.lower() for pattern in priority_columns),
                col.lower().endswith("_id") and col.lower() != f"{table_name}_id",
                col.lower(),
            ),
        )[:max_columns_per_table]

        table_samples: Dict[str, Any] = {}
        for column in prioritized:
            non_null_values = [row.get(column) for row in rows if row.get(column) is not None]
            if not non_null_values:
                continue

            distinct_values = list(dict.fromkeys(non_null_values))
            if len(distinct_values) <= max_distinct_values:
                table_samples[column] = distinct_values
                continue

            sample_entry: Dict[str, Any] = {
                "samples": distinct_values[:5],
                "total_distinct": len(distinct_values),
            }
            try:
                sample_entry["min"] = min(distinct_values)
                sample_entry["max"] = max(distinct_values)
            except TypeError:
                pass
            table_samples[column] = sample_entry

        result[table_name] = table_samples

    return result


def format_sample_values_for_prompt(
    sample_values: Dict[str, Dict[str, Any]],
    max_tables: int = 15,
    column_stats: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Format sample values as a prompt section to help LLM generate valid queries.

    Args:
        sample_values: Output from get_column_sample_values
        max_tables: Maximum tables to include
        column_stats: Optional column statistics with non-null counts

    Returns:
        Formatted string for inclusion in prompt
    """
    if not sample_values:
        return ""

    parts: List[str] = []
    parts.append("=" * 60)
    parts.append("IMPORTANT: USE THESE EXACT VALUES FOR QUERIES THAT RETURN RESULTS")
    parts.append("=" * 60)
    parts.append("")
    parts.append("Copy these values directly into WHERE, LIKE, IN, BETWEEN clauses.")
    parts.append(
        "Queries using values NOT in this list will likely return empty results."
    )
    parts.append("")

    # Collect example WHERE clauses
    where_examples: List[str] = []

    for table_name, columns in list(sample_values.items())[:max_tables]:
        if not columns:
            continue

        # Get stats for this table if available
        table_stats = (column_stats or {}).get(table_name, {})

        parts.append(f"### {table_name.upper()} ###")
        for col_name, values in columns.items():
            # Get non-null count if available (column_stats[table][col] is an int)
            non_null = table_stats.get(col_name, "?")

            if isinstance(values, dict):
                # High cardinality column with min/max
                min_val = values.get("min")
                max_val = values.get("max")
                parts.append(
                    f"  {col_name} ({non_null} non-null): "
                    f"range {min_val} to {max_val}"
                )
                # Add BETWEEN example
                if min_val is not None and max_val is not None:
                    where_examples.append(
                        f"  {table_name}.{col_name} BETWEEN {min_val} AND {max_val}"
                    )
                if values.get("samples"):
                    samples = values["samples"][:5]
                    samples_str = ", ".join(repr(v) for v in samples)
                    parts.append(f"    USE: {samples_str}")
                    # Add equality example
                    if samples:
                        where_examples.append(
                            f"  {table_name}.{col_name} = {repr(samples[0])}"
                        )
            elif isinstance(values, list) and values:
                # Categorical column - show values ready to use
                if len(values) <= 8:
                    values_str = ", ".join(repr(v) for v in values)
                    parts.append(f"  {col_name} ({non_null} non-null): {values_str}")
                else:
                    values_str = ", ".join(repr(v) for v in values[:8])
                    parts.append(
                        f"  {col_name} ({non_null} non-null): {values_str} ..."
                    )

                # Add WHERE clause examples
                if values:
                    where_examples.append(
                        f"  {table_name}.{col_name} = {repr(values[0])}"
                    )
                    if len(values) >= 3:
                        in_vals = ", ".join(repr(v) for v in values[:3])
                        where_examples.append(
                            f"  {table_name}.{col_name} IN ({in_vals})"
                        )
                    # LIKE example for text
                    if isinstance(values[0], str) and len(values[0]) > 3:
                        like_val = values[0][:4]
                        where_examples.append(
                            f"  {table_name}.{col_name} ILIKE '%{like_val}%'"
                        )
        parts.append("")

    # Add WHERE clause examples section
    if where_examples:
        parts.append("=" * 60)
        parts.append("EXAMPLE WHERE CLAUSES (copy these patterns):")
        parts.append("=" * 60)
        for example in where_examples[:20]:  # Limit examples
            parts.append(example)
        parts.append("")

    return "\n".join(parts)


def format_sample_data_smart(
    table_data: Dict[str, List[Dict[str, Any]]],
    column_stats: Optional[Dict[str, Dict[str, int]]] = None,
    max_rows: int = 5,
    max_columns_display: int = 10,
) -> str:
    """Format sample data showing only non-null values.

    Instead of showing rows with 200+ NULL values, shows only
    columns that have actual data.

    Args:
        table_data: Dictionary mapping table names to their row data
        column_stats: Optional column statistics for prioritization
        max_rows: Maximum rows to show per table
        max_columns_display: Maximum columns to display per row

    Returns:
        Formatted string with smart sample data
    """
    if not table_data:
        return "(No table data available.)"

    formatted_parts: List[str] = []

    for table_name, rows in table_data.items():
        if not rows:
            formatted_parts.append(f"-- Table {table_name}: (empty)")
            continue

        # Determine which columns have data
        if column_stats and table_name in column_stats:
            # Prioritize columns by non-null count
            col_counts = column_stats[table_name]
            cols_with_data = sorted(
                [(col, count) for col, count in col_counts.items() if count > 0],
                key=lambda x: -x[1],  # Sort by count descending
            )
            priority_cols = [col for col, _ in cols_with_data[:max_columns_display]]
        else:
            # Fallback: use columns from first row that have values
            priority_cols = []
            for row in rows[:3]:
                for col, val in row.items():
                    if val is not None and col not in priority_cols:
                        priority_cols.append(col)
                        if len(priority_cols) >= max_columns_display:
                            break

        if not priority_cols:
            formatted_parts.append(f"-- Table {table_name}: (no non-null data found)")
            continue

        # Get total row count
        total_rows = len(rows)

        formatted_parts.append(f"-- Table {table_name} ({total_rows} rows):")
        formatted_parts.append(f"-- Columns with data: {', '.join(priority_cols)}")

        # Show sample rows with only non-null values
        sample_rows = rows[:max_rows]
        for i, row in enumerate(sample_rows, 1):
            # Filter to only show non-null values from priority columns
            filtered_row = {
                k: v for k, v in row.items() if k in priority_cols and v is not None
            }
            if filtered_row:
                # Truncate long string values
                display_row = {}
                for k, v in filtered_row.items():
                    str_v = str(v)
                    if len(str_v) > 40:
                        display_row[k] = str_v[:37] + "..."
                    else:
                        display_row[k] = v
                formatted_parts.append(f"--   Row {i}: {display_row}")
            else:
                formatted_parts.append(f"--   Row {i}: (all values null)")

        if total_rows > max_rows:
            formatted_parts.append(f"--   ... and {total_rows - max_rows} more rows")
        formatted_parts.append("")

    return "\n".join(formatted_parts)


def fetch_table_data(
    pg_client: Any,
    table_names: List[str],
    log_fetches: bool = False,
    max_rows_per_table: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all data or sampled rows from specified tables.

    Args:
        pg_client: PostgreSQL client with a connection
        table_names: List of table names to fetch data from
        log_fetches: Whether to log fetch operations
        max_rows_per_table: Optional row cap per table for prompt sampling

    Returns:
        Dictionary mapping table names to their row data
    """
    all_table_data: Dict[str, List[Dict[str, Any]]] = {}

    for table_name in table_names:
        try:
            with pg_client.conn.cursor() as cursor:
                query = f"SELECT * FROM {_quote_identifier(table_name)}"
                if max_rows_per_table is not None and max_rows_per_table > 0:
                    query += f" LIMIT {max_rows_per_table}"
                cursor.execute(query)
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                rows = cursor.fetchall()

                # Convert rows to list of dictionaries
                table_data: List[Dict[str, Any]] = [
                    dict(zip(columns, row)) for row in rows
                ]
                all_table_data[table_name] = table_data

                if log_fetches:
                    logger.info(
                        f"Fetched {len(table_data)} rows from table {table_name}"
                        + (
                            f" (sample cap={max_rows_per_table})"
                            if max_rows_per_table is not None and max_rows_per_table > 0
                            else ""
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch data from table {table_name}: {e}")
            all_table_data[table_name] = []

    return all_table_data


def get_all_table_data_from_schema(
    pg_client: Any,
    schema: List[str],
    log_fetches: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get all data from tables defined in the schema.

    This is a convenience function combining table name extraction and data fetching.

    Args:
        pg_client: PostgreSQL client with a connection
        schema: List of SQL CREATE TABLE statements
        log_fetches: Whether to log fetch operations

    Returns:
        Dictionary mapping table names to their row data
    """
    table_names = extract_table_names_from_schema(schema)
    return fetch_table_data(pg_client, table_names, log_fetches)


def get_all_table_data_from_schema_str(
    pg_client: Any,
    schema: str,
    log_fetches: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get all data from tables defined in the schema string.

    This is a convenience function combining table name extraction and data fetching.

    Args:
        pg_client: PostgreSQL client with a connection
        schema: SQL schema as a single string
        log_fetches: Whether to log fetch operations

    Returns:
        Dictionary mapping table names to their row data
    """
    table_names = extract_table_names_from_schema_str(schema)
    return fetch_table_data(pg_client, table_names, log_fetches)


def format_table_data_for_prompt(
    table_data: Dict[str, List[Dict[str, Any]]],
    max_rows_per_table: Optional[int] = None,
) -> str:
    """Format table data as a human-readable string for use in prompts.

    Args:
        table_data: Dictionary mapping table names to their row data
        max_rows_per_table: Optional limit on rows to include per table

    Returns:
        Formatted string representation of the database state
    """
    if not table_data:
        return "No data exists in the database yet."

    formatted_parts: List[str] = []

    for table_name, rows in table_data.items():
        if not rows:
            formatted_parts.append(f"Table '{table_name}': No data")
            continue

        formatted_parts.append(f"Table '{table_name}':")
        display_rows = rows[:max_rows_per_table] if max_rows_per_table else rows

        for i, row in enumerate(display_rows):
            row_str = ", ".join([f"{k}={repr(v)}" for k, v in row.items()])
            formatted_parts.append(f"  Row {i+1}: {row_str}")

        if max_rows_per_table and len(rows) > max_rows_per_table:
            formatted_parts.append(
                f"  ... and {len(rows) - max_rows_per_table} more rows"
            )

    return "\n".join(formatted_parts)
