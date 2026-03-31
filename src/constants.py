"""Shared constants for the RAQUEL project.

This module provides common constants used across the project including
SQL separators, file patterns, and configuration defaults.
"""

# SQL statement separators
SQL_STATEMENT_SEPARATOR: str = "\n\n"
"""Separator between individual SQL statements in files."""

SQL_QA_PAIR_SEPARATOR: str = "\n\n\n"
"""Separator between QA pair groups in files."""

# Query file separators (from config defaults)
DEFAULT_QUERY_SEPARATOR: str = "\n\n"
"""Default separator between queries in query files."""

# File patterns
SCHEMA_FILE_NAME: str = "schema.sql"
"""Default filename for schema SQL files."""

UPSERT_FILE_NAME: str = "upsert.sql"
"""Default filename for upsert SQL files."""

NULLIFY_FILE_NAME: str = "nullify.sql"
"""Default filename for nullify SQL files."""

# Intermediate result file patterns
INTERMEDIATE_RESULT_FILE_PREFIX: str = "result"
"""Default prefix for intermediate result files."""

ITEM_FILE_PREFIX_PATTERN: str = "{prefix}_{idx:05d}.json"
"""Pattern for indexed item filenames."""

# Logging constants
LOG_FORMAT: str = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
"""Standard logging format used across stages."""

LOG_DATE_FORMAT: str = "%m/%d %H:%M:%S"
"""Standard date format for logging."""

