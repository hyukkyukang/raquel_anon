import logging
import re
from typing import List

import hkkang_utils.sql as sql_utils
import sqlparse

from src.llm.exception import SQLParsingError

# Pre-compile all regex patterns for better performance
_SQL_CODE_BLOCK_PATTERN = re.compile(r"```sql\n(.*?)```", re.DOTALL)
_CODE_BLOCK_PATTERN = re.compile(r"```(.*?)```", re.DOTALL)
_SINGLE_LINE_COMMENT_PATTERN = re.compile(r"--.*")
_NUMBERED_PREFIX_PATTERN = re.compile(r"^\d+\.\s*", re.MULTILINE)
_LEADING_BRACKETS_PATTERN = re.compile(r"^[\]\[]+", re.MULTILINE)
_AUTOINCREMENT_PATTERN = re.compile(r"\bAUTOINCREMENT\b", re.IGNORECASE)
_INT_PATTERN = re.compile(r"\bINT\b", re.IGNORECASE)
_SERIAL_PATTERN = re.compile(r"INTEGER PRIMARY KEY AUTO_INCREMENT", re.IGNORECASE)
_COLUMN_DECLARATION_PATTERN = re.compile(
    r"\b([a-zA-Z_][\w']*)\s+(VARCHAR|TEXT|INTEGER|BOOLEAN|DATE|SERIAL)\b",
    re.IGNORECASE,
)
_APOSTROPHE_PATTERN = re.compile(r"\\'")

logger = logging.getLogger("Parse")


def post_process_sql(response: str) -> List[str]:
    """Parse and format SQL statements from the LLM response."""
    # Strip markdown code blocks if present
    sql_code = _SQL_CODE_BLOCK_PATTERN.sub(r"\1", response)
    sql_code = _CODE_BLOCK_PATTERN.sub(r"\1", sql_code)
    sql_code = sql_code.strip()
    processed_sql_code = normalize_sql(sql_code)

    # Normalize SQL formatting
    formatted_sql: str = sqlparse.format(
        processed_sql_code, reindent=True, keyword_case="upper"
    )

    # Parse the SQL code
    parsed_sql = sqlparse.parse(formatted_sql)
    if not parsed_sql:
        if formatted_sql:
            raise SQLParsingError(formatted_sql)
        else:
            # raise SQLParsingError(response)
            return []

    parsed_sql_raw = [str(stmt).strip() for stmt in parsed_sql]

    # Convert statements to strings with pretty formatting
    parsed_sql_str = [sql_utils.prettify_sql(sql) for sql in parsed_sql_raw]

    # Check the syntax of the SQL statements again
    for sql in parsed_sql_str:
        # Skip validation for statements with OVERRIDING SYSTEM VALUE
        # as this is valid PostgreSQL but may not be recognized by the validator
        if "OVERRIDING SYSTEM VALUE" in sql:
            continue
        try:
            is_valid = sql_utils.is_valid_sql(sql)
        except Exception as e:
            logger.warning(f"Error checking SQL syntax: {e}")
            is_valid = False
        if not is_valid:
            logger.warning(f"Invalid SQL syntax: {sql}")
            raise SQLParsingError(sql, "Invalid SQL syntax")

    return parsed_sql_str


def normalize_sql(sql: str) -> str:
    """Post-process SQL statements to normalize syntax and fix common issues."""
    # Remove LLM artifacts like "(blank LINE)" or "(blank line)" that some models output
    sql = re.sub(r"\(blank\s*line\)", "", sql, flags=re.IGNORECASE)

    # Remove single-line comments and strip empty lines
    sql = "\n".join(
        line
        for line in (
            _SINGLE_LINE_COMMENT_PATTERN.sub("", line) for line in sql.splitlines()
        )
        if line.strip()
    )

    # Remove numbered prefixes like '1.' at the start of lines
    sql = _NUMBERED_PREFIX_PATTERN.sub("", sql)

    # Remove leading bracket characters that LLM sometimes generates
    sql = _LEADING_BRACKETS_PATTERN.sub("", sql)

    # Normalize common syntax
    sql = _AUTOINCREMENT_PATTERN.sub("AUTO_INCREMENT", sql)
    sql = _INT_PATTERN.sub("INTEGER", sql)
    sql = _SERIAL_PATTERN.sub("SERIAL PRIMARY KEY", sql)

    # Fix invalid identifiers: replace apostrophes in column names
    def fix_identifier(match):
        column = match.group(1)
        fixed = re.sub(r"[^\w]", "_", column)  # Replace non-word chars with '_'
        return f"{fixed} {match.group(2)}"

    sql = _COLUMN_DECLARATION_PATTERN.sub(fix_identifier, sql)

    # Hacky fix for the case where LLM returns "]]" at the end of the SQL statements
    if sql.endswith("]]"):
        sql = sql[:-2]

    # Replace apostrophes with single quotes
    sql = _APOSTROPHE_PATTERN.sub("''", sql)

    # Fix OVERRIDING SYSTEM VALUE placement in INSERT statements
    # The issue is that OVERRIDING SYSTEM VALUE appears at the wrong position
    # It should be: INSERT INTO table (...) VALUES (...) OVERRIDING SYSTEM VALUE ON CONFLICT
    # But LLM generates: INSERT INTO table (...) VALUES (...) OVERRIDING SYSTEM VALUE ON CONFLICT
    sql = re.sub(
        r"(\bINSERT\s+INTO\s+\w+\s*\([^)]+\)\s*VALUES\s*\([^)]+\))\s+OVERRIDING\s+SYSTEM\s+VALUE\s+(ON\s+CONFLICT)",
        r"\1 OVERRIDING SYSTEM VALUE \2",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return sql
