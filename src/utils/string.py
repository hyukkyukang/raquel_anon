import re
import unicodedata


def normalize_string(text: str) -> str:
    """Normalize a string for consistent cache key generation.

    This method:
    1. Normalizes Unicode characters
    2. Normalizes whitespace
    3. Strips leading/trailing whitespace

    Args:
        text (str): The string to normalize

    Returns:
        str: Normalized string
    """
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Normalize whitespace (replace multiple spaces with single space)
    text = " ".join(text.split())
    # Strip leading/trailing whitespace
    return text.strip()


def sanitize_identifier(identifier: str) -> str:
    """Normalize identifiers for use in file paths and tracking namespaces."""
    sanitized = identifier.strip().replace("/", "_").replace(" ", "_")
    return sanitized


def sanitize_sql_identifier(identifier: str, *, default: str = "unnamed") -> str:
    """Normalize a value into a safe SQL identifier token.

    This keeps identifiers ASCII-ish and stable across the pipeline so entity
    types like ``lgbtq+_identity`` do not leak invalid characters into table,
    column, junction-table, or foreign-key names.
    """
    if not identifier:
        return default

    sanitized = unicodedata.normalize("NFKC", identifier).strip().lower()
    sanitized = re.sub(r"[\s\-/+]+", "_", sanitized)
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")

    if not sanitized:
        return default
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized
