"""JSON utility functions with LLM-based repair support.

This module provides utilities for parsing JSON with automatic repair
capabilities when parsing fails due to syntax errors, and extraction
of JSON from LLM responses that may contain markdown.
"""

import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Optional, Tuple

from omegaconf import DictConfig

from src.llm.json_repair import JSONRepairer

logger = logging.getLogger("JSONUtils")

# Global repairer instance (lazily initialized)
_global_repairer: Optional[JSONRepairer] = None


def get_json_repairer(api_cfg: DictConfig, global_cfg: DictConfig) -> JSONRepairer:
    """Get or create a global JSONRepairer instance.

    This function provides a singleton-like access to a JSONRepairer,
    creating it on first use with the provided configuration.

    Args:
        api_cfg: LLM API configuration
        global_cfg: Global configuration object

    Returns:
        JSONRepairer instance
    """
    global _global_repairer
    if _global_repairer is None:
        _global_repairer = JSONRepairer(api_cfg, global_cfg)
    return _global_repairer


def safe_json_loads(
    json_str: str,
    repairer: Optional[JSONRepairer] = None,
    repair_on_error: bool = True,
) -> Tuple[Any, bool]:
    """Parse JSON with optional LLM-based repair on failure.

    This function attempts to parse a JSON string using standard json.loads().
    If parsing fails and repair_on_error is True, it will attempt to use an
    LLM to fix the syntax errors and parse again.

    Args:
        json_str: The JSON string to parse
        repairer: Optional JSONRepairer instance. If None and repair_on_error
            is True, no repair will be attempted.
        repair_on_error: Whether to attempt LLM-based repair on parse failure.
            Defaults to True.

    Returns:
        Tuple of (parsed_data, was_repaired) where:
            - parsed_data: The parsed JSON object
            - was_repaired: True if the JSON was successfully repaired

    Raises:
        JSONDecodeError: If parsing fails and repair is disabled or fails
    """
    try:
        # First, try standard parsing
        data = json.loads(json_str)
        return data, False

    except JSONDecodeError as e:
        if not repair_on_error or repairer is None:
            # Re-raise if repair is disabled or no repairer available
            raise

        # Attempt LLM-based repair
        logger.info(f"JSON parse failed: {e}. Attempting LLM repair...")
        parsed, was_repaired = repairer.repair_and_parse(json_str, e)

        if parsed is not None:
            logger.info("JSON successfully repaired and parsed")
            return parsed, was_repaired
        else:
            # Repair failed, re-raise original error
            logger.warning("JSON repair failed, raising original error")
            raise


def try_parse_json(
    json_str: str,
    repairer: Optional[JSONRepairer] = None,
    default: Any = None,
) -> Tuple[Any, bool, Optional[str]]:
    """Try to parse JSON with repair, returning default on failure.

    This is a convenience function that never raises exceptions, returning
    a default value instead when parsing fails.

    Args:
        json_str: The JSON string to parse
        repairer: Optional JSONRepairer instance for repair attempts
        default: Default value to return on failure. Defaults to None.

    Returns:
        Tuple of (parsed_data, was_repaired, error_message) where:
            - parsed_data: The parsed JSON object, or default if failed
            - was_repaired: True if the JSON was successfully repaired
            - error_message: Error message if parsing failed, None otherwise
    """
    try:
        data, was_repaired = safe_json_loads(
            json_str, repairer=repairer, repair_on_error=(repairer is not None)
        )
        return data, was_repaired, None

    except JSONDecodeError as e:
        error_msg = f"{e.msg} at line {e.lineno}, column {e.colno}"
        return default, False, error_msg


def extract_json_from_response(text: str) -> str:
    """Extract JSON from LLM response that may contain markdown code blocks.

    LLM responses often wrap JSON in markdown code blocks (```json ... ```)
    or include explanatory text before/after the JSON. This function
    extracts the actual JSON content.

    Args:
        text: Text potentially containing JSON, possibly wrapped in markdown

    Returns:
        Extracted JSON string, or the original text if no JSON structure found
    """
    # Check for ```json blocks (most common)
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Check for generic ``` blocks
    code_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Try to find JSON array directly (for list responses)
    text_stripped: str = text.strip()

    # Try to find JSON object by matching braces
    if "{" in text_stripped:
        start: int = text_stripped.find("{")
        # Find matching closing brace
        depth: int = 0
        for i, char in enumerate(text_stripped[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text_stripped[start : i + 1]

    # Try to find JSON array by matching brackets
    if "[" in text_stripped:
        start = text_stripped.find("[")
        depth = 0
        for i, char in enumerate(text_stripped[start:], start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text_stripped[start : i + 1]

    # Return as-is if no JSON structure found
    return text


def fix_js_string_concat(text: str) -> str:
    """Fix JavaScript-style string concatenation in JSON.

    Some LLMs produce invalid JSON with JS string concatenation like:
    "value": "part1" + "part2"

    This function converts them to proper JSON strings.

    Args:
        text: JSON string that may contain JS string concatenation

    Returns:
        Fixed JSON string with concatenations resolved
    """
    # Pattern to match "string1" + "string2" patterns
    pattern = r'"([^"]*?)"\s*\+\s*"([^"]*?)"'

    def replace_concat(match) -> str:
        return f'"{match.group(1)}{match.group(2)}"'

    # Keep replacing until no more matches (handles chained concatenations)
    prev_text: str = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, replace_concat, text)

    return text
