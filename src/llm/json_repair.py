"""LLM-based JSON repair for malformed JSON strings.

This module provides functionality to use an LLM to fix syntax errors in
malformed JSON strings that fail to parse with json.loads().
"""

import json
import logging
import re
from functools import cached_property
from json import JSONDecodeError
from typing import Optional

from omegaconf import DictConfig

from src.llm.api import LLMAPICaller
from src.prompt.json_repair.prompt import JSONRepairPrompt

logger = logging.getLogger("JSONRepairer")


class JSONRepairer:
    """LLM-based JSON repair for malformed JSON strings.

    This class uses an LLM to attempt to fix syntax errors in JSON strings
    that fail to parse. It's designed as a fallback mechanism when standard
    JSON parsing fails.

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration object
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the JSONRepairer.

        Args:
            api_cfg: LLM API configuration containing model settings
            global_cfg: Global configuration object
        """
        self.api_cfg: DictConfig = api_cfg
        self.global_cfg: DictConfig = global_cfg

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def llm_api_caller(self) -> LLMAPICaller:
        """Get the LLM API caller for JSON repair.

        Uses the base (fast) model since JSON repair is a simple task.

        Returns:
            LLMAPICaller instance configured for JSON repair
        """
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.base,
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def repair(self, json_str: str, error: JSONDecodeError) -> str:
        """Attempt to repair a malformed JSON string using LLM.

        Args:
            json_str: The malformed JSON string
            error: The JSONDecodeError from the failed parse attempt

        Returns:
            The repaired JSON string, or the original string if repair fails
        """
        error_message = self._format_error_message(error)
        logger.debug(f"Attempting to repair JSON with error: {error_message}")

        # Create prompt for LLM
        prompt = JSONRepairPrompt(
            json_string=json_str,
            error_message=error_message,
        )

        try:
            # Call LLM to repair
            response: str = self.llm_api_caller(
                prompt,
                post_process_fn=None,
                prefix="json_repair",
            )

            # Extract JSON from response (in case LLM added explanations)
            repaired_json = self._extract_json(response)

            # Validate the repaired JSON
            json.loads(repaired_json)
            logger.debug("JSON repair successful")
            return repaired_json

        except JSONDecodeError as e:
            logger.warning(f"Repaired JSON still invalid: {e}")
            return json_str
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return json_str

    def repair_and_parse(
        self, json_str: str, error: JSONDecodeError
    ) -> tuple[Optional[dict], bool]:
        """Repair and parse a malformed JSON string.

        Args:
            json_str: The malformed JSON string
            error: The JSONDecodeError from the failed parse attempt

        Returns:
            Tuple of (parsed_data, was_repaired) where:
                - parsed_data: The parsed JSON object, or None if repair fails
                - was_repaired: True if the JSON was successfully repaired
        """
        repaired_str = self.repair(json_str, error)

        try:
            parsed = json.loads(repaired_str)
            # Check if repair actually changed the string
            was_repaired = repaired_str != json_str
            return parsed, was_repaired
        except JSONDecodeError:
            return None, False

    # =========================================================================
    # Protected Methods
    # =========================================================================

    def _format_error_message(self, error: JSONDecodeError) -> str:
        """Format a JSONDecodeError into a descriptive message.

        Args:
            error: The JSONDecodeError to format

        Returns:
            A formatted error message string
        """
        return (
            f"{error.msg} at line {error.lineno}, column {error.colno} "
            f"(character {error.pos})"
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or explanations.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        # Check for ```json blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Check for ``` blocks
        code_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find JSON object directly (handles both { } and [ ])
        # Find the outermost JSON structure
        text = text.strip()

        # Try to find JSON object
        if "{" in text:
            start = text.find("{")
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        # Try to find JSON array
        if "[" in text:
            start = text.find("[")
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        # Return as-is if no JSON structure found
        return text
