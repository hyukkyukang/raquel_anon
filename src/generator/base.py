"""Base class for LLM-calling components.

This module provides a base class with shared functionality for components
that interact with LLMs, including error handling, JSON parsing, and
fallback mechanisms.
"""

import logging
from functools import cached_property
from typing import Any, Callable, Optional, Type

import dspy
from omegaconf import DictConfig

from src.llm import JSONRepairer, LLMAPICaller, TooMuchThinkingError
from src.utils.json_utils import (
    extract_json_from_response,
    fix_js_string_concat,
    safe_json_loads,
)

logger = logging.getLogger("src.generator.base")


class SimpleTextPrompt:
    """Simple prompt wrapper that takes a text string as the full prompt.

    This class provides the interface expected by LLMAPICaller.call_api(),
    wrapping a simple text string into a prompt object. Use this when you
    need to pass a dynamically constructed prompt string to the LLM.

    Example:
        prompt = SimpleTextPrompt("Extract entities from: ...")
        response = self.llm_api_caller(prompt)

    Attributes:
        _text: The full prompt text
    """

    def __init__(self, text: str) -> None:
        """Initialize with prompt text.

        Args:
            text: The full prompt text (system + user combined)
        """
        self._text: str = text

    @property
    def prompt(self) -> str:
        """Return the full prompt text (for DSPy signature compatibility)."""
        return self._text

    @property
    def system_instruction(self) -> str:
        """Return empty system instruction (text contains everything)."""
        return ""

    def get_user_prompt(self) -> str:
        """Return the full prompt text as user prompt."""
        return self._text

    def __str__(self) -> str:
        """Return string representation."""
        return self._text

    @classmethod
    def signature(cls) -> Type[dspy.Signature]:
        """Return a simple DSPy signature."""
        class SimpleSignature(dspy.Signature):
            """Simple input/output signature for text prompts."""
            prompt: str = dspy.InputField()
            response: str = dspy.OutputField()
        return SimpleSignature


class LLMComponent:
    """Base class for components that call LLMs.

    This class provides shared functionality for LLM interactions:
    - Primary and fallback LLM API callers
    - JSON repair capability
    - Error handling with TooMuchThinkingError fallback
    - JSON extraction and parsing from LLM responses

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration object
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the LLM component.

        Args:
            api_cfg: LLM API configuration containing model settings
            global_cfg: Global configuration object
        """
        self.api_cfg: DictConfig = api_cfg
        self.global_cfg: DictConfig = global_cfg

    # =========================================================================
    # Cached Properties - LLM Callers
    # =========================================================================

    @cached_property
    def json_repairer(self) -> JSONRepairer:
        """Get the JSON repairer for fixing malformed JSON from LLM responses."""
        return JSONRepairer(self.api_cfg, self.global_cfg)

    @cached_property
    def llm_api_caller(self) -> LLMAPICaller:
        """Get the primary LLM API caller (base model)."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.base,
        )

    @cached_property
    def fallback_llm_api_caller(self) -> LLMAPICaller:
        """Get the fallback LLM API caller (smart model for complex tasks)."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.smart,
        )

    # =========================================================================
    # Protected Methods - LLM Interaction
    # =========================================================================

    def _call_with_fallback(
        self,
        prompt: Any,
        prefix: str = "llm_call",
        post_process_fn: Optional[Callable] = None,
    ) -> str:
        """Call the LLM with fallback on TooMuchThinkingError.

        If the primary model raises TooMuchThinkingError (usually due to
        reasoning models spending too much on chain-of-thought), falls back
        to the smart model.

        Args:
            prompt: The prompt object to send to the LLM
            prefix: Logging prefix for tracking API calls
            post_process_fn: Optional function to post-process the response

        Returns:
            The LLM response string
        """
        try:
            return self.llm_api_caller(
                prompt,
                post_process_fn=post_process_fn,
                prefix=prefix,
            )
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking: {e}")
            logger.warning("Using fallback model...")
            return self.fallback_llm_api_caller(
                prompt,
                post_process_fn=post_process_fn,
                prefix=f"{prefix}_fallback",
            )

    def _call_with_text_fallback(
        self,
        prompt_text: str,
        prefix: str = "llm_call",
    ) -> str:
        """Call the LLM with a text prompt, with fallback on TooMuchThinkingError.

        This is a convenience method that wraps a text string in a SimpleTextPrompt
        and calls the LLM with fallback support.

        Args:
            prompt_text: The prompt text to send to the LLM
            prefix: Logging prefix for tracking API calls

        Returns:
            The LLM response string
        """
        prompt = SimpleTextPrompt(prompt_text)
        return self._call_with_fallback(prompt, prefix=prefix, post_process_fn=None)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content.

        This is a convenience wrapper around extract_json_from_response that
        also applies JS string concatenation fixes.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        extracted: str = extract_json_from_response(text)
        return fix_js_string_concat(extracted)

    def _parse_json_response(
        self,
        response: str,
        repair_on_error: bool = True,
    ) -> Any:
        """Extract and parse JSON from an LLM response.

        This method combines JSON extraction and parsing with repair capability.

        Args:
            response: Raw LLM response string
            repair_on_error: Whether to attempt LLM-based repair on parse failure

        Returns:
            Parsed JSON object (dict or list)

        Raises:
            JSONDecodeError: If parsing fails and repair is disabled or fails
        """
        json_str: str = self._extract_json(response)
        repairer: Optional[JSONRepairer] = self.json_repairer if repair_on_error else None
        data, was_repaired = safe_json_loads(
            json_str,
            repairer=repairer,
            repair_on_error=repair_on_error,
        )
        if was_repaired:
            logger.info("JSON response was repaired successfully")
        return data

    def _try_parse_json_response(
        self,
        response: str,
        default: Any = None,
    ) -> tuple[Any, bool, Optional[str]]:
        """Try to parse JSON from LLM response, returning default on failure.

        This is a safe version that never raises exceptions.

        Args:
            response: Raw LLM response string
            default: Default value to return on failure

        Returns:
            Tuple of (parsed_data, was_repaired, error_message) where:
                - parsed_data: The parsed JSON object, or default if failed
                - was_repaired: True if the JSON was successfully repaired
                - error_message: Error message if parsing failed, None otherwise
        """
        try:
            data = self._parse_json_response(response, repair_on_error=True)
            return data, False, None
        except Exception as e:
            error_msg: str = str(e)
            logger.warning(f"Failed to parse JSON response: {error_msg}")
            return default, False, error_msg
