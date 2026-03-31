import logging
from typing import List, Tuple
from omegaconf import DictConfig

from src.llm import LLMAPICaller, TooMuchThinkingError
from src.prompt.db_construction.schema_coverage_check import SchemaCoverageCheckPrompt

logger = logging.getLogger("SchemaCoverageChecker")


class SchemaCoverageChecker:
    """
    Analyzes whether an existing database schema adequately covers the information
    contained in a new question-answer pair, helping to determine if schema
    modification is necessary.
    """

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig):
        """
        Initialize the coverage checker with LLM API configuration.

        Args:
            api_cfg: Configuration for the LLM API caller
            global_cfg: Global configuration containing project settings
        """
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg
        self.llm_api_caller = LLMAPICaller(
            global_cfg=global_cfg,
            **api_cfg.base,
        )
        self.fallback_llm_api_caller = LLMAPICaller(
            global_cfg=global_cfg,
            **api_cfg.smart,
        )

    def check_coverage(
        self, current_schema: List[str], qa_pair: Tuple[str, str]
    ) -> bool:
        """
        Check if the current schema adequately covers the information in the QA pair.

        Args:
            current_schema: List of SQL CREATE TABLE statements representing current schema
            qa_pair: Tuple of (question, answer) to analyze

        Returns:
            bool: True if schema modification is needed, False if current schema is sufficient
        """
        if not current_schema:
            # If no schema exists, we definitely need modification
            logger.info("No existing schema found, modification needed")
            return True

        prompt = SchemaCoverageCheckPrompt(
            current_schema=current_schema,
            new_qa_pair=qa_pair
        )

        try:
            result = self.llm_api_caller(
                prompt,
                post_process_fn=self._parse_coverage_result,
                prefix="schema_coverage_check"
            )
            needs_modification = result[0] if result else True
            logger.info(
                f"Coverage check result: {'modification needed' if needs_modification else 'schema sufficient'}"
            )
            return needs_modification

        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking in coverage check: {e}")
            logger.warning("Using fallback model for coverage check...")

            try:
                result = self.fallback_llm_api_caller(
                    prompt,
                    post_process_fn=self._parse_coverage_result,
                    prefix="schema_coverage_check_fallback"
                )
                needs_modification = result[0] if result else True
                logger.info(
                    f"Fallback coverage check result: {'modification needed' if needs_modification else 'schema sufficient'}"
                )
                return needs_modification

            except Exception as fallback_error:
                logger.error(f"Fallback coverage check failed: {fallback_error}")
                # If both attempts fail, err on the side of caution and assume modification is needed
                logger.warning("Coverage check failed, assuming modification is needed")
                return True

    def _parse_coverage_result(self, response: str) -> List[bool]:
        """
        Parse the LLM response to determine if modification is needed.

        Args:
            response: Raw response from the LLM

        Returns:
            List[bool]: Single-element list with True if modification needed, False otherwise
        """
        response_clean = response.strip().upper()

        # Look for clear indicators in the response
        modification_indicators = [
            "YES", "TRUE", "MODIFICATION NEEDED", "MODIFY", "UPDATE",
            "ADD", "EXTEND", "INSUFFICIENT", "MISSING"
        ]

        sufficient_indicators = [
            "NO", "FALSE", "SUFFICIENT", "ADEQUATE", "COVERS",
            "NO MODIFICATION", "NO CHANGES", "ALREADY COVERED"
        ]

        # Check for clear modification indicators first
        for indicator in modification_indicators:
            if indicator in response_clean:
                return [True]

        # Check for sufficiency indicators
        for indicator in sufficient_indicators:
            if indicator in response_clean:
                return [False]

        # If unclear, err on the side of caution and assume modification is needed
        logger.warning(f"Unclear coverage check response: {response}")
        return [True]