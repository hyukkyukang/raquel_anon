"""LLM call tracking for statistics and monitoring.

This module provides an in-memory singleton tracker for LLM API calls,
enabling detailed statistics collection and reporting during pipeline execution.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("LLMCallTracker")

# Category groupings for better summary formatting
# Maps category names to list of prefixes that belong to that category
CATEGORY_PREFIXES: Dict[str, List[str]] = {
    "Discovery": ["entity_type_discovery", "attribute_discovery"],
    "Normalization": ["attribute_normalization", "entity_type_consolidation"],
    "Schema Generation": ["schema_from_attributes", "final_schema_normalization"],
    "Entity Extraction": ["entity_extraction_dynamic"],
    "Validation": ["extraction_critic"],
    "Verification": ["round_trip_comparison"],
    "Translation": ["sql_to_text_translation"],
}


class LLMCallTracker:
    """Singleton tracker for LLM API calls.

    This class tracks the number of LLM API calls per task/prefix for monitoring,
    debugging, and cost estimation. It provides methods to record calls, query counts,
    and generate formatted summaries.

    Attributes:
        _calls: Dictionary mapping prefix strings to call counts
        _initialized: Flag to prevent re-initialization of singleton

    Example:
        >>> tracker = LLMCallTracker()
        >>> tracker.record_call("entity_type_discovery")
        >>> tracker.record_call("entity_type_discovery")
        >>> tracker.get_count("entity_type_discovery")
        2
        >>> tracker.get_total()
        2
    """

    _instance: Optional["LLMCallTracker"] = None

    def __new__(cls) -> "LLMCallTracker":
        """Create or return the singleton instance.

        Returns:
            LLMCallTracker: The singleton tracker instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the tracker if not already initialized."""
        if self._initialized:
            return
        self._calls: Dict[str, int] = defaultdict(int)
        self._initialized = True

    def record_call(self, prefix: str) -> None:
        """Record an LLM call for the given task prefix.

        Args:
            prefix: The task identifier/prefix for the LLM call
        """
        self._calls[prefix] += 1

    def get_count(self, prefix: str) -> int:
        """Get the call count for a specific prefix.

        Args:
            prefix: The task identifier/prefix to query

        Returns:
            int: Number of calls recorded for this prefix
        """
        return self._calls.get(prefix, 0)

    def get_all_counts(self) -> Dict[str, int]:
        """Get all call counts as a dictionary.

        Returns:
            Dict[str, int]: Dictionary mapping prefixes to their call counts
        """
        return dict(self._calls)

    def get_total(self) -> int:
        """Get total number of LLM calls across all prefixes.

        Returns:
            int: Total number of LLM calls recorded
        """
        return sum(self._calls.values())

    def reset(self) -> None:
        """Reset all counters to zero."""
        self._calls.clear()

    def _get_categorized_counts(self) -> List[Tuple[str, int, Dict[str, int]]]:
        """Organize call counts by category.

        Returns:
            List of tuples: (category_name, category_total, {prefix: count})
        """
        categorized: List[Tuple[str, int, Dict[str, int]]] = []
        categorized_prefixes: set = set()

        # Process known categories
        for category, prefixes in CATEGORY_PREFIXES.items():
            category_counts: Dict[str, int] = {}
            category_total: int = 0

            for prefix in prefixes:
                count = self._calls.get(prefix, 0)
                if count > 0:
                    category_counts[prefix] = count
                    category_total += count
                    categorized_prefixes.add(prefix)

            if category_counts:
                categorized.append((category, category_total, category_counts))

        # Handle uncategorized prefixes
        uncategorized: Dict[str, int] = {}
        uncategorized_total: int = 0

        for prefix, count in self._calls.items():
            if prefix not in categorized_prefixes and count > 0:
                uncategorized[prefix] = count
                uncategorized_total += count

        if uncategorized:
            categorized.append(("Other", uncategorized_total, uncategorized))

        return categorized

    def log_summary(self, title: str = "LLM Call Statistics Summary") -> None:
        """Log a formatted summary of all LLM calls.

        Generates a nicely formatted summary grouped by category, showing
        individual prefix counts and totals.

        Args:
            title: Title to display at the top of the summary
        """
        total = self.get_total()

        if total == 0:
            logger.info(
                f"\n{'=' * 60}\n{title}\n{'=' * 60}\nNo LLM calls recorded.\n{'=' * 60}"
            )
            return

        # Build the summary string
        lines: List[str] = [
            "",
            "=" * 60,
            title,
            "=" * 60,
            "",
        ]

        # Add categorized counts
        categorized = self._get_categorized_counts()
        for category, category_total, prefix_counts in categorized:
            lines.append(f"{category} ({category_total} calls):")
            for prefix, count in sorted(prefix_counts.items()):
                lines.append(f"    {prefix}: {count}")
            lines.append("")

        # Add total
        lines.extend(
            [
                "-" * 60,
                f"TOTAL LLM CALLS: {total}",
                "=" * 60,
            ]
        )

        # Log the summary
        logger.info("\n".join(lines))

    def get_summary_string(self, title: str = "LLM Call Statistics Summary") -> str:
        """Get a formatted summary string of all LLM calls.

        Similar to log_summary but returns the string instead of logging it.

        Args:
            title: Title to display at the top of the summary

        Returns:
            str: Formatted summary string
        """
        total = self.get_total()

        if total == 0:
            return (
                f"\n{'=' * 60}\n{title}\n{'=' * 60}\nNo LLM calls recorded.\n{'=' * 60}"
            )

        # Build the summary string
        lines: List[str] = [
            "",
            "=" * 60,
            title,
            "=" * 60,
            "",
        ]

        # Add categorized counts
        categorized = self._get_categorized_counts()
        for category, category_total, prefix_counts in categorized:
            lines.append(f"{category} ({category_total} calls):")
            for prefix, count in sorted(prefix_counts.items()):
                lines.append(f"    {prefix}: {count}")
            lines.append("")

        # Add total
        lines.extend(
            [
                "-" * 60,
                f"TOTAL LLM CALLS: {total}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


# Module-level singleton instance for convenient access
llm_call_tracker: LLMCallTracker = LLMCallTracker()
