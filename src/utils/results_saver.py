"""Intermediate results saving utilities.

This module provides a consistent pattern for saving intermediate results
during pipeline stages, supporting conditional saving based on configuration.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig

logger = logging.getLogger("src.utils.results_saver")


class IntermediateResultsSaver:
    """A utility class for saving intermediate results during pipeline execution.

    This class provides a consistent interface for saving JSON-formatted
    intermediate results to disk, with conditional saving based on configuration.

    Attributes:
        base_dir: Base directory for saving results
        enabled: Whether saving is enabled
    """

    def __init__(
        self,
        base_dir: str,
        enabled: bool = True,
    ) -> None:
        """Initialize the IntermediateResultsSaver.

        Args:
            base_dir: Base directory for saving results
            enabled: Whether saving is enabled (default: True)
        """
        self._base_dir: str = base_dir
        self._enabled: bool = enabled

    @classmethod
    def from_config(
        cls,
        global_cfg: DictConfig,
        sub_path: str,
    ) -> "IntermediateResultsSaver":
        """Create a saver from Hydra configuration.

        Args:
            global_cfg: Global Hydra configuration
            sub_path: Subdirectory path under the model directory

        Returns:
            Configured IntermediateResultsSaver instance
        """
        base_dir = os.path.join(
            global_cfg.project_path,
            global_cfg.model.dir_path,
            "log",
            sub_path,
        )
        enabled = global_cfg.model.aligned_db.save_intermediate_results
        return cls(base_dir=base_dir, enabled=enabled)

    @property
    def base_dir(self) -> str:
        """Get the base directory for saving results."""
        return self._base_dir

    @property
    def enabled(self) -> bool:
        """Check if saving is enabled."""
        return self._enabled

    def save(
        self,
        sub_dir_name: str,
        data: Dict[str, Any],
        suffix: Optional[str] = None,
        file_prefix: str = "result",
    ) -> Optional[str]:
        """Save intermediate results to a JSON file.

        Args:
            sub_dir_name: Subdirectory name for this step
            data: Dictionary to save as JSON
            suffix: Optional suffix for the filename
            file_prefix: Prefix for the filename (default: "result")

        Returns:
            Path to the saved file, or None if saving is disabled
        """
        if not self._enabled:
            return None

        log_dir = os.path.join(self._base_dir, sub_dir_name)
        os.makedirs(log_dir, exist_ok=True)

        file_name = f"{file_prefix}.json"
        if suffix is not None:
            file_name = f"{file_prefix}_{suffix}.json"

        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "w") as f:
            f.write(json.dumps(data, indent=4, default=str))

        return file_path

    def save_item(
        self,
        idx: int,
        data: Dict[str, Any],
        file_prefix: str = "item",
    ) -> Optional[str]:
        """Save a single indexed item to a JSON file.

        Args:
            idx: Index of the item (used for filename)
            data: Dictionary to save as JSON
            file_prefix: Prefix for the filename (default: "item")

        Returns:
            Path to the saved file, or None if saving is disabled
        """
        if not self._enabled:
            return None

        os.makedirs(self._base_dir, exist_ok=True)

        file_name = f"{file_prefix}_{idx:05d}.json"
        file_path = os.path.join(self._base_dir, file_name)

        with open(file_path, "w") as f:
            f.write(json.dumps(data, indent=4, default=str))

        return file_path

    def save_summary(
        self,
        total_items: int,
        successful: int,
        skipped: int,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save a summary of processing results.

        Args:
            total_items: Total number of items processed
            successful: Number of successfully processed items
            skipped: Number of skipped items
            additional_data: Additional data to include in the summary

        Returns:
            Path to the saved file, or None if saving is disabled
        """
        data: Dict[str, Any] = {
            "total_items": total_items,
            "successful": successful,
            "skipped": skipped,
            "success_rate": successful / total_items if total_items > 0 else 0.0,
        }

        if additional_data:
            data.update(additional_data)

        return self.save(
            sub_dir_name="summary",
            data=data,
            suffix="final",
            file_prefix="summary",
        )


class SQLStatementSaver:
    """A utility class for saving SQL statements to files.

    This class provides methods for saving SQL statements with consistent
    formatting and file naming conventions.

    Attributes:
        save_path: Path to the main SQL file
    """

    # Default separator between SQL statement groups
    DEFAULT_SEPARATOR: str = "\n\n"

    def __init__(self, save_path: str) -> None:
        """Initialize the SQLStatementSaver.

        Args:
            save_path: Path to save the SQL statements
        """
        self._save_path: str = save_path

    @property
    def save_path(self) -> str:
        """Get the save path."""
        return self._save_path

    def save(
        self,
        statements: list,
        separator: Optional[str] = None,
    ) -> str:
        """Save SQL statements to a file.

        Args:
            statements: List of SQL statements or statement groups
            separator: Separator between statement groups (default: double newline)

        Returns:
            Path to the saved file
        """
        if separator is None:
            separator = self.DEFAULT_SEPARATOR

        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)

        # Handle list of strings or list of lists
        if statements and isinstance(statements[0], list):
            # List of statement groups - join each group, then join groups
            formatted = separator.join(
                "\n".join(group) for group in statements if group
            )
        else:
            # Simple list of statements
            formatted = separator.join(str(s) for s in statements if s)

        with open(self._save_path, "w") as f:
            f.write(formatted)

        return self._save_path

    def load(self, separator: Optional[str] = None) -> list:
        """Load SQL statements from a file.

        Args:
            separator: Separator between statement groups (default: double newline)

        Returns:
            List of SQL statements

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if separator is None:
            separator = self.DEFAULT_SEPARATOR

        if not os.path.exists(self._save_path):
            raise FileNotFoundError(f"SQL file not found: {self._save_path}")

        with open(self._save_path, "r") as f:
            content = f.read().strip()

        if not content:
            return []

        return content.split(separator)

    def exists(self) -> bool:
        """Check if the save file exists."""
        return os.path.exists(self._save_path)

