"""Checkpoint management for resumable AlignedDB builds.

This module provides checkpoint functionality for saving and resuming
build progress, enabling recovery from interruptions without starting
from scratch.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("src.utils.checkpoint")


@dataclass
class BuildCheckpoint:
    """Represents a checkpoint in the build process.

    Attributes:
        stage: Current stage ("schema" or "upsert")
        last_completed_index: Index of last completed batch/QA pair
        schema: Current schema state (list of CREATE TABLE statements)
        upserts: List of all upserts executed so far
        timestamp: ISO format timestamp of when checkpoint was saved
    """

    stage: str  # "schema" or "upsert"
    last_completed_index: int
    schema: List[str]
    upserts: List[str]
    timestamp: str


class CheckpointManager:
    """Manages checkpoints for resumable AlignedDB builds.

    Saves progress after each batch/QA pair so builds can be resumed
    after interruption without starting from scratch.

    Attributes:
        _checkpoint_dir: Directory to store checkpoint files
        _checkpoint_file: Full path to the checkpoint file
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self._checkpoint_dir: str = checkpoint_dir
        self._checkpoint_file: str = os.path.join(
            checkpoint_dir, "build_checkpoint.json"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"CheckpointManager initialized at {checkpoint_dir}")

    def save_schema_checkpoint(self, batch_index: int, schema: List[str]) -> None:
        """Save checkpoint after schema batch processing.

        Args:
            batch_index: Index of last completed batch
            schema: Current schema state
        """
        checkpoint = BuildCheckpoint(
            stage="schema",
            last_completed_index=batch_index,
            schema=schema,
            upserts=[],
            timestamp=datetime.now().isoformat(),
        )
        self._save(checkpoint)
        logger.info(
            f"CheckpointManager: Schema checkpoint saved (batch={batch_index}, tables={len(schema)})"
        )

    def save_upsert_checkpoint(
        self, index: int, schema: List[str], upserts: List[str]
    ) -> None:
        """Save checkpoint after upsert processing.

        Args:
            index: Index of last completed QA pair
            schema: Final schema
            upserts: List of all upserts so far
        """
        checkpoint = BuildCheckpoint(
            stage="upsert",
            last_completed_index=index,
            schema=schema,
            upserts=upserts,
            timestamp=datetime.now().isoformat(),
        )
        self._save(checkpoint)
        logger.info(
            f"CheckpointManager: Upsert checkpoint saved (index={index}, upserts={len(upserts)})"
        )

    def load_checkpoint(self) -> Optional[BuildCheckpoint]:
        """Load existing checkpoint if available.

        Returns:
            BuildCheckpoint if exists and valid, None otherwise
        """
        logger.info(
            f"CheckpointManager: Looking for checkpoint at {self._checkpoint_file}"
        )
        if not os.path.exists(self._checkpoint_file):
            logger.info("CheckpointManager: No checkpoint file found")
            return None

        try:
            with open(self._checkpoint_file, "r") as f:
                data = json.load(f)
            checkpoint = BuildCheckpoint(**data)
            logger.info(
                f"CheckpointManager: Loaded checkpoint successfully\n"
                f"  Stage: {checkpoint.stage}\n"
                f"  Last completed index: {checkpoint.last_completed_index}\n"
                f"  Schema tables: {len(checkpoint.schema)}\n"
                f"  Upserts: {len(checkpoint.upserts)}\n"
                f"  Timestamp: {checkpoint.timestamp}"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"CheckpointManager: Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        if os.path.exists(self._checkpoint_file):
            os.remove(self._checkpoint_file)
            logger.info("CheckpointManager: Checkpoint cleared after successful build")
        else:
            logger.info("CheckpointManager: No checkpoint to clear")

    def _save(self, checkpoint: BuildCheckpoint) -> None:
        """Save checkpoint to file.

        Args:
            checkpoint: BuildCheckpoint to save
        """
        with open(self._checkpoint_file, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2)
