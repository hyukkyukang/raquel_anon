"""Database connection utilities, path builders, and mixins."""

import os
from functools import cached_property
from typing import Protocol

import hkkang_utils.pg as pg_utils
from omegaconf import DictConfig


class DatabaseConfig(Protocol):
    """Protocol for database configuration objects."""

    user_id: str
    passwd: str
    host: str
    port: int
    db_id: str
    null_port: int
    null_db_id: str


class PostgresConnectionMixin:
    """Mixin class providing PostgreSQL connection management."""

    def __init__(self, global_cfg: DictConfig) -> None:
        self.global_cfg: DictConfig = global_cfg

    @cached_property
    def pg_client(self) -> pg_utils.PostgresConnector:
        """Get or create a PostgreSQL database connection."""
        return pg_utils.PostgresConnector(
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.port,
            db_id=self.global_cfg.database.db_id,
        )

    @cached_property
    def null_pg_client(self) -> pg_utils.PostgresConnector:
        """Get or create a PostgreSQL connection to the null database."""
        return pg_utils.PostgresConnector(
            user_id=self.global_cfg.database.user_id,
            passwd=self.global_cfg.database.passwd,
            host=self.global_cfg.database.host,
            port=self.global_cfg.database.null_port,
            db_id=self.global_cfg.database.null_db_id,
        )


class PathBuilder:
    """Standalone utility class for building project paths.

    This class provides methods to construct paths within the project's
    data and model directories based on the Hydra configuration.

    Example:
        >>> path_builder = PathBuilder(cfg)
        >>> schema_path = path_builder.build_data_path(cfg.paths.schema)
        >>> model_path = path_builder.build_model_path("checkpoints", "model.pt")
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize PathBuilder with configuration.

        Args:
            cfg: Hydra configuration containing project_path and paths settings
        """
        self._cfg: DictConfig = cfg

    @property
    def project_path(self) -> str:
        """Get the project root path."""
        return self._cfg.project_path

    @property
    def data_dir(self) -> str:
        """Get the data directory path."""
        return os.path.join(self._cfg.project_path, self._cfg.paths.data_dir)

    def build_data_path(self, *path_parts: str) -> str:
        """Build a path within the data directory.

        Args:
            *path_parts: Path components to join after the data directory

        Returns:
            Full path string
        """
        return os.path.join(
            self._cfg.project_path, self._cfg.paths.data_dir, *path_parts
        )

    def build_model_path(self, *path_parts: str) -> str:
        """Build a path within the model directory.

        Args:
            *path_parts: Path components to join after the model directory

        Returns:
            Full path string
        """
        return os.path.join(
            self._cfg.project_path, self._cfg.model.dir_path, *path_parts
        )

    def build_project_path(self, *path_parts: str) -> str:
        """Build a path within the project root.

        Args:
            *path_parts: Path components to join after the project root

        Returns:
            Full path string
        """
        return os.path.join(self._cfg.project_path, *path_parts)


class PathBuilderMixin:
    """Mixin class providing path building utilities.

    Deprecated: Use PathBuilder standalone class instead for new code.
    This mixin is kept for backward compatibility.
    """

    def __init__(self, global_cfg: DictConfig) -> None:
        self.global_cfg: DictConfig = global_cfg

    def build_model_path(self, *path_parts: str) -> str:
        """Build a path within the model directory."""
        return os.path.join(
            self.global_cfg.project_path, self.global_cfg.model.dir_path, *path_parts
        )

    def build_data_path(self, *path_parts: str) -> str:
        """Build a path within the data directory."""
        return os.path.join(
            self.global_cfg.project_path, self.global_cfg.paths.data_dir, *path_parts
        )