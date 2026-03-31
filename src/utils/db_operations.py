"""PostgreSQL shell operations wrapper.

This module provides a class for executing PostgreSQL command-line operations
(pg_dump, psql) in a structured and reusable way.
"""

import logging
import os
import subprocess
from typing import List, Optional

from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


class PostgresShellOperations:
    """Wrapper for PostgreSQL command-line operations.

    Provides methods for common database operations using pg_dump and psql
    command-line tools with proper environment configuration.

    Attributes:
        cfg: Hydra configuration containing database settings
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize PostgreSQL shell operations.

        Args:
            cfg: Hydra configuration with database credentials and connection info
        """
        self.cfg: DictConfig = cfg
        self._env: dict = self._build_env()

    def _build_env(self) -> dict:
        """Build environment with PGPASSWORD set.

        Returns:
            Environment dictionary with PostgreSQL password
        """
        env: dict = os.environ.copy()
        env["PGPASSWORD"] = self.cfg.database.passwd
        return env

    def _run_command(self, cmd: List[str], stdout_file: Optional[str] = None) -> None:
        """Execute a shell command with logging.

        Args:
            cmd: Command and arguments as a list
            stdout_file: Optional file path to redirect stdout

        Raises:
            subprocess.CalledProcessError: If command fails
        """
        cmd_str: str = " ".join(str(c) for c in cmd)
        logger.info(f">>> {cmd_str}")

        if stdout_file:
            with open(stdout_file, "w") as f:
                subprocess.run(cmd, env=self._env, check=True, stdout=f)
        else:
            subprocess.run(cmd, env=self._env, check=True)

    def _get_psql_base_cmd(
        self, host: str, port: int, db_name: str = "postgres"
    ) -> List[str]:
        """Build base psql command with connection parameters.

        Args:
            host: Database host
            port: Database port
            db_name: Database name to connect to

        Returns:
            List of command arguments
        """
        return [
            "psql",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            self.cfg.database.user_id,
            "-d",
            db_name,
        ]

    def dump_database(
        self,
        db_name: str,
        output_path: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        data_only: bool = True,
        use_inserts: bool = True,
    ) -> None:
        """Execute pg_dump to export database.

        Args:
            db_name: Name of database to dump
            output_path: File path to write dump to
            host: Database host (defaults to config host)
            port: Database port (defaults to config port)
            data_only: If True, only dump data (no schema)
            use_inserts: If True, use INSERT statements instead of COPY
        """
        host = host or self.cfg.database.host
        port = port or self.cfg.database.port

        cmd: List[str] = [
            "pg_dump",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            self.cfg.database.user_id,
            "-d",
            db_name,
        ]

        if data_only:
            cmd.append("--data-only")
        if use_inserts:
            cmd.append("--inserts")

        self._run_command(cmd, stdout_file=output_path)

    def drop_database(
        self,
        db_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        if_exists: bool = True,
    ) -> None:
        """Drop a database using psql.

        Args:
            db_name: Name of database to drop
            host: Database host (defaults to config host)
            port: Database port (defaults to config port)
            if_exists: If True, don't error if database doesn't exist
        """
        host = host or self.cfg.database.host
        port = port or self.cfg.database.port

        drop_cmd: str = (
            f"DROP DATABASE IF EXISTS {db_name};"
            if if_exists
            else f"DROP DATABASE {db_name};"
        )
        cmd: List[str] = self._get_psql_base_cmd(host, port) + ["-c", drop_cmd]
        self._run_command(cmd)

    def create_database(
        self,
        db_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Create a new database.

        Args:
            db_name: Name of database to create
            host: Database host (defaults to config host)
            port: Database port (defaults to config port)
        """
        host = host or self.cfg.database.host
        port = port or self.cfg.database.port

        cmd: List[str] = self._get_psql_base_cmd(host, port) + [
            "-c",
            f"CREATE DATABASE {db_name};",
        ]
        self._run_command(cmd)

    def execute_sql_file(
        self,
        db_name: str,
        file_path: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Execute SQL file against a database.

        Args:
            db_name: Name of database to execute against
            file_path: Path to SQL file
            host: Database host (defaults to config host)
            port: Database port (defaults to config port)
        """
        host = host or self.cfg.database.host
        port = port or self.cfg.database.port

        cmd: List[str] = self._get_psql_base_cmd(host, port, db_name) + [
            "-f",
            file_path,
        ]
        self._run_command(cmd)

    def execute_sql(
        self,
        db_name: str,
        sql: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Execute a SQL command against a database.

        Args:
            db_name: Name of database to execute against
            sql: SQL command to execute
            host: Database host (defaults to config host)
            port: Database port (defaults to config port)
        """
        host = host or self.cfg.database.host
        port = port or self.cfg.database.port

        cmd: List[str] = self._get_psql_base_cmd(host, port, db_name) + ["-c", sql]
        self._run_command(cmd)
