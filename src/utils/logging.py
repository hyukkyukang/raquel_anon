"""Utility functions for consistent logger naming across the codebase.

This module provides a function to get logger names using the full module path
from the project root, ensuring consistent logger naming even when scripts are
run directly as __main__.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    """
    Get a logger using the full module path from the project root.

    When a module is imported normally, `__name__` already contains the full
    module path (e.g., 'src.evaluation.lightning_muse'). However, when a script
    is run directly as __main__, `__name__` is '__main__'. In that case, this
    function derives the module path from the file path.

    Args:
        name: The __name__ of the module (usually from __name__)
        file_path: The __file__ of the module (usually from __file__)
                  Required when name == '__main__'

    Returns:
        A Logger instance with the full module path as its name

    Examples:
        >>> # In a normal module
        >>> logger = get_logger(__name__)
        >>> # logger.name will be 'src.evaluation.lightning_muse'

        >>> # In a script run directly
        >>> logger = get_logger(__name__, __file__)
        >>> # logger.name will be 'script.evaluation.run_muse_eval'
    """
    # If name is not __main__, use it directly (already has full module path)
    if name != "__main__":
        return logging.getLogger(name)

    # When run as __main__, derive path from file_path
    if file_path is None:
        raise ValueError(
            "file_path must be provided when name is '__main__'. "
            "Use: get_logger(__name__, __file__)"
        )

    # Convert to Path object and resolve to absolute path
    file_path_obj = Path(file_path).resolve()

    # Find project root by looking for 'src' or 'script' directory
    # Start from the file's directory and walk up
    current = file_path_obj.parent
    project_root = None

    while current != current.parent:  # Stop at filesystem root
        # Check if this directory contains 'src' or 'script' subdirectories
        if (current / "src").is_dir() or (current / "script").is_dir():
            project_root = current
            break
        current = current.parent

    if project_root is None:
        # Fallback: use the file's directory as project root
        # This handles edge cases where the project structure is different
        project_root = file_path_obj.parent

    # Get relative path from project root
    try:
        relative_path = file_path_obj.relative_to(project_root)
    except ValueError:
        # File is not under project root, use filename only
        relative_path = file_path_obj.name

    # Convert path to module path: replace separators with dots, remove .py extension
    module_path = str(relative_path).replace("/", ".").replace("\\", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]

    # Remove __init__ suffix if present
    if module_path.endswith(".__init__"):
        module_path = module_path[:-9]

    return logging.getLogger(module_path)


def patch_hydra_argparser_for_python314() -> None:
    """Patch Hydra's get_args_parser to fix Python 3.14+ compatibility.

    Python 3.14+ requires `help` argument values to implement `__contains__`,
    `__iter__`, and `__len__` methods. Hydra's `LazyCompletionHelp` class doesn't
    implement these methods, causing argparse errors. This function patches
    Hydra's `get_args_parser` with a fixed version that includes a properly
    implemented `LazyCompletionHelp` class.

    Should be called early in the script before any Hydra operations.
    """
    _logger = logging.getLogger("src.utils.logging")

    try:
        import hydra._internal.utils

        def _patched_get_args_parser() -> argparse.ArgumentParser:
            """Create Hydra's argument parser with fixed LazyCompletionHelp."""
            from hydra import __version__

            parser = argparse.ArgumentParser(add_help=False, description="Hydra")
            parser.add_argument(
                "--help", "-h", action="store_true", help="Application's help"
            )
            parser.add_argument(
                "--hydra-help", action="store_true", help="Hydra's help"
            )
            parser.add_argument(
                "--version",
                action="version",
                help="Show Hydra's version and exit",
                version=f"Hydra {__version__}",
            )
            parser.add_argument(
                "overrides",
                nargs="*",
                help="Any key=value arguments to override config values "
                "(use dots for.nested=overrides)",
            )

            parser.add_argument(
                "--cfg",
                "-c",
                choices=["job", "hydra", "all"],
                help="Show config instead of running [job|hydra|all]",
            )
            parser.add_argument(
                "--resolve",
                action="store_true",
                help="Used in conjunction with --cfg, resolve config interpolations "
                "before printing.",
            )

            parser.add_argument("--package", "-p", help="Config package to show")

            parser.add_argument("--run", "-r", action="store_true", help="Run a job")

            parser.add_argument(
                "--multirun",
                "-m",
                action="store_true",
                help="Run multiple jobs with the configured launcher and sweeper",
            )

            # Fixed LazyCompletionHelp with required methods for Python 3.14+
            class LazyCompletionHelp:
                """Help text for shell completion that satisfies Python 3.14+ requirements."""

                def __repr__(self) -> str:
                    return "Install or Uninstall shell completion"

                def __contains__(self, item: object) -> bool:
                    return False

                def __iter__(self):
                    return iter([])

                def __len__(self) -> int:
                    return 0

            parser.add_argument(
                "--shell-completion",
                "-sc",
                action="store_true",
                help=LazyCompletionHelp(),
            )

            parser.add_argument(
                "--config-path",
                "-cp",
                help="Overrides the config_path specified in hydra.main(). "
                "The config_path is absolute or relative to the Python file "
                "declaring @hydra.main()",
            )

            parser.add_argument(
                "--config-name",
                "-cn",
                help="Overrides the config_name specified in hydra.main()",
            )

            parser.add_argument(
                "--config-dir",
                "-cd",
                help="Adds an additional config dir to the config search path",
            )

            parser.add_argument(
                "--experimental-rerun",
                help="Rerun a job from a previous config pickle",
            )

            info_choices = [
                "all",
                "config",
                "defaults",
                "defaults-tree",
                "plugins",
                "searchpath",
            ]
            parser.add_argument(
                "--info",
                "-i",
                const="all",
                nargs="?",
                action="store",
                choices=info_choices,
                help=f"Print Hydra information [{'|'.join(info_choices)}]",
            )
            return parser

        # Apply the patch
        hydra._internal.utils.get_args_parser = _patched_get_args_parser

        # Also patch in hydra.main if it's already imported
        if "hydra.main" in sys.modules:
            sys.modules["hydra.main"].get_args_parser = _patched_get_args_parser
            _logger.debug("hydra.main.get_args_parser patched")

        _logger.debug("hydra._internal.utils.get_args_parser patched")

    except ImportError:
        _logger.debug("hydra._internal.utils not found, patch not applied")
