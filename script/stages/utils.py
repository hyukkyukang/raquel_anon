"""Shared utilities for pipeline stage modules.

This module provides common initialization, logging, and configuration patterns
used across all stage modules to reduce boilerplate code duplication.
"""

import logging
from typing import Callable

from src.constants import LOG_DATE_FORMAT, LOG_FORMAT


def init_stage() -> None:
    """Standard initialization for all stage modules.

    This function performs common setup tasks:
    - Suppresses FutureWarning messages
    - Loads environment variables from .env file

    Should be called at module level before any other imports that might
    trigger warnings or need environment variables.
    """
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    import hkkang_utils.misc as misc_utils

    misc_utils.load_dotenv()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with standard format.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        level=level,
    )

    # Suppress verbose LiteLLM logs (used internally by DSPy)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def run_as_main(main_func: Callable[[], None], logger_name: str = "") -> None:
    """Standard pattern for running a stage module as __main__.

    Sets up logging, runs the main function, and logs completion.

    Args:
        main_func: The Hydra-decorated main function to run
        logger_name: Optional logger name for the completion message
    """
    setup_logging()
    main_func()
    if logger_name:
        logging.getLogger(logger_name).info("Done!")
    else:
        logging.info("Done!")
