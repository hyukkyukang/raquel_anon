"""Generate and apply nullification to the null database.

This module creates the nullified aligned database by removing entities
from the forget set while preserving retain data integrity.
"""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import logging

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.aligned_db.nullified_db import NullifiedDBBuilder
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to create the nullified database.

    This function uses the entity-first approach to nullification:
    1. Loads QAExtractionRegistry to identify forget entities
    2. Uses SchemaRegistry for FK-aware cascade deletion
    3. Verifies retain data integrity after nullification

    Args:
        cfg: Hydra configuration
    """
    # Parse arguments
    overwrite: bool = cfg.get("overwrite", False)

    # Create the nullified database builder
    builder = NullifiedDBBuilder(global_cfg=cfg)

    # Build the nullified database
    logger.info("Building nullified database...")
    result = builder.build(overwrite=overwrite)

    # Log results
    logger.info(
        f"\nNullification Results:\n"
        f"  Entities removed: {result.entities_removed}\n"
        f"  Relations removed: {result.relations_removed}\n"
        f"  Tables affected: {len(result.tables_affected)}\n"
        f"  Retain verified: {result.retain_verified}"
    )

    if result.errors:
        logger.warning(f"  Errors encountered: {len(result.errors)}")
        for error in result.errors:
            logger.warning(f"    - {error}")
        raise RuntimeError(
            "Nullification completed with recorded errors; inspect the stage log for details."
        )

    if not result.retain_verified:
        raise RuntimeError(
            "Retain data integrity verification failed! "
            "Some retain entities may have been inadvertently removed."
        )


if __name__ == "__main__":
    run_as_main(main, logger.name)
