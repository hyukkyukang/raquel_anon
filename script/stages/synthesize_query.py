"""Synthesize SQL queries from database schema.

This module generates SQL queries that can be used to retrieve data
from the aligned database, using LLM-based query synthesis.
"""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import json
import os

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.generator.synthesizer import QuerySynthesizer
from src.utils.data_loaders import load_schema
from src.utils.database import PathBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to synthesize SQL queries.

    Args:
        cfg: Hydra configuration
    """
    # Build paths using PathBuilder
    path_builder = PathBuilder(cfg)
    schema_path: str = path_builder.build_data_path(cfg.paths.schema)
    output_path: str = path_builder.build_data_path(cfg.paths.sql_queries)
    metadata_path: str = (
        path_builder.build_data_path(cfg.paths.sql_queries_metadata)
        if hasattr(cfg.paths, "sql_queries_metadata")
        else ""
    )

    # Check if output already exists and overwrite is False
    overwrite: bool = cfg.get("overwrite", False)
    if not overwrite and os.path.exists(output_path):
        logger.info(
            f"Skipping synthesize_query (output exists: {output_path}). "
            "Use overwrite=True to force re-run."
        )
        if metadata_path and not os.path.exists(metadata_path):
            logger.warning(
                "Metadata path configured but missing: %s. "
                "Run with overwrite=True to regenerate metadata.",
                metadata_path,
            )
        return

    # Load the schema
    schema: str = load_schema(schema_path)

    # Instantiate the synthesizer
    synthesizer = QuerySynthesizer(cfg.model.synthesizer, cfg)

    # Synthesize queries (data will be loaded directly from database)
    if metadata_path:
        queries, metadata_records = synthesizer(schema=schema, return_metadata=True)
    else:
        queries = synthesizer(schema=schema)
        metadata_records = []

    logger.info(f"Synthesized {len(queries)} SQL queries")

    if metadata_path and metadata_records:
        if len(metadata_records) != len(queries):
            logger.warning(
                "Metadata count (%d) does not match query count (%d).",
                len(metadata_records),
                len(queries),
            )
        for idx, record in enumerate(metadata_records):
            record.setdefault("query_index", idx)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)
        logger.info("Saved query metadata to %s", metadata_path)


if __name__ == "__main__":
    run_as_main(main, logger.name)
