"""Clean and format INSERT statements from pg_dump output.

This module processes raw INSERT statements from pg_dump, removing
the 'public.' prefix and organizing them by table for cleaner output.
"""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import json
import logging
import os
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.utils.database import PathBuilder
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to clean INSERT statements.

    Args:
        cfg: Hydra configuration
    """
    # Build paths using PathBuilder
    path_builder = PathBuilder(cfg)
    raw_inserts_path: str = path_builder.build_data_path(cfg.paths.raw_inserts)
    cleaned_inserts_path: str = path_builder.build_data_path(cfg.paths.cleaned_inserts)

    # Read the raw inserts
    with open(raw_inserts_path, "r") as f:
        sql_statements: List[str] = f.readlines()

    # Filter and clean INSERT statements
    final_sql_statements: List[str] = []
    for sql_statement in sql_statements:
        if sql_statement.startswith("INSERT INTO"):
            # Remove the public. prefix
            sql_statement = sql_statement.replace(" public.", " ")
            final_sql_statements.append(sql_statement)

    # Write the final sql statements to a file with table grouping
    stats: Dict[str, int] = {}
    with open(cleaned_inserts_path, "w") as f:
        prev_table_name: str = ""
        for sql_statement in final_sql_statements:
            # Get the current table name
            current_table_name: str = sql_statement.split(" ")[2]

            # Count the number of rows in the table
            if current_table_name not in stats:
                stats[current_table_name] = 0
            stats[current_table_name] += 1

            # Write the sql statement to the file (with blank line between tables)
            if current_table_name != prev_table_name:
                f.write("\n")
                prev_table_name = current_table_name
            f.write(sql_statement)

    # Add total count for all tables
    stats["all"] = sum(stats.values())

    # Log statistics to a JSON file
    log_path: str = os.path.join(os.path.dirname(cleaned_inserts_path), "db_stats.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(json.dumps(stats, indent=4))

    logger.info(f"Cleaned {stats['all']} INSERT statements across {len(stats) - 1} tables")


if __name__ == "__main__":
    run_as_main(main, logger.name)
