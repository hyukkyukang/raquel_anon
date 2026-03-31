import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import logging
import os
from typing import *
import hydra
from omegaconf import DictConfig
import json

from src.aligned_db.utils import get_schema_stats
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


@hydra.main(
    version_base=None, config_path="/home/user/RAQUEL/config", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """
    Main function to construct the database.
    """
    # Parse arguments for this script
    # --sample_num: number of QA pairs to process at a time
    sample_num = cfg.get("sample_num", None)
    assert sample_num is not None, "sample_num is not set"

    # Get schema directory path
    schema_path: str = os.path.join(
        cfg.project_path, f"aligned_dbs/sample_{sample_num}/combined_db/schema.sql"
    )

    # Read the schema
    logger.info(f"Reading schema from {schema_path}...")
    with open(schema_path, "r") as f:
        schema: str = f.read()

    # Get the statistics
    stats: Dict[str, int] = get_schema_stats(schema)

    # Print the statistics
    logger.info(json.dumps(stats, indent=4))

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logging.info("Done!")
