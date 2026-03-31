"""Execute synthesized SQL queries and collect affected/unaffected QA results."""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import json
import os
from typing import Any, Dict, List, Type

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.generator.query_execution import DEFAULT_MAX_WORKERS, execute_queries
from src.llm import LLMAPICaller
from src.prompt.registry import RESULT_TO_TEXT_PROMPT_REGISTRY
from src.prompt.sql_result_to_text.prompt import ResultToTextPrompt
from src.utils.data_loaders import load_sql_queries, load_translated_queries
from src.utils.database import PathBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to execute queries and collect results.

    Args:
        cfg: Hydra configuration
    """
    # Build paths using PathBuilder
    path_builder = PathBuilder(cfg)
    sql_queries_path: str = path_builder.build_data_path(cfg.paths.sql_queries)
    translated_queries_path: str = path_builder.build_data_path(
        cfg.paths.translated_queries
    )
    metadata_path: str = (
        path_builder.build_data_path(cfg.paths.sql_queries_metadata)
        if hasattr(cfg.paths, "sql_queries_metadata")
        else ""
    )
    affected_path: str = path_builder.build_data_path(cfg.paths.affected_query_results)
    unaffected_path: str = path_builder.build_data_path(
        cfg.paths.unaffected_query_results
    )

    # Check if outputs already exist and overwrite is False
    overwrite: bool = cfg.get("overwrite", False)
    if not overwrite and os.path.exists(affected_path) and os.path.exists(unaffected_path):
        logger.info(
            f"Skipping execute_query (outputs exist: {affected_path}). "
            "Use overwrite=True to force re-run."
        )
        return

    # Load SQL queries
    sql_queries: List[str] = load_sql_queries(
        sql_queries_path, separator=cfg.paths.separator
    )

    # Load translated natural language queries
    try:
        translated_queries: List[str] = load_translated_queries(
            translated_queries_path, separator=cfg.paths.separator
        )
        if len(translated_queries) != len(sql_queries):
            logger.error(
                f"Number of translated queries ({len(translated_queries)}) "
                f"does not match number of SQL queries ({len(sql_queries)}). "
                "Please regenerate translated queries."
            )
            return
    except FileNotFoundError:
        logger.warning(f"Translated queries file not found: {translated_queries_path}")
        logger.info(
            "Please run translate_query.py first to generate the translated queries"
        )
        return

    # Load query metadata if available
    metadata_by_index: Dict[int, Dict[str, Any]] = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                for idx, record in enumerate(payload):
                    if not isinstance(record, dict):
                        continue
                    record_index = record.get("query_index", idx)
                    if isinstance(record_index, int):
                        metadata_by_index[record_index] = record
        except Exception as exc:
            logger.warning("Failed to load metadata file %s: %s", metadata_path, exc)

    # Prepare database config for thread-local connections
    db_config: Dict[str, Any] = {
        "db_id": cfg.database.db_id,
        "null_db_id": cfg.database.null_db_id,
        "user_id": cfg.database.user_id,
        "passwd": cfg.database.passwd,
        "host": cfg.database.host,
        "port": cfg.database.port,
        "null_port": cfg.database.null_port,
    }

    # Load LLM for result-to-text conversion
    api_cfg = cfg.llm.base
    llm_api = LLMAPICaller(
        model_name=api_cfg.model_name,
        max_tokens=api_cfg.max_tokens,
        temperature=api_cfg.temperature,
        use_custom_api=api_cfg.use_custom_api,
        global_cfg=cfg,
    )
    result_to_text_prompt_cls: Type[ResultToTextPrompt] = (
        RESULT_TO_TEXT_PROMPT_REGISTRY["default"]
    )

    # Get parallelism settings from config
    max_workers = (
        cfg.get("model", {})
        .get("synthesizer", {})
        .get("max_concurrency", DEFAULT_MAX_WORKERS)
    )
    affected_data, unaffected_data, skip_cnt = execute_queries(
        sql_queries=sql_queries,
        translated_queries=translated_queries,
        db_config=db_config,
        llm_api=llm_api,
        result_to_text_prompt_cls=result_to_text_prompt_cls,
        metadata_by_index=metadata_by_index,
        max_workers=max_workers,
    )

    # Log summary
    logger.info(f"Skipped {skip_cnt} queries out of {len(sql_queries)} queries")
    logger.info(
        f"{len(unaffected_data)} unaffected queries, {len(affected_data)} affected queries"
    )

    # Save results (paths already defined at the top of function)
    with open(unaffected_path, "w") as f:
        f.write(json.dumps(unaffected_data, indent=4))

    with open(affected_path, "w") as f:
        f.write(json.dumps(affected_data, indent=4))


if __name__ == "__main__":
    run_as_main(main, logger.name)
