"""Translate SQL queries to natural language.

This module converts SQL queries into natural language questions
using LLM-based translation. Supports parallel translation for speed.
"""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import hydra
import tqdm
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.generator.translator import SQLToTextTranslator
from src.utils.data_loaders import load_sql_queries
from src.utils.database import PathBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)

# Default parallelism settings
DEFAULT_MAX_WORKERS = 20


def translate_single_query(
    translator: SQLToTextTranslator,
    sql_query: str,
    index: int,
) -> Tuple[int, str]:
    """Translate a single SQL query to natural language.

    Args:
        translator: The translator instance
        sql_query: SQL query to translate
        index: Original index for ordering

    Returns:
        Tuple of (index, translated_text)
    """
    try:
        text_query = translator(sql_query=sql_query)
        return (index, text_query)
    except Exception as e:
        logger.error(f"Error translating query {index}: {e}")
        # Return the SQL query itself as fallback
        return (index, f"[Translation failed] {sql_query[:100]}...")


def translate_queries_parallel(
    translator: SQLToTextTranslator,
    sql_queries: List[str],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> List[str]:
    """Translate SQL queries to natural language in parallel.

    Args:
        translator: The translator instance
        sql_queries: List of SQL queries to translate
        max_workers: Maximum number of parallel workers

    Returns:
        List of translated text queries in original order
    """
    results: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        futures = {
            executor.submit(translate_single_query, translator, sql, idx): idx
            for idx, sql in enumerate(sql_queries)
        }

        # Collect results with progress bar
        with tqdm.tqdm(total=len(sql_queries), desc="Translating queries") as pbar:
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                pbar.update(1)

    # Return results in original order
    return [results[i] for i in range(len(sql_queries))]


def _load_rendered_question_overrides(metadata_path: str) -> Dict[int, str]:
    """Load pre-rendered questions from synthesized query metadata."""
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load metadata file %s: %s", metadata_path, exc)
        return {}

    overrides: Dict[int, str] = {}
    if not isinstance(payload, list):
        return overrides

    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            continue
        question = record.get("rendered_question")
        if not isinstance(question, str) or not question.strip():
            continue
        record_index = record.get("query_index", idx)
        if isinstance(record_index, int):
            overrides[record_index] = question.strip()
    return overrides


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to translate SQL queries to natural language.

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

    # Check if output already exists and overwrite is False
    overwrite: bool = cfg.get("overwrite", False)
    if not overwrite and os.path.exists(translated_queries_path):
        logger.info(
            f"Skipping translate_query (output exists: {translated_queries_path}). "
            "Use overwrite=True to force re-run."
        )
        return

    # Load SQL queries
    sql_queries: List[str] = load_sql_queries(
        sql_queries_path, separator=cfg.paths.separator
    )
    rendered_question_overrides = _load_rendered_question_overrides(metadata_path)

    # Get parallelism settings from config (use synthesizer settings if available)
    max_workers = (
        cfg.get("model", {})
        .get("synthesizer", {})
        .get("max_concurrency", DEFAULT_MAX_WORKERS)
    )
    logger.info(
        f"Translating {len(sql_queries)} queries with {max_workers} parallel workers..."
    )

    text_queries: List[str] = [""] * len(sql_queries)
    queries_to_translate: List[str] = []
    translate_indices: List[int] = []

    for idx, sql_query in enumerate(sql_queries):
        rendered_question = rendered_question_overrides.get(idx)
        if rendered_question:
            text_queries[idx] = rendered_question
        else:
            translate_indices.append(idx)
            queries_to_translate.append(sql_query)

    if queries_to_translate:
        translator = SQLToTextTranslator(cfg.llm.base, cfg)
        translated_subset = translate_queries_parallel(
            translator=translator,
            sql_queries=queries_to_translate,
            max_workers=max_workers,
        )
        for idx, translated_text in zip(translate_indices, translated_subset):
            text_queries[idx] = translated_text

    reused_count = len(rendered_question_overrides)
    if reused_count:
        logger.info(
            "Reused %d pre-rendered questions from metadata; translated %d remaining queries.",
            reused_count,
            len(queries_to_translate),
        )

    # Save the translated queries
    logger.info(f"Saving {len(text_queries)} queries to {translated_queries_path}...")
    with open(translated_queries_path, "w") as f:
        for text_query in text_queries:
            f.write(text_query + cfg.paths.separator)


if __name__ == "__main__":
    run_as_main(main, logger.name)
