"""Data augmentation pipeline for RAQUEL.

This module orchestrates the full data augmentation pipeline including:

1. Constructing the aligned database from QA pairs
2. Dumping and cleaning insert statements
3. Creating and populating the null database
4. Generating nullify statements to null out forget set data
5. Synthesizing SQL queries for evaluation
6. Translating queries to natural language
7. Executing queries and collecting results

Usage:
    python -m script.run_augmentation [hydra overrides]

Example:
    python -m script.run_augmentation model.aligned_db.sample_num=50
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.clean_insert_statements import main as clean_insert_statements_main
from script.stages.construct_aligned_db import main as construct_aligned_db_main
from script.stages.execute_query import main as execute_query_main
from script.stages.synthesize_query import main as synthesize_query_main
from script.stages.translate_query import main as translate_query_main
from script.stages.update_null import main as update_null_main
from script.stages.utils import setup_logging
from src.llm.tracker import llm_call_tracker
from src.utils.database import PathBuilder
from src.utils.db_operations import PostgresShellOperations
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


@dataclass
class PipelineConfig:
    """Configuration extracted from Hydra config for pipeline execution.

    This dataclass provides a cleaner interface for accessing pipeline-specific
    configuration values without repeatedly accessing nested config objects.

    Attributes:
        aligned_db_name: Name of the aligned database
        null_db_name: Name of the null database
        aligned_host: Host for the aligned database
        aligned_port: Port for the aligned database
        null_host: Host for the null database
        null_port: Port for the null database
        raw_inserts_path: Path to raw INSERT statements file
        schema_path: Path to schema SQL file
        cleaned_inserts_path: Path to cleaned INSERT statements file
        save_dir_path: Directory where aligned DB results are saved
        sql_queries_path: Path to synthesized SQL queries
        translated_queries_path: Path to translated queries
        affected_results_path: Path to affected query results
        unaffected_results_path: Path to unaffected query results
    """

    aligned_db_name: str
    null_db_name: str
    aligned_host: str
    aligned_port: int
    null_host: str
    null_port: int
    raw_inserts_path: str
    schema_path: str
    cleaned_inserts_path: str
    save_dir_path: str
    sql_queries_path: str
    translated_queries_path: str
    affected_results_path: str
    unaffected_results_path: str

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "PipelineConfig":
        """Extract pipeline config from Hydra config.

        Args:
            cfg: Full Hydra configuration

        Returns:
            PipelineConfig instance with extracted values
        """
        path_builder = PathBuilder(cfg)
        return cls(
            aligned_db_name=cfg.database.db_id,
            null_db_name=cfg.database.null_db_id,
            aligned_host=cfg.database.host,
            aligned_port=cfg.database.port,
            null_host=cfg.database.host,
            null_port=cfg.database.null_port,
            raw_inserts_path=path_builder.build_data_path(cfg.paths.raw_inserts),
            schema_path=path_builder.build_data_path(cfg.paths.schema),
            cleaned_inserts_path=path_builder.build_data_path(
                cfg.paths.cleaned_inserts
            ),
            save_dir_path=os.path.join(cfg.project_path, cfg.model.dir_path),
            sql_queries_path=path_builder.build_data_path(cfg.paths.sql_queries),
            translated_queries_path=path_builder.build_data_path(
                cfg.paths.translated_queries
            ),
            affected_results_path=path_builder.build_data_path(
                cfg.paths.affected_query_results
            ),
            unaffected_results_path=path_builder.build_data_path(
                cfg.paths.unaffected_query_results
            ),
        )

    def get_verification_summary_path(self) -> str:
        """Get path to aligned DB verification summary."""
        return os.path.join(self.save_dir_path, "verification_summary.json")

    def get_nullify_summary_path(self) -> str:
        """Get path to nullification summary."""
        return os.path.join(self.save_dir_path, "log", "nullify", "summary.json")


def _stage_completed(output_path: str) -> bool:
    """Check if a stage has completed by verifying its output file exists.

    Args:
        output_path: Path to the stage's output file

    Returns:
        True if the output file exists, False otherwise
    """
    return os.path.exists(output_path)


def _log_stage_skip(stage_name: str, output_path: str) -> None:
    """Log that a stage is being skipped.

    Args:
        stage_name: Name of the stage being skipped
        output_path: Path to the existing output file
    """
    logger.info(
        f"Skipping {stage_name} (output exists: {output_path}). "
        "Use overwrite=True to force re-run."
    )


def _log_stage_start(step_num: int, stage_name: str) -> None:
    """Log that a stage is starting.

    Args:
        step_num: Step number in the pipeline
        stage_name: Name of the stage
    """
    logger.info(f"Step {step_num}: Running {stage_name}...")


# Type alias for stage functions
StageFunc = Callable[[DictConfig], None]

# Pipeline stage registry: list of (stage_name, stage_function) tuples
# Note: Some stages (dump_database, create_null_db) are handled specially
# due to their use of shell operations
PIPELINE_STAGES: List[Tuple[str, StageFunc]] = [
    ("construct_aligned_db", construct_aligned_db_main),
    ("update_null", update_null_main),
    ("synthesize_query", synthesize_query_main),
    ("translate_query", translate_query_main),
    ("execute_query", execute_query_main),
]


def _dump_aligned_database(
    db_ops: PostgresShellOperations, pipeline_cfg: PipelineConfig
) -> None:
    """Dump the aligned database to a SQL file.

    Args:
        db_ops: PostgreSQL shell operations wrapper
        pipeline_cfg: Pipeline configuration
    """
    logger.info("Step 2: Dumping aligned database...")
    db_ops.dump_database(
        db_name=pipeline_cfg.aligned_db_name,
        output_path=pipeline_cfg.raw_inserts_path,
        host=pipeline_cfg.aligned_host,
        port=pipeline_cfg.aligned_port,
        data_only=True,
        use_inserts=True,
    )


def _create_null_database(
    db_ops: PostgresShellOperations, pipeline_cfg: PipelineConfig, overwrite: bool
) -> None:
    """Create and populate the null database.

    Args:
        db_ops: PostgreSQL shell operations wrapper
        pipeline_cfg: Pipeline configuration
        overwrite: Whether to recreate the null database even if it exists
    """
    # Check if we should skip (null DB exists and nullification was completed)
    # Path matches NullifiedDBBuilder.nullify_log_dir_path property
    nullify_summary_path = os.path.join(
        pipeline_cfg.save_dir_path, "log", "nullify", "summary.json"
    )

    if not overwrite and os.path.exists(nullify_summary_path):
        logger.info(
            "Step 3: Skipping null database creation (already completed). "
            "Use overwrite=True to recreate."
        )
        return

    logger.info("Step 3: Creating and populating null database...")

    # Drop the null database if it exists
    db_ops.drop_database(
        db_name=pipeline_cfg.null_db_name,
        host=pipeline_cfg.null_host,
        port=pipeline_cfg.null_port,
        if_exists=True,
    )

    # Create the null database
    db_ops.create_database(
        db_name=pipeline_cfg.null_db_name,
        host=pipeline_cfg.null_host,
        port=pipeline_cfg.null_port,
    )

    # Create the schema in the null database
    db_ops.execute_sql_file(
        db_name=pipeline_cfg.null_db_name,
        file_path=pipeline_cfg.schema_path,
        host=pipeline_cfg.null_host,
        port=pipeline_cfg.null_port,
    )

    # Populate the null database
    db_ops.execute_sql_file(
        db_name=pipeline_cfg.null_db_name,
        file_path=pipeline_cfg.cleaned_inserts_path,
        host=pipeline_cfg.null_host,
        port=pipeline_cfg.null_port,
    )

    # If null DB was recreated, remove the old nullification summary
    # to ensure update_null stage runs
    if os.path.exists(nullify_summary_path):
        os.remove(nullify_summary_path)
        logger.info("  Removed old nullification summary (will re-run nullification)")


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Run the full data augmentation pipeline with stage-level checkpointing.

    Each stage checks if its output already exists and skips if not overwriting.
    This enables resumable pipeline execution.

    Args:
        cfg: Hydra configuration
    """
    # Reset LLM call tracker at start of full pipeline
    llm_call_tracker.reset()

    # Extract pipeline configuration
    pipeline_cfg: PipelineConfig = PipelineConfig.from_hydra(cfg)
    db_ops: PostgresShellOperations = PostgresShellOperations(cfg)
    overwrite: bool = cfg.get("overwrite", False)

    # Skip constructing the aligned database. We already have the aligned database.
    # # ==========================================================================
    # # Step 1: Construct Aligned Database
    # # ==========================================================================
    # verification_path = pipeline_cfg.get_verification_summary_path()
    # if overwrite or not _stage_completed(verification_path):
    #     _log_stage_start(1, "construct_aligned_db")
    #     construct_aligned_db_main(cfg)
    # else:
    #     _log_stage_skip("construct_aligned_db", verification_path)

    # ==========================================================================
    # Step 2: Dump the aligned database
    # ==========================================================================
    if overwrite or not _stage_completed(pipeline_cfg.raw_inserts_path):
        _log_stage_start(2, "dump_aligned_database")
        _dump_aligned_database(db_ops, pipeline_cfg)
    else:
        _log_stage_skip("dump_aligned_database", pipeline_cfg.raw_inserts_path)

    # ==========================================================================
    # Step 3: Clean insert statements
    # ==========================================================================
    if overwrite or not _stage_completed(pipeline_cfg.cleaned_inserts_path):
        _log_stage_start(3, "clean_insert_statements")
        clean_insert_statements_main(cfg)
    else:
        _log_stage_skip("clean_insert_statements", pipeline_cfg.cleaned_inserts_path)

    # ==========================================================================
    # Step 4: Create and populate null database
    # ==========================================================================
    _log_stage_start(4, "create_null_database")
    _create_null_database(db_ops, pipeline_cfg, overwrite)

    # ==========================================================================
    # Step 5: Update null (nullification)
    # ==========================================================================
    nullify_path = pipeline_cfg.get_nullify_summary_path()
    if overwrite or not _stage_completed(nullify_path):
        _log_stage_start(5, "update_null")
        update_null_main(cfg)
    else:
        _log_stage_skip("update_null", nullify_path)

    # ==========================================================================
    # Step 6: Synthesize SQL queries
    # ==========================================================================
    if overwrite or not _stage_completed(pipeline_cfg.sql_queries_path):
        _log_stage_start(6, "synthesize_query")
        synthesize_query_main(cfg)
    else:
        _log_stage_skip("synthesize_query", pipeline_cfg.sql_queries_path)

    # ==========================================================================
    # Step 7: Translate queries to natural language
    # ==========================================================================
    if overwrite or not _stage_completed(pipeline_cfg.translated_queries_path):
        _log_stage_start(7, "translate_query")
        translate_query_main(cfg)
    else:
        _log_stage_skip("translate_query", pipeline_cfg.translated_queries_path)

    # ==========================================================================
    # Step 8: Execute queries and collect results
    # ==========================================================================
    # Check both output files for execute_query stage
    results_exist = _stage_completed(
        pipeline_cfg.affected_results_path
    ) and _stage_completed(pipeline_cfg.unaffected_results_path)
    if overwrite or not results_exist:
        _log_stage_start(8, "execute_query")
        execute_query_main(cfg)
    else:
        _log_stage_skip("execute_query", pipeline_cfg.affected_results_path)

    # Log LLM call statistics summary for full pipeline
    llm_call_tracker.log_summary(title="Full Augmentation Pipeline LLM Statistics")

    logger.info("Data generation pipeline completed successfully!")


if __name__ == "__main__":
    setup_logging()
    main()
    logger.info("Data generation pipeline completed successfully!")
