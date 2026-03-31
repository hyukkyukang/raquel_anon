"""Pipeline stage modules for data augmentation.

This package contains individual stage modules that can be run
independently or orchestrated through run_augmentation.py.

Available stages:
- construct_aligned_db: Build the aligned database from QA pairs
- clean_insert_statements: Process pg_dump output
- update_null: Generate and apply nullify statements
- synthesize_query: Generate SQL queries from schema
- translate_query: Convert SQL to natural language
- execute_query: Execute queries and collect results
"""

from script.stages.utils import init_stage, run_as_main, setup_logging

__all__ = [
    "init_stage",
    "setup_logging",
    "run_as_main",
]
