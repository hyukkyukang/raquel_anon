"""Construct the aligned database from QA pairs.

This module creates a PostgreSQL database with tables populated from
question-answer pairs loaded from a HuggingFace dataset.
"""

from script.stages.utils import init_stage

# Initialize stage (suppress warnings, load dotenv)
init_stage()

import logging
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.aligned_db.db import AlignedDB
from src.dataset import resolve_benchmark_bundle
from src.generator.qa_text_naturalizer import naturalize_qa_pairs_for_aligned_build
from src.generator.qa_text_normalizer import normalize_qa_pairs_for_aligned_build
from src.utils.data_loaders import load_qa_split
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314

logger = get_logger(__name__, __file__)

patch_hydra_argparser_for_python314()


def _build_base_qa_pair_records(
    qa_pairs: List[Tuple[str, str]],
    qa_sources: List[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for idx, (question, answer) in enumerate(qa_pairs):
        records.append(
            {
                "qa_index": idx,
                "source": qa_sources[idx] if idx < len(qa_sources) else "unknown",
                "original_question": question,
                "original_answer": answer,
                "normalized_question": question,
                "normalized_answer": answer,
                "changed": False,
                "changes": [],
            }
        )
    return records


def _load_aligned_db_qa_pairs(
    cfg: DictConfig,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load benchmark-backed retain/forget QA pairs for aligned-db generation."""
    bundle = resolve_benchmark_bundle(cfg)
    if not bundle.supports_aligned_db:
        raise ValueError(
            f"Benchmark '{bundle.benchmark}' does not support aligned-db generation yet. "
            "The current aligned-db pipeline requires QA retain/forget splits. "
            "Add a text-to-QA stage for text benchmarks such as MUSE-News first."
        )

    if not bundle.has_split("aligned_db_retain") or not bundle.has_split("aligned_db_forget"):
        raise ValueError(
            f"Benchmark '{bundle.benchmark}' is marked as aligned-db capable but does not "
            "define both 'aligned_db_retain' and 'aligned_db_forget' logical splits."
        )

    return (
        load_qa_split(cfg, "aligned_db_retain"),
        load_qa_split(cfg, "aligned_db_forget"),
    )


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to construct the aligned database.

    Args:
        cfg: Hydra configuration
    """
    # Parse arguments
    overwrite: bool = cfg.get("overwrite", False)
    max_retries: int = cfg.model.aligned_db.max_retries
    sample_num: int = cfg.model.aligned_db.sample_num

    retain_qa_pairs, forget_qa_pairs = _load_aligned_db_qa_pairs(cfg)

    # Sample QA pairs
    sampled_retain: List[Tuple[str, str]] = retain_qa_pairs[:sample_num]
    sampled_forget: List[Tuple[str, str]] = forget_qa_pairs[:sample_num]

    # Combine the sampled QA pairs
    full_sampled_qa_pairs: List[Tuple[str, str]] = sampled_retain + sampled_forget

    # Build source labels for nullification tracking
    qa_sources: List[str] = (
        ["retain"] * len(sampled_retain) + ["forget"] * len(sampled_forget)
    )
    logger.info(
        f"QA sources: {len(sampled_retain)} retain, {len(sampled_forget)} forget"
    )

    qa_pair_records = _build_base_qa_pair_records(full_sampled_qa_pairs, qa_sources)
    normalization_summary = None
    naturalization_summary = None
    discovery_qa_pairs = full_sampled_qa_pairs
    normalized_qa_pairs = full_sampled_qa_pairs
    naturalized_qa_pairs = None
    extraction_qa_pairs = full_sampled_qa_pairs
    normalization_cfg = cfg.model.aligned_db.get("qa_text_normalization", {})
    if normalization_cfg.get("enabled", False):
        normalized_pairs, qa_pair_records_obj, normalization_summary = (
            normalize_qa_pairs_for_aligned_build(
                full_sampled_qa_pairs,
                qa_sources=qa_sources,
            )
        )
        normalized_qa_pairs = normalized_pairs
        qa_pair_records = [record.to_dict() for record in qa_pair_records_obj]
        if normalization_cfg.get("use_for_discovery", False):
            discovery_qa_pairs = normalized_pairs
        if normalization_cfg.get("use_for_stage4", True):
            extraction_qa_pairs = normalized_pairs
        logger.info(
            "QA normalization: changed %d/%d pairs (%s)",
            normalization_summary.get("changed_pairs", 0),
            normalization_summary.get("total_pairs", 0),
            normalization_summary.get("change_counts", {}),
        )

    naturalization_cfg = cfg.model.aligned_db.get("qa_text_naturalization", {})
    if naturalization_cfg.get("enabled", False):
        naturalized_pairs, naturalization_records, naturalization_summary = (
            naturalize_qa_pairs_for_aligned_build(
                full_sampled_qa_pairs,
                normalized_qa_pairs=normalized_qa_pairs,
                qa_sources=qa_sources,
                style_mode=str(
                    naturalization_cfg.get("style_mode", "deterministic")
                ),
                fallback_to_canonical=bool(
                    naturalization_cfg.get("fallback_to_canonical", True)
                ),
                global_cfg=cfg,
                naturalization_cfg=naturalization_cfg,
            )
        )
        naturalized_qa_pairs = naturalized_pairs
        naturalization_by_index = {
            record.qa_index: record.to_dict() for record in naturalization_records
        }
        for record in qa_pair_records:
            qa_index = int(record.get("qa_index", -1))
            record["naturalization"] = naturalization_by_index.get(qa_index, {})
        if naturalization_cfg.get("use_for_stage4", False):
            extraction_qa_pairs = naturalized_pairs
        logger.info(
            "QA naturalization: accepted %d/%d pairs (%s)",
            naturalization_summary.get("accepted_pairs", 0),
            naturalization_summary.get("total_pairs", 0),
            naturalization_summary.get("style_counts", {}),
        )

    # Create and build the aligned database
    aligned_db = AlignedDB(global_cfg=cfg)
    aligned_db.build(
        full_sampled_qa_pairs,
        qa_sources=qa_sources,
        discovery_qa_pairs=discovery_qa_pairs,
        normalized_qa_pairs=normalized_qa_pairs,
        naturalized_qa_pairs=naturalized_qa_pairs,
        extraction_qa_pairs=extraction_qa_pairs,
        qa_pair_records=qa_pair_records,
        qa_pair_normalization_summary=normalization_summary,
        qa_pair_naturalization_summary=naturalization_summary,
        overwrite=overwrite,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    run_as_main(main, logger.name)
