"""Typed settings for the aligned DB generator pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class AlignedDBPipelineSettings:
    """Resolved generator-pipeline settings derived from Hydra config."""

    entity_type_batch_size: int
    attribute_batch_size: int
    extraction_max_concurrency: int
    validation_enabled: bool
    validation_coverage_threshold: float
    validation_max_iterations: int
    verification_enabled: bool
    verification_max_iterations: int
    save_intermediate_results: bool
    discovery_max_workers: int

    @classmethod
    def from_config(cls, global_cfg: DictConfig) -> "AlignedDBPipelineSettings":
        """Build pipeline settings from the aligned-db config subtree."""
        aligned_db_cfg = global_cfg.model.aligned_db
        extraction_validation = aligned_db_cfg.get("extraction_validation", {})

        return cls(
            entity_type_batch_size=aligned_db_cfg.get("entity_type_batch_size", 20),
            attribute_batch_size=aligned_db_cfg.get("attribute_batch_size", 20),
            extraction_max_concurrency=aligned_db_cfg.get(
                "extraction_max_concurrency",
                10,
            ),
            validation_enabled=extraction_validation.get("enabled", True),
            validation_coverage_threshold=extraction_validation.get(
                "coverage_threshold",
                0.8,
            ),
            validation_max_iterations=extraction_validation.get("max_iterations", 2),
            verification_enabled=aligned_db_cfg.get("verification_enabled", True),
            verification_max_iterations=aligned_db_cfg.get(
                "verification_max_iterations",
                3,
            ),
            save_intermediate_results=aligned_db_cfg.get(
                "save_intermediate_results",
                False,
            ),
            discovery_max_workers=aligned_db_cfg.get("discovery_max_workers", 4),
        )
