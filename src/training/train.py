"""Main training entry point for RAQUEL using PyTorch Lightning and Hydra."""

from typing import Any, Dict, Optional
from omegaconf import DictConfig
from src.training.callback import ExperimentResultsCallback
from src.training.builders import build_tokenizer, create_datamodule, create_pl_module
from src.training.callback_setup import create_callbacks
from src.training.external_logging import (
    create_external_logger,
    finalize_external_logger,
    log_external_artifacts,
)
from src.training.experiment import ExperimentTracker
from src.training.runtime import (
    cleanup_checkpoints,
    coerce_metrics,
    create_trainer,
    log_effective_batch_size,
    setup_environment,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


TAG_FOR_DEBUGGING = "debugging"


def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 70)
    logger.info("RAQUEL Training with PyTorch Lightning")
    logger.info("=" * 70)

    # Setup environment
    setup_environment(cfg)

    # Initialize experiment tracker early to capture metadata
    tracker: ExperimentTracker = ExperimentTracker.from_config(cfg)
    tracker.start()

    tokenizer = build_tokenizer(cfg)

    # Create datamodule
    datamodule = create_datamodule(cfg, tokenizer)

    # Create model
    module = create_pl_module(cfg, datamodule)
    if module is None:
        logger.info("Training skipped - model already exists")
        tracker.mark_completed(
            summary={
                "status": "skipped",
                "reason": "model_exists",
                "output_dir": cfg.output.dir,
            }
        )
        return

    # Create callbacks
    callbacks = create_callbacks(cfg, tracker=tracker)
    if tracker.enabled:
        callbacks.append(ExperimentResultsCallback(tracker=tracker))

    # Optional external experiment logger integration
    external_logger: Optional[Any] = None
    if cfg.tag.strip().lower() != TAG_FOR_DEBUGGING:
        external_logger = create_external_logger(cfg)

    log_effective_batch_size(cfg)
    trainer = create_trainer(
        cfg,
        callbacks=callbacks,
        logger_instance=external_logger if external_logger is not None else False,
    )

    # Train
    logger.info("Start training...")
    final_metrics: Dict[str, float] = {}
    last_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = cfg.trainer.get("resume_from_checkpoint")
    try:
        # Resume from checkpoint if configured, otherwise start fresh.
        if resume_checkpoint:
            # Some checkpoints (notably those involving quantization / PEFT wrappers) can
            # contain extra state entries that are safe to ignore across library versions.
            # Lightning respects `module.strict_loading`; disabling strictness prevents
            # resume failures due to unexpected keys.
            setattr(module, "strict_loading", False)
            # Torch 2.6+ defaults to `weights_only=True` which can fail for Lightning
            # checkpoints containing OmegaConf objects (e.g., DictConfig in hyperparameters).
            # These checkpoints are produced locally by this project, so we can safely disable
            # weights-only loading to support resuming.
            trainer.fit(
                module,
                datamodule,
                ckpt_path=resume_checkpoint,
                weights_only=False,
            )
        else:
            trainer.fit(module, datamodule)
        final_metrics = coerce_metrics(trainer.callback_metrics)
        last_checkpoint = getattr(
            getattr(trainer, "checkpoint_callback", None), "last_model_path", None
        )
    except Exception as exc:
        tracker.mark_failed(exc)
        raise
    finally:
        if external_logger is not None:
            finalize_external_logger(external_logger)

    # Save final model
    if cfg.output.get("save_model", True):
        logger.info("Saving final model to %s", cfg.output.dir)
        module.save_model(cfg.output.dir)  # type: ignore
        cleanup_checkpoints(cfg)

    tracker.mark_completed(
        summary={
            "status": "completed",
            "output_dir": cfg.output.dir,
            "last_checkpoint": last_checkpoint,
            "final_metrics": final_metrics,
        }
    )
    if external_logger is not None:
        log_external_artifacts(
            external_logger,
            cfg,
            run_dir=tracker.run_dir if tracker.enabled else None,
        )

    logger.info("=" * 70)
    logger.info("Training completed!")
    logger.info("=" * 70)
