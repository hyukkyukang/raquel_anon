"""Factories for training tokenizers, datamodules, and Lightning modules."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import lightning.pytorch as pl
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.training.data import FineTuneDataModule, UnlearningDataModule
from src.training.device_map import resolve_training_device_map
from src.training.methods import needs_idk_dataset, needs_reference_model
from src.training.model import FinetuneModule, UnlearningModule
from src.training.task import TASK_FINETUNE, TASK_UNLEARN
from src.training.utils import check_model_exists, load_model_and_tokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)

TAG_FOR_DEBUGGING = "debugging"


def resolve_model_device_map(cfg: DictConfig) -> object:
    """Resolve the model device map for Lightning-managed training."""
    requested_device_map = cfg.model.get("device_map", "auto")
    cuda_available = torch.cuda.is_available()
    resolved_device_map = resolve_training_device_map(
        requested_device_map=requested_device_map,
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
    )

    if resolved_device_map != requested_device_map:
        logger.info(
            "Resolved model.device_map=%r to %r for Lightning-managed training "
            "(accelerator=%s, devices=%s).",
            requested_device_map,
            resolved_device_map,
            cfg.trainer.get("accelerator", "auto"),
            cfg.trainer.get("devices", "auto"),
        )

    return resolved_device_map


def build_tokenizer(cfg: DictConfig) -> PreTrainedTokenizer:
    """Load the tokenizer used for datamodule setup."""
    tokenizer_path = cfg.model.get("path") or cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_datamodule(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
) -> pl.LightningDataModule:
    """Create the task-specific datamodule."""
    if cfg.task == TASK_FINETUNE:
        logger.info("Creating FineTuneDataModule")
        training_cfg = cfg.training
        trainer_cfg = cfg.trainer
        return FineTuneDataModule(
            name=cfg.data.name,
            train_subset_name=cfg.data.get("train_subset_name"),
            train_split=cfg.data.train_split,
            val_split=cfg.data.get("val_split"),
            tokenizer=tokenizer,
            batch_size=training_cfg.get("train_batch_size"),
            train_batch_size=training_cfg.get("train_batch_size"),
            val_batch_size=training_cfg.get("val_batch_size"),
            val_sample_num=training_cfg.get("val_sample_num"),
            num_workers=trainer_cfg.get("num_workers", 4),
            max_length=cfg.data.get("max_length", 512),
            val_subset_names=cfg.data.get("val_subset_names"),
            train_file=cfg.data.get("train_file"),
            val_file=cfg.data.get("val_file"),
            val_files=cfg.data.get("val_files"),
        )

    if cfg.task == TASK_UNLEARN:
        logger.info("Creating UnlearningDataModule")
        training_cfg = cfg.training
        trainer_cfg = cfg.trainer
        data_cfg = cfg.data
        create_idk = needs_idk_dataset(cfg.unlearning.method, cfg.regularization.method)
        return UnlearningDataModule(
            tokenizer=tokenizer,
            dataset_name=data_cfg.get("name"),
            forget_split=data_cfg.get("forget_split"),
            retain_split=data_cfg.get("retain_split", data_cfg.get("forget_split")),
            forget_file=data_cfg.get("forget_file"),
            retain_file=data_cfg.get("retain_file"),
            forget_subset_name=data_cfg.get("forget_subset_name"),
            retain_subset_name=data_cfg.get("retain_subset_name"),
            batch_size=training_cfg.get("train_batch_size"),
            train_batch_size=training_cfg.get("train_batch_size"),
            val_batch_size=training_cfg.get("val_batch_size"),
            num_workers=trainer_cfg.get("num_workers", 4),
            max_length=data_cfg.get("max_length", 512),
            create_idk=create_idk,
            idk_variation=cfg.unlearning.get("idk_variation", "random"),
            is_debugging=cfg.tag == TAG_FOR_DEBUGGING,
        )

    raise ValueError(f"Unknown task: {cfg.task}")


def resolve_model_precision(cfg: DictConfig) -> bool:
    """Resolve whether models should be loaded in fp16."""
    trainer_precision = str(cfg.trainer.get("precision", "32")).lower()
    load_in_fp16 = cfg.model.get("use_fp16", True)
    if "bf16" in trainer_precision and load_in_fp16:
        load_in_fp16 = False
    elif "16-mixed" in trainer_precision and load_in_fp16:
        load_in_fp16 = False
    return load_in_fp16


def maybe_resolve_unlearning_model_path(cfg: DictConfig) -> str:
    """Resolve the fine-tuned model path used for unlearning."""
    model_path = cfg.model.get("path")
    if model_path:
        return str(model_path)

    if cfg.model.trained_tag:
        model_path = os.path.join(
            "model", cfg.model.trained_tag, "finetune", cfg.model.name
        )
        cfg.model.path = model_path

    if not model_path:
        raise ValueError("model.path must be specified for unlearning")
    return str(model_path)


def build_unlearning_reference_model(
    cfg: DictConfig,
    *,
    load_in_fp16: bool,
) -> Optional[object]:
    """Load the frozen reference model when the method requires one."""
    if not needs_reference_model(cfg.unlearning.method, cfg.regularization.method):
        return None

    logger.info("Loading reference model for %s", cfg.unlearning.method)
    ref_model_path = (
        cfg.unlearning.get("reference_model_path")
        if cfg.unlearning.get("reference_model_path")
        else cfg.model.name
    )
    resolved_device_map = resolve_model_device_map(cfg)
    reference_model, _ = load_model_and_tokenizer(
        ref_model_path,
        device_map=resolved_device_map,
        use_fp16=load_in_fp16,
        quantize_4bit=bool(cfg.model.get("quantize_4bit", False)),
        adapter_trainable=False,
    )
    return reference_model


def create_pl_module(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
) -> Optional[pl.LightningModule]:
    """Create the task-specific Lightning module, or None when training should skip."""
    if cfg.task == TASK_FINETUNE:
        logger.info("Creating FinetuneModule")
        if check_model_exists(cfg.output.dir):
            logger.info("Model already exists at %s, skipping training", cfg.output.dir)
            return None

        load_in_fp16 = resolve_model_precision(cfg)
        resolved_device_map = resolve_model_device_map(cfg)
        model_load_path = cfg.model.get("path") or cfg.model.name
        model, tokenizer = load_model_and_tokenizer(
            model_load_path,
            device_map=resolved_device_map,
            use_fp16=load_in_fp16,
            quantize_4bit=bool(cfg.model.get("quantize_4bit", False)),
            lora=cfg.model.get("lora"),
            adapter_trainable=True,
        )
        return FinetuneModule(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.get("weight_decay", 0.01),
            warmup_ratio=cfg.training.get("warmup_ratio", 0.1),
            max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        )

    if cfg.task == TASK_UNLEARN:
        logger.info("Creating UnlearningModule for method=%s", cfg.unlearning.method)
        load_in_fp16 = resolve_model_precision(cfg)
        resolved_device_map = resolve_model_device_map(cfg)
        if "bf16" in str(cfg.trainer.get("precision", "32")).lower() and cfg.model.get(
            "use_fp16", True
        ):
            logger.warning(
                "Trainer precision=%s uses bfloat16; overriding model.use_fp16 to "
                "load weights in the model's native dtype.",
                str(cfg.trainer.get("precision", "32")).lower(),
            )
        elif "16-mixed" in str(cfg.trainer.get("precision", "32")).lower() and cfg.model.get(
            "use_fp16", True
        ):
            logger.warning(
                "Trainer precision=%s uses mixed precision; overriding model.use_fp16 "
                "to load weights in float32 for AMP compatibility.",
                str(cfg.trainer.get("precision", "32")).lower(),
            )

        model_path = maybe_resolve_unlearning_model_path(cfg)
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            device_map=resolved_device_map,
            use_fp16=load_in_fp16,
            quantize_4bit=bool(cfg.model.get("quantize_4bit", False)),
            lora=cfg.model.get("lora"),
            base_model_name_for_adapters=cfg.model.name,
            adapter_trainable=True,
        )
        reference_model = build_unlearning_reference_model(
            cfg,
            load_in_fp16=load_in_fp16,
        )
        return UnlearningModule(
            model=model,
            tokenizer=tokenizer,
            datamodule=datamodule,  # type: ignore[arg-type]
            cfg=cfg,
            unlearning_method=cfg.unlearning.method,
            regularization_method=cfg.regularization.method,
            alpha=cfg.unlearning.get("alpha", 1.0),
            beta=cfg.unlearning.get("beta", 1.0),
            reference_model=reference_model,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.get("weight_decay", 0.01),
            warmup_ratio=cfg.training.get("warmup_ratio", 0.1),
            max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
            idk_variation=cfg.unlearning.get("idk_variation", "random"),
        )

    raise ValueError(f"Unknown task: {cfg.task}")
