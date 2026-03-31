"""Convenience script for fine-tuning with retain90/holdout10 splits."""

# Suppress unnecessary warnings
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
from transformers.utils import logging as hf_logging

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Precision bf16-mixed is not supported by the model summary.*",
    category=UserWarning,
)
hf_logging.get_logger("transformers.generation.utils").setLevel(hf_logging.ERROR)  # type: ignore

# Fix the argparse error in hydra when using Python3.14
from script.hotfix import apply_python314_compatibility_patches

apply_python314_compatibility_patches()

# Load environment variables early
import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()

import hydra
import torch
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.training.train import main as train_main


@hydra.main(
    config_path=ABS_CONFIG_DIR,
    config_name="finetune/retain_base",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Fine-tune a model on the retain90 subset while validating on retain90 + holdout10.

    Usage:
        python script/train/finetune_retain.py
        python script/train/finetune_retain.py model.name=meta-llama/Llama-3.2-3B
        # Local JSON/JSONL override (bypasses HF dataset)
        python script/train/finetune_retain.py data.train_file=data/raquel/unaffected_train.json data.val_file=data/raquel/unaffected_val.json
    """
    train_main(cfg)


if __name__ == "__main__":
    main()
