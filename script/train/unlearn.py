"""Convenience script for unlearning."""

# Suppress unnecessary warnings
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning
from transformers.utils import logging as hf_logging

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
# Suppress PyTorch TF32 deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*Please use the new API settings to control TF32 behavior.*",
    category=UserWarning,
)
# Suppress Lightning DataLoader worker warning
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
warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size will be removed in a future PyTorch release.*",
    category=FutureWarning,
)
# Suppress Transformers generation config warning
warnings.filterwarnings(
    "ignore",
    message=".*The following generation flags are not valid.*",
    category=UserWarning,
)
hf_logging.get_logger("transformers.generation.utils").setLevel(hf_logging.ERROR)  # type: ignore


# Fix the argparse error in hydra when using Python3.14
from script.hotfix import apply_python314_compatibility_patches

apply_python314_compatibility_patches()

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.training.train import main as train_main


@hydra.main(
    config_path=ABS_CONFIG_DIR,
    config_name="unlearn/ga",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Perform unlearning on a fine-tuned model.

    This is a convenience wrapper around the main training script
    that uses unlearning configurations by default.

    Usage:
        # Use default method (ga + gd)
        python script/train/unlearn.py \
            model.path=model/default/meta-llama/Llama-3.2-1B/full_model

        # Specify method via config override
        python script/train/unlearn.py --config-name=unlearn/ga \
            unlearn/method@_global_=npo \
            unlearn/regularization@_global_=kl \
            model.path=model/default/meta-llama/Llama-3.2-1B/full_model

        # Override parameters
        python script/train/unlearn.py model.path=... unlearning.alpha=2.0 unlearning.beta=0.5

        # Multiple methods
        for method in ga npo idk dpo; do
            for reg in gd kl; do
                python script/train/unlearn.py --config-name=unlearn/ga \\
                    unlearn/method@_global_=$method \\
                    unlearn/regularization@_global_=$reg \\
                    model.path=model/default/meta-llama/Llama-3.2-1B/full_model
            done
        done
    """
    train_main(cfg)


if __name__ == "__main__":
    main()
