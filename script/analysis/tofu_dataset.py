import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import json
import logging
from typing import Dict, List, Tuple

import hydra
from datasets import load_dataset
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


def load_qa_dataset(cfg: DictConfig, name: str) -> List[Dict[str, str]]:
    """Load the dataset."""
    dataset = load_dataset(
        path=cfg.dataset.huggingface_path,
        name=name,
        split=cfg.dataset.split,
    )
    retain_qa_pairs: List[Dict[str, str]] = [
        {"question": item["question"], "answer": item["answer"]}
        for item in dataset  # type: ignore
    ]
    return retain_qa_pairs


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """
    Main function to construct the database.
    """

    # Load dataset (retain)
    retain_qa_pairs: List[Dict[str, str]] = load_qa_dataset(cfg, "retain90")
    # Load dataset (forget10)
    forget_qa_pairs: List[Dict[str, str]] = load_qa_dataset(cfg, "forget10")

    # Save each dataset to a json file
    with open("retain90.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(retain_qa_pairs, ensure_ascii=False, indent=4))
    with open("forget10.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(forget_qa_pairs, ensure_ascii=False, indent=4))
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logging.info("Done!")
