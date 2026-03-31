"""Generate paraphrased variants of affected and unaffected QA datasets."""

from script.stages.utils import init_stage

init_stage()

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import hydra
import tqdm
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.llm import LLMAPICaller
from src.prompt.paraphrasing import ParaphrasingPrompt
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


# Helper to get output path for paraphrased results
def get_paraphrased_output_path(input_path: str) -> str:
    if input_path.endswith(".json"):
        return input_path.replace(".json", "_paraphrased.json")
    elif input_path.endswith(".jsonl"):
        return input_path.replace(".jsonl", "_paraphrased.jsonl")
    else:
        return input_path + ".paraphrased"


class ParaphraseGenerator:
    """Class for generating paraphrased versions of questions and answers."""

    def __init__(self, cfg: DictConfig, api_cfg: DictConfig, global_cfg: DictConfig):
        """
        Initialize the paraphrase generator.

        Args:
            cfg (DictConfig): Configuration for the paraphrase generator
            global_cfg (DictConfig): Global configuration
        """
        self.cfg = cfg
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg
        self.max_retries: int = int(self.cfg.max_retries)
        self.max_workers: int = max(int(self.cfg.get("max_workers", 1)), 1)

    def __call__(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generate paraphrased versions of questions and answers.

        Args:
            data (List[Dict[str, str]]): List of dictionaries with "question" and "answer" keys

        Returns:
            List[Dict[str, str]]: List of dictionaries with original and paraphrased versions
        """
        if self.max_workers <= 1:
            paraphrased_data: List[Dict[str, str]] = []
            total_items: int = len(data)
            for i, item in enumerate(tqdm.tqdm(data, desc="Paraphrasing data")):
                enhanced_item: Dict[str, str] = self._paraphrase_item(
                    i, item, total_items
                )[1]
                paraphrased_data.append(enhanced_item)
            return paraphrased_data

        logger.info("Paraphrasing with %d workers.", self.max_workers)
        results: List[Optional[Dict[str, str]]] = [None] * len(data)
        futures: List[Future[Tuple[int, Dict[str, str]]]] = []
        total_items = len(data)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, item in enumerate(data):
                futures.append(
                    executor.submit(self._paraphrase_item, i, item, total_items)
                )
            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Paraphrasing data",
            ):
                index: int
                enhanced_item: Dict[str, str]
                index, enhanced_item = future.result()
                results[index] = enhanced_item

        paraphrased_data = [item for item in results if item is not None]
        return paraphrased_data

    def _paraphrase_item(
        self, item_index: int, item: Dict[str, str], total_items: int
    ) -> Tuple[int, Dict[str, str]]:
        """Paraphrase a single QA item and return it with index for ordering."""
        question: str = item["question"]
        answer: str = item["answer"]

        # Generate paraphrased question then answer to reduce concurrent API pressure.
        paraphrased_question: str = self._generate_paraphrase(
            question, "question", item_index
        )
        paraphrased_answer: str = self._generate_paraphrase(
            answer, "answer", item_index
        )

        enhanced_item: Dict[str, str] = {
            "question": question,
            "answer": answer,
            "paraphrased_question": paraphrased_question,
            "paraphrased_answer": paraphrased_answer,
        }
        logger.info("Processed item %d/%d", item_index + 1, total_items)
        return item_index, enhanced_item

    def _generate_paraphrase(self, text: str, text_type: str, item_index: int) -> str:
        """
        Generate a paraphrased version of the given text.

        Args:
            text (str): The text to paraphrase
            text_type (str): Type of text ("question" or "answer")
            item_index (int): Index of the current item for logging

        Returns:
            str: The paraphrased text
        """
        for attempt in range(self.max_retries):
            try:
                prompt = ParaphrasingPrompt(text=text, text_type=text_type)
                response = self.llm_api_caller(
                    prompt=prompt,
                    temperature=self.api_cfg.temperature,
                    prefix="paraphrase",
                )

                # Clean up the response
                paraphrased_text = response.strip()

                # Basic validation
                if paraphrased_text and len(paraphrased_text) > 0:
                    logger.debug(
                        f"Successfully paraphrased {text_type} for item {item_index + 1}"
                    )
                    return paraphrased_text
                else:
                    logger.warning(
                        f"Empty paraphrase response for {text_type} in item {item_index + 1}, attempt {attempt + 1}"
                    )

            except Exception as e:
                logger.error(
                    f"Error paraphrasing {text_type} for item {item_index + 1}, attempt {attempt + 1}: {e}"
                )

        # If all attempts failed, return the original text
        logger.warning(
            f"Failed to paraphrase {text_type} for item {item_index + 1} after {self.max_retries} attempts, using original"
        )
        return text

    @property
    def llm_api_caller(self) -> LLMAPICaller:
        """Get the LLM API caller instance."""
        return LLMAPICaller(
            model_name=self.api_cfg.model_name,
            max_tokens=self.api_cfg.max_tokens,
            temperature=self.api_cfg.temperature,
            use_custom_api=self.api_cfg.use_custom_api,
            global_cfg=self.global_cfg,
        )


def load_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load the tofu data from the JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        List[Dict[str, str]]: List of dictionaries with "question" and "answer" keys
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Validate the data structure
        if not isinstance(data, list):
            raise ValueError("Data should be a list")

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} should be a dictionary")
            if "question" not in item or "answer" not in item:
                raise ValueError(f"Item {i} should have 'question' and 'answer' keys")

        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(data: List[Dict[str, str]], file_path: str) -> None:
    """
    Save the paraphrased data to a JSON file.

    Args:
        data (List[Dict[str, str]]): The data to save
        file_path (str): Path to save the JSON file
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved {len(data)} items to {file_path}")

    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Main function to paraphrase the tofu data."""

    # List of (input_path, output_path) to process, now using config paths for output
    file_pairs = [
        (
            os.path.join(
                cfg.project_path, cfg.paths.data_dir, cfg.paths.affected_query_results
            ),
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.affected_query_results_paraphrased,
            ),
        ),
        (
            os.path.join(
                cfg.project_path, cfg.paths.data_dir, cfg.paths.unaffected_query_results
            ),
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.unaffected_query_results_paraphrased,
            ),
        ),
    ]

    # Initialize the paraphrase generator
    paraphrase_generator = ParaphraseGenerator(cfg.model.paraphraser, cfg.llm.base, cfg)

    for input_path, output_path in file_pairs:
        # Check if input file exists
        try:
            data = load_data(input_path)
        except Exception as e:
            logger.error(f"Failed to load input data from {input_path}: {e}")
            continue

        # Generate paraphrased versions
        try:
            paraphrased_data = paraphrase_generator(data)
            logger.info(
                f"Successfully generated paraphrased versions for {len(paraphrased_data)} items from {input_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate paraphrased versions for {input_path}: {e}"
            )
            continue

        # Save the results
        try:
            save_data(paraphrased_data, output_path)
            logger.info(
                f"Paraphrasing completed successfully for {input_path}! Saved to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save paraphrased data for {input_path}: {e}")
            continue

    return None


if __name__ == "__main__":
    run_as_main(main, logger.name)
