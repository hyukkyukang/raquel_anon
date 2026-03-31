"""Generate perturbed-answer variants for paraphrased QA datasets."""

from script.stages.utils import init_stage

init_stage()

import json
import os
from typing import Dict, List

import hydra
import tqdm
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from script.stages.utils import run_as_main
from src.llm import LLMAPICaller
from src.prompt.perturb_answer import PerturbAnswerPrompt
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


class PerturbAnswerGenerator:
    """Class for generating perturbed (incorrect) answers for questions."""

    def __init__(self, cfg: DictConfig, api_cfg: DictConfig, global_cfg: DictConfig):
        """
        Initialize the perturb answer generator.

        Args:
            cfg (DictConfig): Configuration for the perturb answer generator
            api_cfg (DictConfig): API configuration
            global_cfg (DictConfig): Global configuration
        """
        self.cfg = cfg
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg
        self.max_retries = self.cfg.max_retries

    def __call__(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generate perturbed (incorrect) answers for questions.

        Args:
            data (List[Dict[str, str]]): List of dictionaries with "question" and "answer" keys

        Returns:
            List[Dict[str, str]]: List of dictionaries with original and perturbed answers
        """
        perturbed_data = []

        for i, item in enumerate(tqdm.tqdm(data, desc="Perturbing answers")):
            question = item["question"]
            answer = item["answer"]

            perturbed_answers = self._generate_perturbed_answers(question, answer, i)

            # Create the enhanced data item
            enhanced_item = dict(item)
            enhanced_item["perturbed_answer"] = perturbed_answers
            perturbed_data.append(enhanced_item)

            logger.info(f"Processed item {i+1}/{len(data)}")

        return perturbed_data

    def _generate_perturbed_answers(
        self, question: str, answer: str, item_index: int
    ) -> List[str]:
        """
        Generate perturbed (incorrect) answers for a given question and answer.

        Args:
            question (str): The question
            answer (str): The correct answer
            item_index (int): Index of the current item for logging

        Returns:
            List[str]: List of perturbed answers
        """
        for attempt in range(self.max_retries):
            try:
                prompt = PerturbAnswerPrompt(
                    question=question, answer=answer, num_perturbations=5
                )
                response = self.llm_api_caller(
                    prompt=prompt,
                    temperature=self.api_cfg.temperature,
                    prefix="perturb_answer",
                )

                # Expecting five answers separated by newlines
                perturbed_answers = [
                    x.strip() for x in response.split("\n") if x.strip()
                ]

                if len(perturbed_answers) >= 5:
                    logger.debug(
                        f"Successfully perturbed answer for item {item_index + 1}"
                    )
                    return perturbed_answers[:5]
                else:
                    logger.warning(
                        f"Fewer than 5 perturbed answers for item {item_index + 1}, attempt {attempt + 1}"
                    )

            except Exception as e:
                logger.error(
                    f"Error perturbing answer for item {item_index + 1}, attempt {attempt + 1}: {e}"
                )

        logger.warning(
            f"Failed to perturb answer for item {item_index + 1} after {self.max_retries} attempts, using empty list"
        )
        return []

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
    Load the data from the JSON file.

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
    Save the perturbed data to a JSON file.

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
    """Main function to perturb the answers in the data."""

    # List of (input_path, output_path) to process
    file_pairs = [
        (
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.affected_query_results_paraphrased,
            ),
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.affected_query_results_perturbed,
            ),
        ),
        (
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.unaffected_query_results_paraphrased,
            ),
            os.path.join(
                cfg.project_path,
                cfg.paths.data_dir,
                cfg.paths.unaffected_query_results_perturbed,
            ),
        ),
    ]

    # Initialize the perturb answer generator
    perturb_generator = PerturbAnswerGenerator(cfg.model.perturber, cfg.llm.base, cfg)

    for input_path, output_path in file_pairs:
        # Check if input file exists
        try:
            data = load_data(input_path)
        except Exception as e:
            logger.error(f"Failed to load input data from {input_path}: {e}")
            continue

        # Generate perturbed answers
        try:
            perturbed_data = perturb_generator(data)
            logger.info(
                f"Successfully generated perturbed answers for {len(perturbed_data)} items from {input_path}"
            )
        except Exception as e:
            logger.error(f"Failed to generate perturbed answers for {input_path}: {e}")
            continue

        # Save the results
        try:
            save_data(perturbed_data, output_path)
            logger.info(
                f"Perturbation completed successfully for {input_path}! Saved to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save perturbed data for {input_path}: {e}")
            continue

    return None


if __name__ == "__main__":
    run_as_main(main, logger.name)
