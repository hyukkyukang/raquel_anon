import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import logging as hf_logging

from script.evaluation.utils import get_base_model_dir_component, load_fine_tuned_model

# Silence non-critical Transformers warnings to keep logs clean
hf_logging.set_verbosity_error()

from src.utils.logging import get_logger

# Configure application logger (aligned with eval.py style)
logger: logging.Logger = get_logger(__name__, __file__)
if not logger.handlers:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)

# Constants
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE: str = "Question: {question}\nAnswer:"
# Limit how many queries to generate per dataset
SAMPLE_NUM: int = 20
BASE_MODEL_NAME: str = "meta-llama/Llama-3.2-1B"
MODEL_CHOICES: List[str] = [
    "base_model",
    "full_model",
    "retain_model",
    "unlearned_model",
]
DATASET_CHOICES: List[str] = ["retain", "forget", "raquel"]

logger.info("Device selected: %s", DEVICE)
logger.info("Default base model: %s", BASE_MODEL_NAME)


BASE_MODEL_DIR: str = os.path.join(
    "model", get_base_model_dir_component(BASE_MODEL_NAME)
)
logger.info("Model artifacts (if local) will be read from under: %s", BASE_MODEL_DIR)


# Load dataset function
def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON dataset from file, assuming list of dicts with 'question' and 'answer'.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: List of QA pairs (may include extra fields).
    """
    logger.info("Loading dataset from %s", file_path)
    with open(file_path, "r") as f:
        data: List[Dict[str, Any]] = json.load(
            f
        )  # Assume list of {"question": str, "answer": str}
    logger.info("Loaded %d examples from %s", len(data), file_path)
    return data


# Load model and tokenizer
def load_model_and_tokenizer(
    model_identifier: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from the specified directory or Hugging Face model name.

    Args:
        model_identifier (str): Directory or HF model name where the model and tokenizer are located.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer.
    """
    # Use shared loader to handle LoRA adapters vs full weights consistently
    logger.info(
        "Loading model and tokenizer from '%s' (base=%s)",
        model_identifier,
        BASE_MODEL_NAME,
    )
    model, tokenizer = load_fine_tuned_model(
        model_identifier=model_identifier,
        base_model_name=BASE_MODEL_NAME,
        device_map_auto=True,
        quantize_4bit=True,
        as_trainable=False,
    )
    logger.info(
        "Model and tokenizer loaded: %s",
        getattr(model, "name_or_path", str(type(model))),
    )
    return model, tokenizer


# Generate top-5 answers for a question
def generate_top5_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    max_new_tokens: int = 50,
    num_beams: int = 5,
    num_return_sequences: int = 5,
) -> List[str]:
    """
    Generate top-5 answer sequences for the given question using beam search.

    Args:
        model (PreTrainedModel): The loaded model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        question (str): The input question.
        max_new_tokens (int): Maximum new tokens to generate (default: 50).
        num_beams (int): Number of beams for beam search (default: 5).
        num_return_sequences (int): Number of sequences to return (default: 5).

    Returns:
        List[str]: List of top-5 generated answers.
    """
    prompt: str = PROMPT_TEMPLATE.format(question=question)
    inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs: torch.Tensor = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            do_sample=False,  # Use deterministic beam search
        )

    generated_answers: List[str] = []
    for output in outputs:
        generated: str = tokenizer.decode(output, skip_special_tokens=True)
        # Extract only the answer part
        if "\nAnswer:" in generated:
            answer: str = generated.split("\nAnswer:")[1].strip()
        else:
            answer = generated.strip()
        generated_answers.append(answer)

    return generated_answers


# Save generations to file
def save_generations_to_file(
    generations: List[Dict[str, Any]], output_file: str
) -> None:
    """
    Save the generated answers to a JSON file.

    Args:
        generations (List[Dict[str, Any]]): List of dicts with question and top5_answers.
        output_file (str): Path to save the JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(generations, f, indent=4)
    logger.info("Generations saved to %s", output_file)


# Main function to generate and save
def main(
    model_choice: str = "base_model",
    dataset_choice: str = "retain",
    output_file: str = "generated_answers.json",
) -> None:
    """
    Main function to load model, generate top-5 answers, and save to file.

    Args:
        model_choice (str): One of 'base_model', 'full_model', 'retain_model', 'unlearned_model' (default: 'base_model').
        dataset_choice (str): One of 'retain', 'forget', or 'raquel' (default: 'retain').
        output_file (str): Output JSON filename. Will be saved inside the model's parameter directory.
    """
    if model_choice not in MODEL_CHOICES:
        raise ValueError(f"Invalid model choice. Must be one of {MODEL_CHOICES}")
    if dataset_choice not in DATASET_CHOICES:
        raise ValueError(f"Invalid dataset choice. Must be one of {DATASET_CHOICES}")

    # Set model identifier: HF name for base, local dir for others
    if model_choice == "base_model":
        model_identifier: str = BASE_MODEL_NAME
    else:
        # Use base-model-specific directory to avoid collisions between bases
        model_identifier: str = os.path.join(BASE_MODEL_DIR, model_choice)
    logger.info(
        "Selected model_choice=%s → model_identifier=%s", model_choice, model_identifier
    )

    # Determine output directory and output path early to allow skipping if it already exists
    if model_choice == "base_model" or not os.path.isdir(model_identifier):
        output_dir: str = os.path.join(BASE_MODEL_DIR, "base_model_outputs")
    else:
        output_dir = model_identifier
    output_filename: str = os.path.basename(output_file)
    output_path: str = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        logger.info("Output file already exists; skipping generation: %s", output_path)
        return

    # Set dataset path
    if dataset_choice == "retain":
        dataset_path: str = "data/tofu/retain90.json"
    elif dataset_choice == "forget":
        dataset_path = "data/tofu/forget10.json"
    elif dataset_choice == "raquel":
        dataset_path = "data/tofu/raquel_data.json"
    else:
        raise ValueError(f"Unsupported dataset choice: {dataset_choice}")
    logger.info(
        "Selected dataset_choice=%s → dataset_path=%s", dataset_choice, dataset_path
    )

    # Load model and dataset
    model, tokenizer = load_model_and_tokenizer(model_identifier)
    dataset: List[Dict[str, Any]] = load_dataset(dataset_path)
    # Restrict to SAMPLE_NUM examples for generation
    subset: List[Dict[str, Any]] = (
        dataset[:SAMPLE_NUM] if len(dataset) > SAMPLE_NUM else dataset
    )
    logger.info(
        "Preparing to generate for %d / %d examples (SAMPLE_NUM=%d)",
        len(subset),
        len(dataset),
        SAMPLE_NUM,
    )

    # Ensure output directory exists (path already resolved above)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Resolved output path under model directory: %s", output_path)

    generations: List[Dict[str, Any]] = []
    for ex in tqdm(subset, desc="Generating (top-5 answers)", leave=False):
        question: str = ex["question"]
        top5_answers: List[str] = generate_top5_answers(model, tokenizer, question)
        generations.append(
            {
                "question": question,
                "original_answer": ex.get(
                    "answer", "N/A"
                ),  # Include original if available
                "top5_generated": top5_answers,
            }
        )

    save_generations_to_file(generations, output_path)


if __name__ == "__main__":
    # Run once to generate for BOTH datasets. Adjust `chosen_model` if needed.
    chosen_models: List[str] = [
        "base_model",
        "full_model",
        "retain_model",
        "unlearned_model",
    ]

    for chosen_model in tqdm(chosen_models, desc="Models"):
        model_tag: str = chosen_model.replace("_model", "")  # For cleaner filenames

        for ds_choice in tqdm(("retain", "forget", "raquel"), desc="Datasets"):
            # Construct dataset-specific output filename under the model's directory
            out_file: str = f"generated_answers_{model_tag}_{ds_choice}.json"
            main(
                model_choice=chosen_model,
                dataset_choice=ds_choice,
                output_file=out_file,
            )
