import json
import os
from typing import Dict, List

from datasets import Dataset, load_dataset

# Output directory where JSON files will be written
OUTPUT_DIR: str = "data/tofu"


def save_dataset(dataset: Dataset, dir_path: str, file_name: str) -> None:
    """
    Save a Hugging Face Dataset's question/answer pairs to a pretty-printed JSON file.

    Args:
        dataset (Dataset): The dataset split (e.g., 'train') to serialize.
        dir_path (str): Directory to create if missing and write into.
        file_name (str): JSON file name to write.

    Returns:
        None: Writes a file to disk and returns None.
    """
    # Ensure output directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Convert the dataset to a list of dicts with only the required keys
    rows: List[Dict[str, str]] = []
    for example in dataset:  # Each example is a dict-like object
        question: str = str(example.get("question", ""))  # Ensure string type
        answer: str = str(example.get("answer", ""))  # Ensure string type
        paraphrased_answer: str = str(example["paraphrased_answer"])
        perturbed_answer: List[str] = example["perturbed_answer"]
        paraphrased_question: str = str(example["paraphrased_question"])
        rows.append(
            {
                "question": question,
                "answer": answer,
                "paraphrased_answer": paraphrased_answer,
                "perturbed_answer": perturbed_answer,
                "paraphrased_question": paraphrased_question,
            }
        )

    # Write JSON with indentation for readability
    target_path: str = os.path.join(dir_path, file_name)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def main() -> None:
    """
    Download specified TOFU subsets and save their 'train' split as JSON lists
    of question/answer pairs in OUTPUT_DIR.
    """
    subsets: List[str] = [
        "forget10_perturbed",
        "retain_perturbed",
    ]

    for subset_name in subsets:
        # Load only the 'train' split to get a Dataset (not a DatasetDict)
        train_split: Dataset = load_dataset("locuslab/TOFU", subset_name, split="train")
        save_dataset(train_split, OUTPUT_DIR, f"{subset_name}.json")


if __name__ == "__main__":
    main()
