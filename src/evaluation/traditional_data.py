"""Lightweight configuration and IO helpers for traditional evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

EvaluationExample = Dict[str, Any]


@dataclass(frozen=True)
class TraditionalEvalConfig:
    """Configuration for the legacy truth-ratio/utility evaluation."""

    sample_num: Optional[int] = None
    generation_max_new_tokens: int = 64
    random_seed: Optional[int] = None


def _coerce_evaluation_examples(obj: Any, source: str) -> List[EvaluationExample]:
    """Normalize raw JSON data while preserving optional evaluation fields."""
    if isinstance(obj, dict):
        for candidate in ("data", "examples"):
            payload = obj.get(candidate)
            if isinstance(payload, list):
                obj = payload
                break
        else:
            raise ValueError(
                f"File '{source}' must contain 'data' or 'examples' with a list payload."
            )

    if not isinstance(obj, list):
        raise ValueError(f"File '{source}' must contain a list of examples.")

    normalized: List[EvaluationExample] = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(
                f"Example {idx} in '{source}' is not a JSON object (got {type(item)})."
            )
        if "question" not in item or "answer" not in item:
            raise ValueError(
                f"Example {idx} in '{source}' is missing 'question'/'answer' keys."
            )

        example: EvaluationExample = dict(item)
        example["question"] = str(item["question"]).strip()
        example["answer"] = str(item["answer"]).strip()

        paraphrased_answer = example.get("paraphrased_answer")
        if paraphrased_answer is not None:
            example["paraphrased_answer"] = str(paraphrased_answer).strip()

        perturbed_answer = example.get("perturbed_answer")
        if perturbed_answer is not None:
            if isinstance(perturbed_answer, list):
                example["perturbed_answer"] = [
                    str(value).strip()
                    for value in perturbed_answer
                    if str(value).strip()
                ]
            else:
                value = str(perturbed_answer).strip()
                example["perturbed_answer"] = [value] if value else []

        normalized.append(example)
    return normalized


def load_traditional_examples(file_path: str) -> List[EvaluationExample]:
    """Load traditional evaluation examples from a JSON or JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' was not found.")

    if path.suffix.lower() == ".jsonl":
        records: List[EvaluationExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of '{file_path}': {exc}"
                    ) from exc
        return _coerce_evaluation_examples(records, file_path)

    with path.open("r", encoding="utf-8") as handle:
        try:
            parsed = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse '{file_path}': {exc}") from exc

    return _coerce_evaluation_examples(parsed, file_path)
