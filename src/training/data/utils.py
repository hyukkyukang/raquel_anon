"""Utility helpers for dataset preparation and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

Example = Dict[str, str]


def _coerce_examples(obj: Any, source: str) -> List[Example]:
    """
    Normalize raw JSON structures into a list of QA examples.

    Args:
        obj: Parsed JSON payload that may be a dict, list, or other type.
        source: Path or dataset identifier used for error reporting.

    Returns:
        List of normalized QA example dictionaries.
    """
    # Attempt to unwrap common container keys if the payload is a mapping.
    if isinstance(obj, dict):
        for candidate in ("data", "examples"):
            payload: Any = obj.get(candidate)
            if isinstance(payload, list):
                obj = payload
                break
        else:
            raise ValueError(
                f"File '{source}' must contain 'data' or 'examples' with a list payload."
            )

    if not isinstance(obj, list):
        raise ValueError(f"File '{source}' must contain a list of examples.")

    normalized: List[Example] = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(
                f"Example {idx} in '{source}' is not a JSON object (got {type(item)})."
            )
        if "question" not in item or "answer" not in item:
            raise ValueError(
                f"Example {idx} in '{source}' is missing 'question'/'answer' keys."
            )
        normalized.append(
            {
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
        )
    return normalized


def load_json_dataset(file_path: str) -> List[Example]:
    """
    Load dataset examples from a JSON or JSONL file.

    Args:
        file_path: Path to the dataset file on disk.

    Returns:
        List of normalized QA example dictionaries.
    """
    path: Path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' was not found.")

    if path.suffix.lower() == ".jsonl":
        records: List[Example] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                line: str = raw_line.strip()
                if not line:
                    continue
                try:
                    obj: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of '{file_path}': {exc}"
                    ) from exc
                records.append(obj)
        return _coerce_examples(records, file_path)

    with path.open("r", encoding="utf-8") as handle:
        try:
            parsed: Any = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse '{file_path}': {exc}") from exc

    return _coerce_examples(parsed, file_path)

