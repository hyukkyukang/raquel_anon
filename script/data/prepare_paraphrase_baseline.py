"""Prepare paraphrase-only training data from paraphrased RAQUEL outputs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger("script.data.prepare_paraphrase_baseline", __file__)


@dataclass
class ParaphraseConfig:
    """Configuration for paraphrase-only dataset conversion."""

    use_paraphrased_answer: bool
    fallback_to_original: bool
    keep_metadata: bool
    keep_original_fields: bool


def _load_examples(file_path: str) -> List[Dict[str, Any]]:
    """Load QA examples from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)

    # Support both list payloads and wrapped formats.
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or payload

    if not isinstance(payload, list):
        raise ValueError(f"Expected list of examples in {file_path}")

    examples: List[Dict[str, Any]] = [
        item for item in payload if isinstance(item, dict)
    ]
    return examples


def _coerce_text(value: Any) -> str:
    """Normalize a potentially missing value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _select_text(
    example: Dict[str, Any],
    primary_key: str,
    fallback_key: str,
    allow_fallback: bool,
) -> str:
    """Select text from a primary or fallback key."""
    primary_text: str = _coerce_text(example.get(primary_key))
    if primary_text:
        return primary_text
    if allow_fallback:
        fallback_text: str = _coerce_text(example.get(fallback_key))
        return fallback_text
    return ""


def _convert_example(
    example: Dict[str, Any], config: ParaphraseConfig
) -> Optional[Dict[str, Any]]:
    """Convert a single example to paraphrase-only format."""
    question_text: str = _select_text(
        example,
        primary_key="paraphrased_question",
        fallback_key="question",
        allow_fallback=config.fallback_to_original,
    )
    if not question_text:
        return None

    answer_key: str = "paraphrased_answer" if config.use_paraphrased_answer else "answer"
    answer_text: str = _select_text(
        example,
        primary_key=answer_key,
        fallback_key="answer",
        allow_fallback=config.fallback_to_original,
    )
    if not answer_text:
        return None

    converted: Dict[str, Any] = {"question": question_text, "answer": answer_text}

    if config.keep_metadata and "metadata" in example:
        converted["metadata"] = example.get("metadata")

    if config.keep_original_fields:
        converted["original_question"] = _coerce_text(example.get("question"))
        converted["original_answer"] = _coerce_text(example.get("answer"))

    return converted


def _write_examples(file_path: str, examples: List[Dict[str, Any]]) -> None:
    """Write converted examples to JSON."""
    output_dir: str = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)


def _process_file(
    input_path: str, output_path: str, config: ParaphraseConfig
) -> None:
    """Convert a paraphrased JSON file into training-ready QA pairs."""
    raw_examples: List[Dict[str, Any]] = _load_examples(input_path)
    converted_examples: List[Dict[str, Any]] = []
    skipped: int = 0

    for idx, example in enumerate(raw_examples):
        converted: Optional[Dict[str, Any]] = _convert_example(example, config)
        if converted is None:
            skipped += 1
            logger.warning(
                "Skipping example %d in %s (missing paraphrase fields).",
                idx,
                input_path,
            )
            continue
        converted_examples.append(converted)

    _write_examples(output_path, converted_examples)
    logger.info(
        "Saved %d paraphrase-only examples to %s (skipped=%d).",
        len(converted_examples),
        output_path,
        skipped,
    )


def main() -> None:
    """CLI entry point for paraphrase-only dataset preparation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prepare paraphrase-only training datasets."
    )
    parser.add_argument("--input", help="Input JSON file to convert.")
    parser.add_argument("--output", help="Output JSON file path.")
    parser.add_argument(
        "--affected_in",
        help="Affected paraphrased JSON file (paired mode).",
    )
    parser.add_argument(
        "--unaffected_in",
        help="Unaffected paraphrased JSON file (paired mode).",
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory for paired mode (affected_paraphrase.json, unaffected_paraphrase.json).",
    )
    parser.add_argument(
        "--use_paraphrased_answer",
        action="store_true",
        help="Use paraphrased answers instead of original answers.",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable fallback to original question/answer.",
    )
    parser.add_argument(
        "--keep_metadata",
        action="store_true",
        help="Keep the metadata field if present.",
    )
    parser.add_argument(
        "--keep_original_fields",
        action="store_true",
        help="Keep original question/answer fields as 'original_question'/'original_answer'.",
    )
    args: argparse.Namespace = parser.parse_args()

    config: ParaphraseConfig = ParaphraseConfig(
        use_paraphrased_answer=bool(args.use_paraphrased_answer),
        fallback_to_original=not bool(args.no_fallback),
        keep_metadata=bool(args.keep_metadata),
        keep_original_fields=bool(args.keep_original_fields),
    )

    if args.input:
        if not args.output:
            raise ValueError("--output is required when using --input.")
        _process_file(args.input, args.output, config)
        return None

    if not (args.affected_in and args.unaffected_in and args.out_dir):
        raise ValueError(
            "Paired mode requires --affected_in, --unaffected_in, and --out_dir."
        )

    out_dir: str = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    affected_out: str = f"{out_dir}/affected_paraphrase.json"
    unaffected_out: str = f"{out_dir}/unaffected_paraphrase.json"

    _process_file(args.affected_in, affected_out, config)
    _process_file(args.unaffected_in, unaffected_out, config)
    return None


if __name__ == "__main__":
    main()
