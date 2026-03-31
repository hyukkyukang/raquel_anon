"""Materialize a small RAQUEL subset by query index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _parse_query_indices(raw: str) -> List[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _get_query_index(example: Dict[str, Any]) -> int | None:
    value = example.get("query_index")
    if isinstance(value, int):
        return value
    metadata = example.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("query_index")
        if isinstance(nested, int):
            return nested
    return None


def _select_examples(
    examples: Sequence[Dict[str, Any]],
    query_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    for example in examples:
        if not isinstance(example, dict):
            continue
        query_index = _get_query_index(example)
        if query_index is None:
            continue
        by_index[query_index] = example

    missing = [query_index for query_index in query_indices if query_index not in by_index]
    if missing:
        raise ValueError(f"Missing requested query indices: {missing}")

    return [by_index[query_index] for query_index in query_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a RAQUEL subset.")
    parser.add_argument("--source", required=True, help="Source RAQUEL JSON file.")
    parser.add_argument(
        "--query_indices",
        required=True,
        help="Comma-separated query indices in desired output order.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--summary_out",
        help="Optional summary JSON with counts and selected query indices.",
    )
    args = parser.parse_args()

    payload = _read_json(args.source)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {args.source}")

    query_indices = _parse_query_indices(args.query_indices)
    selected = _select_examples(payload, query_indices)
    _write_json(args.out, selected)

    if args.summary_out:
        _write_json(
            args.summary_out,
            {
                "source": args.source,
                "out": args.out,
                "count": len(selected),
                "query_indices": list(query_indices),
            },
        )


if __name__ == "__main__":
    main()
