"""Inject fresh RAQUEL eval predictions into a frozen subset review artifact.

This is useful when reusing the existing capability-controlled subset JSON for a
new model checkpoint. The subset already contains denotations and metadata; this
script refreshes `review_context.predictions` for selected model labels from
`run_raquel_eval.py --save_predictions` outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping


_WS_RE = re.compile(r"\s+")


def _normalize_question(text: str) -> str:
    return _WS_RE.sub(" ", str(text).strip())


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_prediction_map(path: str, split: str) -> Dict[str, str]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split '{split}' in {path}")
    rows = split_payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"Missing predictions list for split '{split}' in {path}")

    out: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = _normalize_question(str(row.get("question", "")))
        if not question:
            continue
        prediction = str(row.get("prediction", ""))
        out[question] = prediction
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update frozen subset predictions from RAQUEL eval prediction files."
    )
    parser.add_argument("--subset_json", required=True)
    parser.add_argument(
        "--prediction_spec",
        action="append",
        default=[],
        help="Prediction source in LABEL=PATH form. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--split",
        default="affected",
        help="Prediction split to read from each RAQUEL eval file.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.prediction_spec:
        raise ValueError("At least one --prediction_spec LABEL=PATH is required.")

    subset = _read_json(args.subset_json)
    if not isinstance(subset, dict):
        raise ValueError(f"Expected dict payload in {args.subset_json}")
    examples = subset.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in {args.subset_json}")

    sources: Dict[str, Dict[str, str]] = {}
    for spec in args.prediction_spec:
        if "=" not in spec:
            raise ValueError(f"Invalid prediction spec: {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid prediction spec: {spec!r}")
        sources[label] = _load_prediction_map(path, args.split)

    updated_counts: MutableMapping[str, int] = {label: 0 for label in sources}
    missing: List[Dict[str, Any]] = []

    for example in examples:
        if not isinstance(example, dict):
            continue
        question = _normalize_question(str(example.get("question", "")))
        review_context = example.setdefault("review_context", {})
        if not isinstance(review_context, dict):
            raise ValueError("review_context must be an object when present.")
        predictions = review_context.setdefault("predictions", {})
        if not isinstance(predictions, dict):
            raise ValueError("review_context.predictions must be an object when present.")

        for label, prediction_map in sources.items():
            prediction = prediction_map.get(question)
            if prediction is None:
                missing.append(
                    {
                        "label": label,
                        "question": question,
                        "query_index": (
                            example.get("metadata", {}).get("query_index")
                            if isinstance(example.get("metadata"), dict)
                            else None
                        ),
                    }
                )
                continue
            predictions[label] = prediction
            updated_counts[label] += 1

    output_payload = dict(subset)
    output_payload.setdefault("metadata", {})
    if isinstance(output_payload["metadata"], dict):
        output_payload["metadata"]["prediction_update"] = {
            "source_subset_json": args.subset_json,
            "split": args.split,
            "prediction_specs": args.prediction_spec,
            "updated_counts": dict(updated_counts),
            "missing_count": len(missing),
        }
    if missing:
        output_payload["missing_predictions"] = missing

    _write_json(Path(args.out), output_payload)
    print(
        json.dumps(
            {
                "out": args.out,
                "updated_counts": dict(updated_counts),
                "missing_count": len(missing),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if missing:
        raise SystemExit(
            f"Missing {len(missing)} subset predictions. See 'missing_predictions' in {args.out}."
        )


if __name__ == "__main__":
    main()
