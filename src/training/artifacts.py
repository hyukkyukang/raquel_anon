"""Artifact helpers for qualitative training outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SemanticEvalResults = Mapping[str, Tuple[str, str, bool]]


def build_generation_artifact_records(
    generations: Sequence[Mapping[str, Any]],
    *,
    semantic_eval_results: Optional[SemanticEvalResults] = None,
) -> list[Dict[str, Any]]:
    """Normalize cached generations into JSON-serializable artifact records."""
    records: list[Dict[str, Any]] = []
    for generation in generations:
        prompt = str(generation["prompt"])
        prediction = str(generation["prediction"])
        target = str(generation["target"])
        record: Dict[str, Any] = {
            "prompt": prompt,
            "target": target,
            "prediction": prediction,
            "rougeL_fmeasure": generation.get("rougeL_fmeasure"),
            "subset": generation.get("subset"),
            "semantic_accuracy": None,
        }

        if semantic_eval_results is not None and prompt in semantic_eval_results:
            prediction_, target_, is_correct = semantic_eval_results[prompt]
            assert target == target_, f"Target mismatch: {target} != {target_}"
            assert prediction == prediction_, (
                f"Prediction mismatch: {prediction} != {prediction_}"
            )
            record["semantic_accuracy"] = bool(is_correct)

        records.append(record)
    return records


def write_generation_artifact(
    root_dir: str,
    *,
    category: str,
    subset_name: str,
    epoch: int,
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Write qualitative generation records to a stable JSONL artifact path."""
    artifact_path = (
        Path(root_dir)
        / "artifacts"
        / category
        / subset_name
        / f"epoch_{int(epoch):02d}.jsonl"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True, default=str))
            handle.write("\n")
    return str(artifact_path)
