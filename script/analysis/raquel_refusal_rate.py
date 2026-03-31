"""Compute a simple refusal-rate metric from RAQUEL evaluation outputs.

This is intended for targeted methods (IDK/DPO/NPO) where "refusal" behavior
("I don't know", "cannot", "not sure", ...) is meaningful to report alongside
leakage/utility proxies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.utils.logging import get_logger

logger = get_logger("script.analysis.raquel_refusal_rate", __file__)


_REFUSAL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bi\s+do\s+not\s+know\b", re.IGNORECASE),
    re.compile(r"\bi\s+don't\s+know\b", re.IGNORECASE),
    re.compile(r"\bnot\s+sure\b", re.IGNORECASE),
    re.compile(r"\b(can(?:not|'t)|unable\s+to)\b", re.IGNORECASE),
]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _extract_split(payload: Mapping[str, Any], split: str) -> Optional[Mapping[str, Any]]:
    """Support both script outputs and training-callback outputs."""
    split_payload = payload.get(split)
    if isinstance(split_payload, dict):
        return split_payload
    return None


def _iter_predictions(split_payload: Mapping[str, Any]) -> Iterable[str]:
    preds = split_payload.get("predictions")
    if not isinstance(preds, list):
        return []
    out: List[str] = []
    for item in preds:
        if not isinstance(item, dict):
            continue
        pred = item.get("prediction")
        if isinstance(pred, str):
            out.append(pred)
    return out


def _is_refusal(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(p.search(stripped) is not None for p in _REFUSAL_PATTERNS)


def _refusal_rate(split_payload: Mapping[str, Any]) -> Dict[str, float]:
    predictions = list(_iter_predictions(split_payload))
    if not predictions:
        return {"count": 0.0, "refusal_count": 0.0, "refusal_rate": 0.0}

    refusal_count = sum(1 for p in predictions if _is_refusal(p))
    return {
        "count": float(len(predictions)),
        "refusal_count": float(refusal_count),
        "refusal_rate": float(refusal_count / len(predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute refusal rate from RAQUEL eval JSON.")
    parser.add_argument("--raquel_eval", required=True, help="Path to RAQUEL eval JSON (with predictions).")
    parser.add_argument(
        "--out",
        help="Output JSON path (default: <raquel_eval_dir>/raquel_refusal_rate.json).",
    )
    args = parser.parse_args()

    raquel_eval_path = os.path.abspath(str(args.raquel_eval))
    payload = _read_json(raquel_eval_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at {raquel_eval_path}")

    affected = _extract_split(payload, "affected")
    unaffected = _extract_split(payload, "unaffected")
    if affected is None or unaffected is None:
        raise ValueError(
            "Expected top-level 'affected' and 'unaffected' splits in RAQUEL eval JSON."
        )

    result = {
        "raquel_eval": raquel_eval_path,
        "affected": _refusal_rate(affected),
        "unaffected": _refusal_rate(unaffected),
    }

    out_path = (
        os.path.abspath(str(args.out))
        if args.out
        else os.path.join(os.path.dirname(raquel_eval_path), "raquel_refusal_rate.json")
    )
    _write_json(out_path, result)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

