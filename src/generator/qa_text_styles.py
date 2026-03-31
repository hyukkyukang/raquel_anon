"""Style selection helpers for QA text naturalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class QATextStyle:
    """A deterministic style family for QA naturalization."""

    style_id: str
    label: str
    instruction: str


STYLE_REGISTRY: Dict[str, QATextStyle] = {
    "plain_factual": QATextStyle(
        style_id="plain_factual",
        label="Plain factual",
        instruction="Use direct factual wording with minimal stylistic flourish.",
    ),
    "biographical": QATextStyle(
        style_id="biographical",
        label="Biographical",
        instruction="Use a short, biography-like tone while staying concise.",
    ),
    "indirect_reference": QATextStyle(
        style_id="indirect_reference",
        label="Indirect reference",
        instruction="Use slightly more natural indirect wording without changing facts.",
    ),
    "short_explanatory": QATextStyle(
        style_id="short_explanatory",
        label="Short explanatory",
        instruction="Use short explanatory phrasing with natural flow.",
    ),
}

STYLE_ORDER: List[str] = list(STYLE_REGISTRY.keys())


def list_qa_text_styles() -> List[str]:
    """Return the stable style ids available for naturalization."""
    return list(STYLE_ORDER)


def select_qa_text_style(qa_index: int, *, mode: str = "deterministic") -> str:
    """Choose a stable style id for one QA pair."""
    if mode != "deterministic":
        raise ValueError(f"Unsupported naturalization style mode: {mode}")
    if not STYLE_ORDER:
        raise ValueError("No QA text naturalization styles are configured")
    return STYLE_ORDER[qa_index % len(STYLE_ORDER)]
