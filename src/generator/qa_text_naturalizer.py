"""QA text naturalization for aligned-build preprocessing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from src.generator.naturalization_audit import summarize_naturalization_records
from src.generator.qa_text_styles import STYLE_REGISTRY, select_qa_text_style
from src.generator.qa_text_validation import validate_naturalized_qa_pair

if TYPE_CHECKING:
    from src.llm.api import LLMAPICaller


@dataclass(frozen=True)
class QATextNaturalizationRecord:
    """Canonical, normalized, and naturalized text for one QA pair."""

    qa_index: int
    source: str
    canonical_question: str
    canonical_answer: str
    normalized_question: str
    normalized_answer: str
    naturalized_question: str
    naturalized_answer: str
    style: str
    rewrite_changed: bool
    validation_passed: bool
    fallback_to_canonical: bool
    validation_fail_reasons: List[str]
    preserved_names: List[str]
    preserved_numbers: List[str]
    preserved_dates: List[str]
    preserved_titles: List[str]
    preserved_content_tokens: List[str]
    missing_content_tokens: List[str]
    content_token_recall: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class QATextNaturalizer:
    """LLM-backed QA naturalizer with deterministic validation and fallback."""

    def __init__(self, global_cfg: DictConfig, naturalization_cfg: Any) -> None:
        self.global_cfg = global_cfg
        self.naturalization_cfg = naturalization_cfg

    @cached_property
    def naturalization_model_name(self) -> str:
        return str(
            self.naturalization_cfg.get(
                "model", self.global_cfg.llm.base.model_name
            )
        )

    @cached_property
    def naturalization_max_tokens(self) -> int:
        return int(self.naturalization_cfg.get("max_tokens", 512))

    @cached_property
    def naturalization_seed(self) -> int | None:
        seed = self.naturalization_cfg.get("seed", None)
        if seed is not None:
            return int(seed)
        base_seed = getattr(self.global_cfg.llm.base, "seed", None)
        return int(base_seed) if base_seed is not None else None

    @cached_property
    def naturalization_temperature(self) -> float | None:
        temperature = self.naturalization_cfg.get("temperature", None)
        if temperature is None:
            return None
        return float(temperature)

    @cached_property
    def naturalization_api_caller(self) -> "LLMAPICaller":
        from src.llm.api import LLMAPICaller

        base_cfg = OmegaConf.to_container(self.global_cfg.llm.base, resolve=True)
        assert isinstance(base_cfg, dict)
        return LLMAPICaller(
            model_name=self.naturalization_model_name,
            max_tokens=self.naturalization_max_tokens,
            use_custom_api=bool(base_cfg.get("use_custom_api", False)),
            global_cfg=self.global_cfg,
            temperature=self.naturalization_temperature,
            seed=self.naturalization_seed,
        )

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Naturalization response was not a JSON object")
        return payload

    def rewrite_pair(
        self,
        *,
        style_id: str,
        canonical_question: str,
        canonical_answer: str,
    ) -> Tuple[str, str]:
        """Rewrite one QA pair using the configured model."""
        from src.prompt.db_construction.qa_naturalization.prompt import (
            QATextNaturalizationPrompt,
        )

        style = STYLE_REGISTRY[style_id]
        prompt = QATextNaturalizationPrompt(
            style_instruction=style.instruction,
            question=canonical_question,
            answer=canonical_answer,
        )
        response = self.naturalization_api_caller(
            prompt,
            post_process_fn=None,
            prefix="qa_text_naturalization",
        )
        payload = self._parse_json_response(response)
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Naturalization response missing question or answer")
        return question, answer


def naturalize_qa_pairs_for_aligned_build(
    canonical_qa_pairs: Sequence[Tuple[str, str]],
    *,
    normalized_qa_pairs: Sequence[Tuple[str, str]] | None = None,
    qa_sources: Sequence[str] | None = None,
    style_mode: str = "deterministic",
    fallback_to_canonical: bool = True,
    global_cfg: DictConfig | None = None,
    naturalization_cfg: Any | None = None,
) -> tuple[List[Tuple[str, str]], List[QATextNaturalizationRecord], Dict[str, object]]:
    """Return a naturalized QA corpus and per-QA naturalization metadata.

    If global_cfg/naturalization_cfg are provided, this will attempt a real
    LLM-backed rewrite. Any rewrite that fails parsing or deterministic
    validation falls back to the pre-naturalization text.
    """
    if normalized_qa_pairs is None:
        normalized_qa_pairs = canonical_qa_pairs

    naturalizer: QATextNaturalizer | None = None
    if global_cfg is not None and naturalization_cfg is not None:
        naturalizer = QATextNaturalizer(global_cfg, naturalization_cfg)
    rewrite_scope = "full_qa"
    if naturalization_cfg is not None:
        rewrite_scope = str(naturalization_cfg.get("scope", "full_qa"))
    min_content_token_recall = 0.65
    if naturalization_cfg is not None:
        raw_threshold = naturalization_cfg.get("validation_min_content_recall", None)
        if raw_threshold is not None:
            min_content_token_recall = float(raw_threshold)

    naturalized_pairs: List[Tuple[str, str]] = []
    records: List[QATextNaturalizationRecord] = []
    llm_rewrites_attempted = 0

    for idx, (canonical_pair, normalized_pair) in enumerate(
        zip(canonical_qa_pairs, normalized_qa_pairs)
    ):
        canonical_question, canonical_answer = canonical_pair
        normalized_question, normalized_answer = normalized_pair
        style = select_qa_text_style(idx, mode=style_mode)

        naturalized_question = normalized_question
        naturalized_answer = normalized_answer
        rewrite_error = None
        if naturalizer is not None:
            llm_rewrites_attempted += 1
            try:
                naturalized_question, naturalized_answer = naturalizer.rewrite_pair(
                    style_id=style,
                    canonical_question=canonical_question,
                    canonical_answer=canonical_answer,
                )
                if rewrite_scope == "question_only":
                    naturalized_answer = normalized_answer
            except Exception as exc:  # pragma: no cover - exercised in integration
                rewrite_error = str(exc)
                naturalized_question = normalized_question
                naturalized_answer = normalized_answer

        validation = validate_naturalized_qa_pair(
            canonical_question=canonical_question,
            canonical_answer=canonical_answer,
            rewritten_question=naturalized_question,
            rewritten_answer=naturalized_answer,
            min_content_token_recall=min_content_token_recall,
        )
        validation_fail_reasons = list(validation.fail_reasons)
        if rewrite_error is not None:
            validation_fail_reasons = validation_fail_reasons + [f"rewrite_error:{rewrite_error}"]

        record = QATextNaturalizationRecord(
            qa_index=idx,
            source=qa_sources[idx] if qa_sources and idx < len(qa_sources) else "unknown",
            canonical_question=canonical_question,
            canonical_answer=canonical_answer,
            normalized_question=normalized_question,
            normalized_answer=normalized_answer,
            naturalized_question=naturalized_question,
            naturalized_answer=naturalized_answer,
            style=style,
            rewrite_changed=(
                naturalized_question != normalized_question
                or naturalized_answer != normalized_answer
            ),
            validation_passed=validation.passed,
            fallback_to_canonical=(not validation.passed and fallback_to_canonical),
            validation_fail_reasons=validation_fail_reasons,
            preserved_names=validation.preserved_names,
            preserved_numbers=validation.preserved_numbers,
            preserved_dates=validation.preserved_dates,
            preserved_titles=validation.preserved_titles,
            preserved_content_tokens=validation.preserved_content_tokens,
            missing_content_tokens=validation.missing_content_tokens,
            content_token_recall=validation.content_token_recall,
        )
        records.append(record)

        if not validation.passed and fallback_to_canonical:
            naturalized_pairs.append((normalized_question, normalized_answer))
        else:
            naturalized_pairs.append((naturalized_question, naturalized_answer))

    summary = summarize_naturalization_records(record.to_dict() for record in records)
    summary["style_mode"] = style_mode
    summary["max_retries"] = 0
    summary["llm_rewrites_attempted"] = llm_rewrites_attempted
    summary["scope"] = rewrite_scope
    summary["model"] = (
        str(naturalization_cfg.get("model"))
        if naturalization_cfg is not None and naturalization_cfg.get("model") is not None
        else None
    )
    return naturalized_pairs, records, summary
