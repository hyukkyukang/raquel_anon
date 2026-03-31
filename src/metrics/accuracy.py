"""Semantic equivalence metric powered by shared LLM utilities."""

import asyncio
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omegaconf import DictConfig

from src.llm import LLMAPICaller
from src.prompt.semantic_equivalence.prompt import SemanticEquivalencePrompt
from src.utils.async_utils import AsyncRateLimiter
from src.utils.env import GOOGLE_API_KEY, OPENAI_API_KEY
from src.utils.logging import get_logger
from src.utils.string import normalize_string

logger = get_logger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are evaluating question answering outputs. "
    "Given a question, a gold answer, and a model prediction, decide if the "
    "prediction correctly answers the question in a way that is semantically "
    "consistent with the gold answer. Focus on meaning rather than exact "
    'wording. Respond with JSON: {"results": [{"index": <int>, "reason": <str>, "match": <bool>}]}.'
)


@dataclass
class SemanticMetricConfig:
    """Configuration container for the semantic accuracy metric."""

    enabled: bool = True
    model_name: str = "openai/gpt-5.4-nano-2026-03-17"
    temperature: float = 0.0
    max_tokens: int = 2048
    use_custom_api: bool = False
    batch_size: int = 8
    max_examples: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrency: int = 1
    requests_per_second: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class SemanticAccuracyMetric:
    """Accumulate QA pairs and judge semantic accuracy via LLM."""

    def __init__(
        self,
        config: SemanticMetricConfig,
        global_cfg: Optional[Any] = None,
        llm_caller: Optional[LLMAPICaller] = None,
    ) -> None:
        if llm_caller is None:
            if "gemini" in config.model_name.lower():
                if GOOGLE_API_KEY is None:
                    raise RuntimeError(
                        "GOOGLE_API_KEY is required to enable semantic accuracy evaluation with Gemini."
                    )
            elif OPENAI_API_KEY is None:
                raise RuntimeError(
                    "OPENAI_API_KEY is required to enable semantic accuracy evaluation."
                )

        self.config = config
        self._global_cfg = self._sanitize_global_cfg(global_cfg)
        self._llm = llm_caller or self._build_llm_caller()
        self._pending_pairs: List[Tuple[str, str, str]] = []
        self._cache: Dict[str, bool] = {}
        self._cache_lock = threading.Lock()
        self._last_eval_results: List[Tuple[str, str, str, bool]] = []

    def update(
        self,
        questions: Iterable[str],
        predictions: Iterable[str],
        references: Iterable[str],
    ) -> None:
        """Store new QA triples (question, prediction, reference) for later evaluation."""
        qs = list(questions)
        preds = list(predictions)
        refs = list(references)

        if not (len(qs) == len(preds) == len(refs)):
            raise ValueError(
                "Questions, predictions, and references must have matching lengths."
            )

        if not preds or not qs:
            return

        for question, pred, ref in zip(qs, preds, refs):
            if not ref or not question:
                continue
            if self.config.max_examples is not None:
                if len(self._pending_pairs) >= self.config.max_examples:
                    break
            self._pending_pairs.append((question, pred, ref))

    def compute(self) -> Optional[float]:
        """Run OpenAI judgments and return semantic accuracy."""
        if not self._pending_pairs:
            self._last_eval_results = []
            return None

        batches = list(self._iter_batches(self._pending_pairs, self.config.batch_size))
        if len(batches) > 1 and self.config.max_concurrency > 1:
            judgments = self._score_batches_parallel(batches)
        else:
            judgments = self._score_batches_serial(batches)

        if not judgments:
            self._last_eval_results = []
            return None

        correct = sum(1 for match in judgments if match)
        self._last_eval_results = [
            (question, prediction, reference, match)
            for (question, prediction, reference), match in zip(
                self._pending_pairs, judgments
            )
        ]
        return correct / len(judgments)

    def reset(self) -> None:
        """Clear accumulated QA pairs."""
        self._pending_pairs = []
        self._last_eval_results = []

    def get_last_eval_results(self) -> Sequence[Tuple[str, str, str, bool]]:
        """Return the detailed results from the most recent compute call."""
        return list(self._last_eval_results)

    def get_last_eval_results_in_dic(self) -> Dict[str, Tuple[str, str, bool]]:
        """Return the detailed results from the most recent compute call in dictionary format.
        Key is the question string.
        Value contains a tuple of (prediction, reference, match_result).
        """
        return {
            question: (prediction, reference, is_correct)
            for question, prediction, reference, is_correct in self._last_eval_results
        }

    def _score_batch(self, batch: Sequence[Tuple[str, str, str]]) -> List[bool]:
        """Score a batch, reusing cached decisions when available."""
        decisions: List[Optional[bool]] = [None] * len(batch)
        eval_payload: List[Dict[str, str]] = []
        payload_to_batch: Dict[int, int] = {}

        for idx, (question, prediction, reference) in enumerate(batch):
            cache_key = self._make_cache_key(question, prediction, reference)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                decisions[idx] = cached
                continue

            payload_index = len(eval_payload)
            eval_payload.append(
                {
                    "index": payload_index,
                    "question": question,
                    "prediction": prediction,
                    "gold_answer": reference,
                }
            )
            payload_to_batch[payload_index] = idx

        if eval_payload:
            new_decisions = self._request_judgments(eval_payload)
            for payload_index, match in new_decisions.items():
                absolute_idx = payload_to_batch.get(payload_index)
                if absolute_idx is None:
                    continue

                decisions[absolute_idx] = match
                cache_key = self._make_cache_key(*batch[absolute_idx])
                with self._cache_lock:
                    self._cache[cache_key] = match

        for idx, choice in enumerate(decisions):
            if choice is None:
                decisions[idx] = False

        return [bool(choice) for choice in decisions]

    def _score_batches_serial(
        self, batches: Sequence[Sequence[Tuple[str, str, str]]]
    ) -> List[bool]:
        judgments: List[bool] = []
        for batch in batches:
            judgments.extend(self._score_batch(batch))
        return judgments

    def _score_batches_parallel(
        self, batches: Sequence[Sequence[Tuple[str, str, str]]]
    ) -> List[bool]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._score_batches_async(batches))
        logger.warning(
            "Running event loop detected; falling back to sequential semantic scoring."
        )
        return self._score_batches_serial(batches)

    async def _score_batches_async(
        self, batches: Sequence[Sequence[Tuple[str, str, str]]]
    ) -> List[bool]:
        max_concurrency = max(int(self.config.max_concurrency), 1)
        rate_limit = float(self.config.requests_per_second or 0.0)
        semaphore = asyncio.Semaphore(max_concurrency)
        rate_limiter = AsyncRateLimiter(rate=rate_limit)
        executor = ThreadPoolExecutor(max_workers=max_concurrency)
        results: List[Optional[List[bool]]] = [None] * len(batches)

        async def _score_one(idx: int, batch: Sequence[Tuple[str, str, str]]):
            async with semaphore:
                await rate_limiter.acquire()
                loop = asyncio.get_running_loop()
                batch_scores = await loop.run_in_executor(
                    executor, self._score_batch, batch
                )
                return idx, batch_scores

        try:
            tasks = [
                asyncio.create_task(_score_one(idx, batch))
                for idx, batch in enumerate(batches)
            ]
            for coro in asyncio.as_completed(tasks):
                idx, batch_scores = await coro
                results[idx] = batch_scores
        finally:
            executor.shutdown(wait=False)

        return [
            score for batch_scores in results if batch_scores for score in batch_scores
        ]

    def _request_judgments(
        self,
        pairs: Sequence[Dict[str, str]],
    ) -> Dict[int, bool]:
        """Call the shared LLM API and parse boolean judgments per pair."""
        content = json.dumps({"pairs": pairs}, ensure_ascii=False)
        attempt = 0

        while attempt < self.config.max_retries:
            attempt += 1
            try:
                prompt = SemanticEquivalencePrompt(
                    instruction=self.config.system_prompt,
                    payload=content,
                )
                raw_response = self._llm(
                    prompt,
                    temperature=self.config.temperature,
                )
                message = self._coerce_string_output(raw_response)
                return self._parse_response(message, len(pairs))
            except Exception as exc:
                logger.warning(
                    "Semantic metric request failed (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(self.config.retry_delay)

        logger.error(
            "Failed to obtain semantic judgments after %d attempts; "
            "defaulting unmatched pairs to False.",
            self.config.max_retries,
        )
        return {idx: False for idx in range(len(pairs))}

    @staticmethod
    def _coerce_string_output(raw_response: Any) -> str:
        """Normalize raw LLM output to a string."""
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, Sequence):
            # Flatten simple sequence outputs by taking the first element.
            return SemanticAccuracyMetric._coerce_string_output(raw_response[0])
        return str(raw_response)

    @staticmethod
    def _parse_response(raw_response: str, expected: int) -> Dict[int, bool]:
        """Extract boolean judgments from the LLM response."""
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            logger.error("Could not decode OpenAI response as JSON: %s", exc)
            return {idx: False for idx in range(expected)}

        results = payload.get("results")
        if not isinstance(results, list):
            logger.error("OpenAI response missing 'results' list: %s", payload)
            return {idx: False for idx in range(expected)}

        decisions: Dict[int, bool] = {}
        for entry in results:
            raw_index = entry.get("index")
            match = entry.get("match")

            if not isinstance(raw_index, int) or not isinstance(match, bool):
                logger.warning("Malformed entry in OpenAI response: %s", entry)
                continue

            decisions[raw_index] = match

        # Fill in missing indices with False to stay conservative.
        for idx in range(expected):
            decisions.setdefault(idx, False)

        return decisions

    @staticmethod
    def _iter_batches(
        items: Sequence[Tuple[str, str, str]], batch_size: int
    ) -> Iterable[Sequence[Tuple[str, str, str]]]:
        if batch_size <= 0:
            batch_size = len(items)

        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    @staticmethod
    def _make_cache_key(question: str, prediction: str, reference: str) -> str:
        normalized_question = normalize_string(question)
        normalized_pred = normalize_string(prediction)
        normalized_ref = normalize_string(reference)
        digest = hashlib.md5(
            f"{normalized_question}|||{normalized_pred}|||{normalized_ref}".encode(
                "utf-8"
            ),  # nosec B303
            usedforsecurity=False,
        ).hexdigest()
        return digest

    def _build_llm_caller(self) -> LLMAPICaller:
        """Instantiate the shared LLM caller for semantic evaluation."""
        return LLMAPICaller(
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            use_custom_api=self.config.use_custom_api,
            global_cfg=self._global_cfg,
        )

    def _sanitize_global_cfg(self, cfg: Optional[Any]) -> SimpleNamespace:
        """Construct a minimal global config required by the LLM caller."""
        project_path = self._extract_cfg_value(cfg, "project_path", str(Path.cwd()))
        redis_cfg = self._extract_cfg_value(cfg, "redis", None)

        if redis_cfg is None:
            redis_ns = SimpleNamespace(host="localhost", port=6379, db=0)
        else:
            redis_ns = SimpleNamespace(
                host=self._extract_cfg_value(redis_cfg, "host", "localhost"),
                port=self._extract_cfg_value(redis_cfg, "port", 6379),
                db=self._extract_cfg_value(redis_cfg, "db", 0),
            )

        return SimpleNamespace(project_path=project_path, redis=redis_ns)

    def _extract_cfg_value(self, cfg: Optional[Any], key: str, default: Any) -> Any:
        """Helper to extract a configuration value from various container types."""
        if cfg is None:
            return default

        if isinstance(cfg, DictConfig):
            return cfg.get(key, default)

        if isinstance(cfg, dict):
            return cfg.get(key, default)

        if hasattr(cfg, "get"):
            try:
                value = cfg.get(key)
            except Exception:
                value = default
            if value is not None:
                return value

        value = getattr(cfg, key, default)
        return default if value is None else value
