"""Round-trip verifier for verifying data integrity through reconstruction.

This module provides functionality to verify that data stored in the database
can be faithfully reconstructed into the original QA pair content.

Supports both synchronous and async parallel processing modes for improved
performance while respecting API rate limits.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import litellm
from omegaconf import DictConfig
from tqdm import tqdm

from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry, TableSchema
from src.llm import JSONRepairer, LLMAPICaller, TooMuchThinkingError
from src.prompt.registry import (
    ROUND_TRIP_COMPARISON_PROMPT_REGISTRY,
    ROUND_TRIP_LOOKUP_PROMPT_REGISTRY,
    ROUND_TRIP_ANSWER_PROMPT_REGISTRY,
    ROUND_TRIP_JUDGMENT_PROMPT_REGISTRY,
    FIX_GENERATION_PROMPT_REGISTRY,
)
from src.utils.async_utils import AsyncRateLimiter
from src.utils.json_utils import (
    extract_json_from_response,
    fix_js_string_concat,
    safe_json_loads,
)
from src.utils.results_saver import IntermediateResultsSaver

logger = logging.getLogger("RoundTripVerifier")

# Natural keys for UPSERT conflict resolution
# Maps entity/table names to the column used for ON CONFLICT clause
DEFAULT_CONFLICT_KEYS: Dict[str, str] = {
    "person": "name",
    "work": "title",
    "award": "award_name",
    "character": "name",
    "location": "name",
    "genre": "name",
    "occupation": "name",
    "nationality": "nationality",
    "language": "language",
    "series": "series",
    "theme": "theme",
    "birth_date": "birth_date",
    "date": "birth_date",
    "age": "age",
    "autobiography": "has_autobiography",
    "document": "document_type",
    "employment_status": "employment_status",
    "event": "event_type",
    "field": "field",
    "gender_identity": "gender_identity",
    "mythology": "mythology_name",
    "profession": "occupation_name",
}


@dataclass
class VerificationResult:
    """Result of round-trip verification for a single QA pair.

    Attributes:
        qa_index: Index of the QA pair in the batch
        original_qa: The original (question, answer) tuple
        reconstructed_text: Text reconstructed from database
        similarity_score: Semantic similarity score (0.0 to 1.0)
        missing_facts: List of facts missing from reconstruction
        extra_facts: List of facts added in reconstruction
        needs_fix: Whether this QA pair needs data fixes
        suggested_fixes: List of suggested database fixes
        has_qa_inconsistency: Whether the QA pair has native inconsistency
        inconsistency_details: Details about the QA inconsistency if present
    """

    qa_index: int
    original_qa: Tuple[str, str]
    reconstructed_text: str
    similarity_score: float = 1.0
    missing_facts: List[Dict[str, str]] = field(default_factory=list)
    extra_facts: List[Dict[str, str]] = field(default_factory=list)
    needs_fix: bool = False
    suggested_fixes: List[Dict[str, Any]] = field(default_factory=list)
    has_qa_inconsistency: bool = False
    inconsistency_details: str = ""


@dataclass
class VerificationIntermediateData:
    """Intermediate data from each verification step for debugging.

    This dataclass captures all intermediate results during round-trip
    verification to help diagnose where the pipeline is failing.

    Attributes:
        qa_index: Index of the QA pair being verified
        question: Original question
        answer: Original answer
        lookup_plan: Generated lookup plan (tables and conditions)
        lookup_plan_raw_response: Raw LLM response for lookup plan
        retrieved_records: Records fetched from database per table
        records_as_text: Records formatted as text for LLM
        synthesized_answer: LLM-synthesized answer from records
        comparison_result: Raw comparison result from LLM
        final_score: Final similarity score
        needs_fix: Whether fixes are needed
        error: Any error encountered during verification
    """

    qa_index: int
    question: str
    answer: str
    lookup_plan: List[Dict[str, Any]] = field(default_factory=list)
    lookup_plan_raw_response: str = ""
    retrieved_records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    records_as_text: str = ""
    synthesized_answer: str = ""
    comparison_result: Dict[str, Any] = field(default_factory=dict)
    final_score: float = 0.0
    needs_fix: bool = False
    error: Optional[str] = None
    has_qa_inconsistency: bool = False
    inconsistency_details: str = ""


@dataclass(frozen=True)
class TableLookupMetadata:
    """Cached lookup metadata for a table during one verifier run."""

    columns: FrozenSet[str]
    primary_key: Optional[str] = None
    foreign_key_columns: FrozenSet[str] = field(default_factory=frozenset)
    text_columns: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class VerificationPerformanceStats:
    """Lightweight performance counters for round-trip verification."""

    lookup_plans_executed: int = 0
    lookup_plan_items: int = 0
    metadata_queries: int = 0
    metadata_cache_hits: int = 0
    table_queries: int = 0
    exact_lookup_queries: int = 0
    prefix_lookup_queries: int = 0
    substring_lookup_queries: int = 0
    fallback_queries: int = 0
    total_lookup_time_seconds: float = 0.0
    total_query_time_seconds: float = 0.0


class RoundTripVerifier:
    """Verifies data integrity through reconstruction comparison.

    This class performs round-trip verification by:
    1. Querying relevant entities from the database using LLM-guided lookup plan
    2. Synthesizing an answer from the retrieved records using LLM
    3. Comparing with original QA pair using semantic similarity
    4. Generating fixes for any discrepancies

    Attributes:
        api_cfg: LLM API configuration
        global_cfg: Global configuration
        _api_call_counter: Thread-local counter for actual (non-cached) API calls
    """

    # Thread-local storage for tracking actual API calls (not cache hits)
    _api_call_counter: threading.local = threading.local()

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig) -> None:
        """Initialize the RoundTripVerifier.

        Args:
            api_cfg: LLM API configuration
            global_cfg: Global configuration
        """
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg
        self._verification_stats: VerificationPerformanceStats = (
            VerificationPerformanceStats()
        )
        self._verification_stats_lock = threading.Lock()
        self._table_lookup_metadata_cache: Dict[str, TableLookupMetadata] = {}
        self._table_lookup_metadata_lock = threading.Lock()

    def _reset_api_call_count(self) -> None:
        """Reset the thread-local API call counter to 0."""
        self._api_call_counter.count = 0

    def _increment_api_call_count(self) -> None:
        """Increment the thread-local API call counter by 1."""
        current: int = getattr(self._api_call_counter, "count", 0)
        self._api_call_counter.count = current + 1

    def _get_api_call_count(self) -> int:
        """Get the current thread-local API call count.

        Returns:
            Number of actual (non-cached) API calls made in this thread.
        """
        return getattr(self._api_call_counter, "count", 0)

    # =========================================================================
    # Cached Properties
    # =========================================================================

    @cached_property
    def llm_api_caller(self) -> LLMAPICaller:
        """Get the primary LLM API caller."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.base,
        )

    @cached_property
    def fallback_llm_api_caller(self) -> LLMAPICaller:
        """Get the fallback LLM API caller for complex tasks."""
        return LLMAPICaller(
            global_cfg=self.global_cfg,
            **self.api_cfg.smart,
        )

    @cached_property
    def round_trip_comparison_prompt(self):
        """Get the round-trip comparison prompt class."""
        prompt_name = self.global_cfg.prompt.get("round_trip_comparison", "default")
        return ROUND_TRIP_COMPARISON_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def round_trip_lookup_prompt(self):
        """Get the round-trip lookup prompt class."""
        prompt_name = self.global_cfg.prompt.get("round_trip_lookup", "default")
        return ROUND_TRIP_LOOKUP_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def round_trip_answer_prompt(self):
        """Get the round-trip answer prompt class."""
        prompt_name = self.global_cfg.prompt.get("round_trip_answer", "default")
        return ROUND_TRIP_ANSWER_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def round_trip_judgment_prompt(self):
        """Get the round-trip judgment prompt class."""
        prompt_name = self.global_cfg.prompt.get("round_trip_judgment", "default")
        return ROUND_TRIP_JUDGMENT_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def fix_generation_prompt(self):
        """Get the fix generation prompt class."""
        prompt_name = self.global_cfg.prompt.get("fix_generation", "default")
        return FIX_GENERATION_PROMPT_REGISTRY[prompt_name]

    @cached_property
    def enable_round_trip_verification(self) -> bool:
        """Whether round-trip verification is enabled."""
        return self.global_cfg.model.aligned_db.get(
            "enable_round_trip_verification", True
        )

    @cached_property
    def json_repairer(self) -> JSONRepairer:
        """Get the JSON repairer for fixing malformed JSON responses."""
        return JSONRepairer(self.api_cfg, self.global_cfg)

    @cached_property
    def similarity_threshold(self) -> float:
        """Similarity threshold below which fixes are needed."""
        # Lowered from 0.85 to 0.6 - 0.6+ means core facts are present
        return self.global_cfg.model.aligned_db.get("similarity_threshold", 0.6)

    @cached_property
    def enable_async_round_trip(self) -> bool:
        """Whether async parallel processing is enabled for round-trip verification."""
        return self.global_cfg.model.aligned_db.get("enable_async_round_trip", True)

    @cached_property
    def max_concurrency(self) -> int:
        """Maximum concurrent LLM calls for async round-trip verification."""
        return self.global_cfg.model.aligned_db.get("round_trip_max_concurrency", 5)

    @cached_property
    def requests_per_second(self) -> float:
        """Rate limit for LLM requests per second."""
        return self.global_cfg.model.aligned_db.get(
            "round_trip_requests_per_second", 5.0
        )

    @cached_property
    def enable_iterative_verification(self) -> bool:
        """Whether iterative verification loop is enabled."""
        return self.global_cfg.model.aligned_db.get(
            "enable_iterative_verification", True
        )

    @cached_property
    def verification_max_iterations(self) -> int:
        """Maximum number of fix iterations."""
        return self.global_cfg.model.aligned_db.get("verification_max_iterations", 3)

    @cached_property
    def enable_fix_validation(self) -> bool:
        """Whether schema-aware fix validation is enabled."""
        return self.global_cfg.model.aligned_db.get("enable_fix_validation", True)

    @cached_property
    def use_dedicated_fix_prompt(self) -> bool:
        """Whether to use dedicated LLM prompt for fix generation."""
        return self.global_cfg.model.aligned_db.get("use_dedicated_fix_prompt", True)

    @cached_property
    def enable_re_extraction_for_failures(self) -> bool:
        """Whether to re-extract entities for very low similarity scores."""
        return self.global_cfg.model.aligned_db.get(
            "enable_re_extraction_for_failures", True
        )

    @cached_property
    def re_extraction_similarity_threshold(self) -> float:
        """Similarity threshold below which re-extraction is triggered."""
        return self.global_cfg.model.aligned_db.get(
            "re_extraction_similarity_threshold", 0.5
        )

    @cached_property
    def save_intermediate_results(self) -> bool:
        """Whether to save intermediate verification results for debugging."""
        return self.global_cfg.model.aligned_db.get("save_intermediate_results", False)

    @cached_property
    def _results_saver(self) -> IntermediateResultsSaver:
        """Get intermediate results saver for verification logs."""
        return IntermediateResultsSaver.from_config(
            self.global_cfg, "round_trip_verification"
        )

    def _reset_verification_run_state(
        self, schema_registry: Optional[SchemaRegistry] = None
    ) -> None:
        """Reset per-run caches and counters before verification starts."""
        with self._verification_stats_lock:
            self._verification_stats = VerificationPerformanceStats()
        with self._table_lookup_metadata_lock:
            self._table_lookup_metadata_cache = {}
            if schema_registry is not None:
                for table_name in schema_registry.get_table_names():
                    table = schema_registry.get_table(table_name)
                    if table is None:
                        continue
                    self._table_lookup_metadata_cache[table_name] = (
                        self._build_table_lookup_metadata_from_schema(table)
                    )

    def _clear_verification_run_state(self) -> None:
        """Clear per-run caches once verification finishes."""
        with self._verification_stats_lock:
            self._verification_stats = VerificationPerformanceStats()
        with self._table_lookup_metadata_lock:
            self._table_lookup_metadata_cache = {}

    def _record_verification_stat(self, field_name: str, amount: float = 1) -> None:
        """Atomically increment a verifier performance counter."""
        with self._verification_stats_lock:
            current = getattr(self._verification_stats, field_name)
            setattr(self._verification_stats, field_name, current + amount)

    def _snapshot_verification_stats(self) -> Dict[str, Any]:
        """Return a serializable snapshot of current verifier counters."""
        with self._verification_stats_lock:
            return asdict(self._verification_stats)

    @staticmethod
    def _build_table_lookup_metadata_from_schema(
        table: TableSchema,
    ) -> TableLookupMetadata:
        """Build cached lookup metadata from a schema-registry table definition."""
        text_like_types = {
            "TEXT",
            "VARCHAR",
            "CHAR",
            "CHARACTER",
            "CHARACTER VARYING",
            "CITEXT",
        }
        text_columns = frozenset(
            col.name
            for col in table.columns
            if col.data_type.upper() in text_like_types
        )
        return TableLookupMetadata(
            columns=frozenset(table.get_column_names()),
            primary_key=table.get_primary_key(),
            foreign_key_columns=frozenset(fk.column_name for fk in table.foreign_keys),
            text_columns=text_columns,
        )

    def _get_table_lookup_metadata(
        self,
        table_name: str,
        pg_client,
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> Optional[TableLookupMetadata]:
        """Get cached lookup metadata for a table, querying once on cache miss."""
        with self._table_lookup_metadata_lock:
            cached = self._table_lookup_metadata_cache.get(table_name)
        if cached is not None:
            self._record_verification_stat("metadata_cache_hits")
            return cached

        if schema_registry is not None:
            table = schema_registry.get_table(table_name)
            if table is not None:
                metadata = self._build_table_lookup_metadata_from_schema(table)
                with self._table_lookup_metadata_lock:
                    self._table_lookup_metadata_cache[table_name] = metadata
                return metadata

        try:
            self._record_verification_stat("metadata_queries")
            with pg_client.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table_name,),
                )
                rows = cursor.fetchall()
        except Exception as e:
            logger.debug(f"Failed to get columns for {table_name}: {e}")
            return None

        if not rows:
            return None

        columns = frozenset(row[0] for row in rows)
        text_columns = frozenset(
            row[0]
            for row in rows
            if row[1].upper()
            in {"TEXT", "VARCHAR", "CHAR", "CHARACTER", "CHARACTER VARYING", "CITEXT"}
        )
        metadata = TableLookupMetadata(
            columns=columns,
            primary_key=f"{table_name}_id" if f"{table_name}_id" in columns else None,
            foreign_key_columns=frozenset(col for col in columns if col.endswith("_id")),
            text_columns=text_columns,
        )
        with self._table_lookup_metadata_lock:
            self._table_lookup_metadata_cache[table_name] = metadata
        return metadata

    @staticmethod
    def _normalize_lookup_string(value: str) -> str:
        """Normalize free-text lookup values for exact/prefix matching."""
        return " ".join(str(value).split()).strip().lower()

    def _build_string_lookup_modes(self, value: str) -> List[Tuple[str, str]]:
        """Return ordered string lookup modes from strict to permissive."""
        normalized = self._normalize_lookup_string(value)
        if not normalized:
            return []
        return [
            ("exact", normalized),
            ("prefix", f"{normalized}%"),
            ("substring", f"%{normalized}%"),
        ]

    def _build_string_where_clause(self, column: str, mode: str) -> str:
        """Build a SQL predicate for the requested text matching mode."""
        normalized_column = f"LOWER(TRIM({column}::text))"
        if mode == "exact":
            return f"{normalized_column} = %s"
        return f"{normalized_column} LIKE %s"

    def _fetch_rows(
        self,
        pg_client,
        query: str,
        params: Optional[List[Any]] = None,
        *,
        is_fallback: bool = False,
        match_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a query and record lightweight verifier performance stats."""
        self._record_verification_stat("table_queries")
        if is_fallback:
            self._record_verification_stat("fallback_queries")
        if match_mode == "exact":
            self._record_verification_stat("exact_lookup_queries")
        elif match_mode == "prefix":
            self._record_verification_stat("prefix_lookup_queries")
        elif match_mode == "substring":
            self._record_verification_stat("substring_lookup_queries")

        start = time.perf_counter()
        try:
            with pg_client.conn.cursor() as cursor:
                cursor.execute(query, params or [])
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                rows = cursor.fetchall()
                if rows and columns:
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.debug(f"Query error: {e}")
        finally:
            self._record_verification_stat(
                "total_query_time_seconds", time.perf_counter() - start
            )

        return []

    def _save_verification_intermediate(
        self, intermediate_data: VerificationIntermediateData
    ) -> None:
        """Save intermediate verification data for debugging.

        Saves detailed information about each verification step to help diagnose
        failures in the round-trip verification pipeline.

        Args:
            intermediate_data: The intermediate data from verification
        """
        if not self.save_intermediate_results:
            return

        # Convert dataclass to dict for JSON serialization
        data: Dict[str, Any] = asdict(intermediate_data)

        # Save to a per-QA-pair file
        self._results_saver.save_item(
            idx=intermediate_data.qa_index,
            data=data,
            file_prefix="verification",
        )

        # Log summary for failed verifications (score < threshold)
        if intermediate_data.needs_fix:
            logger.debug(
                f"[QA {intermediate_data.qa_index}] Verification details saved "
                f"(score={intermediate_data.final_score:.2f})"
            )

    @staticmethod
    def _compute_score_distribution(
        results: List[VerificationResult],
    ) -> Dict[str, int]:
        """Compute the distribution of similarity scores.

        Args:
            results: List of verification results

        Returns:
            Dict mapping score ranges to counts
        """
        return {
            "perfect (1.0)": sum(1 for r in results if r.similarity_score == 1.0),
            "high (0.85-0.99)": sum(
                1 for r in results if 0.85 <= r.similarity_score < 1.0
            ),
            "medium (0.5-0.84)": sum(
                1 for r in results if 0.5 <= r.similarity_score < 0.85
            ),
            "low (0.0-0.49)": sum(1 for r in results if 0.0 < r.similarity_score < 0.5),
            "zero (0.0)": sum(1 for r in results if r.similarity_score == 0.0),
        }

    @staticmethod
    def _build_failed_qa_details(
        results: List[VerificationResult],
    ) -> List[Dict[str, Any]]:
        """Build details for failed QA pairs.

        Args:
            results: List of verification results

        Returns:
            List of dicts with failed QA details
        """
        details = []
        for r in results:
            if r.needs_fix:
                detail = {
                    "qa_index": r.qa_index,
                    "score": r.similarity_score,
                    "missing_facts_count": len(r.missing_facts),
                    "question_preview": (
                        r.original_qa[0][:100] + "..."
                        if len(r.original_qa[0]) > 100
                        else r.original_qa[0]
                    ),
                    "has_qa_inconsistency": r.has_qa_inconsistency,
                }
                if r.has_qa_inconsistency:
                    detail["inconsistency_details"] = r.inconsistency_details
                details.append(detail)
        return details

    def _analyze_zero_score_failures(
        self,
        results: List[VerificationResult],
    ) -> Dict[str, Any]:
        """Analyze patterns in zero-score failures to identify systemic issues.

        Zero-score failures (similarity_score == 0.0) indicate complete extraction
        misses where no relevant data was captured for the QA pair. This analysis
        helps identify common patterns that can guide pipeline improvements.

        Args:
            results: List of all verification results

        Returns:
            Analysis dict with patterns and recommendations
        """
        zero_scores = [r for r in results if r.similarity_score == 0.0]

        if not zero_scores:
            return {"count": 0, "message": "No zero-score failures"}

        analysis: Dict[str, Any] = {
            "count": len(zero_scores),
            "question_patterns": {},
            "missing_entity_types": {},
            "common_missing_facts": {},
            "sample_failures": [],
        }

        # Categorize questions by pattern
        question_categories = {
            "character_questions": 0,
            "award_questions": 0,
            "date_questions": 0,
            "location_questions": 0,
            "relationship_questions": 0,
            "work_detail_questions": 0,
            "other": 0,
        }

        character_keywords = ["character", "protagonist", "antagonist", "hero", "appears in"]
        award_keywords = ["award", "prize", "honor", "won", "received", "nominated"]
        date_keywords = ["when", "date", "year", "born", "died", "published"]
        location_keywords = ["where", "place", "city", "country", "born in", "from"]
        relationship_keywords = ["father", "mother", "parent", "child", "married", "spouse"]
        work_keywords = ["book", "novel", "wrote", "author", "title", "theme"]

        for r in zero_scores:
            q_lower = r.original_qa[0].lower()

            if any(kw in q_lower for kw in character_keywords):
                question_categories["character_questions"] += 1
            elif any(kw in q_lower for kw in award_keywords):
                question_categories["award_questions"] += 1
            elif any(kw in q_lower for kw in date_keywords):
                question_categories["date_questions"] += 1
            elif any(kw in q_lower for kw in location_keywords):
                question_categories["location_questions"] += 1
            elif any(kw in q_lower for kw in relationship_keywords):
                question_categories["relationship_questions"] += 1
            elif any(kw in q_lower for kw in work_keywords):
                question_categories["work_detail_questions"] += 1
            else:
                question_categories["other"] += 1

        analysis["question_patterns"] = {
            k: v for k, v in question_categories.items() if v > 0
        }

        # Analyze missing facts patterns
        missing_fact_types: Dict[str, int] = {}
        for r in zero_scores:
            for fact in r.missing_facts[:5]:  # Limit to first 5 per QA
                # Handle both string and dict facts
                if isinstance(fact, dict):
                    fact_str = str(fact.get("fact", fact.get("description", str(fact))))
                else:
                    fact_str = str(fact)
                fact_lower = fact_str.lower()
                
                # Categorize missing facts
                if "character" in fact_lower or "protagonist" in fact_lower:
                    missing_fact_types["character_info"] = missing_fact_types.get("character_info", 0) + 1
                elif "award" in fact_lower or "prize" in fact_lower:
                    missing_fact_types["award_info"] = missing_fact_types.get("award_info", 0) + 1
                elif "date" in fact_lower or "year" in fact_lower or "birth" in fact_lower:
                    missing_fact_types["date_info"] = missing_fact_types.get("date_info", 0) + 1
                elif "father" in fact_lower or "mother" in fact_lower or "parent" in fact_lower:
                    missing_fact_types["parent_info"] = missing_fact_types.get("parent_info", 0) + 1
                elif "theme" in fact_lower or "genre" in fact_lower:
                    missing_fact_types["work_details"] = missing_fact_types.get("work_details", 0) + 1
                else:
                    missing_fact_types["other"] = missing_fact_types.get("other", 0) + 1

        analysis["common_missing_facts"] = missing_fact_types

        # Add sample failures for debugging
        for r in zero_scores[:5]:  # First 5 samples
            analysis["sample_failures"].append({
                "qa_index": r.qa_index,
                "question_preview": r.original_qa[0][:150],
                "missing_facts_sample": r.missing_facts[:3],
            })

        # Log warning with summary
        logger.warning(
            f"\n=== Zero-Score Failure Analysis ===\n"
            f"  Count: {len(zero_scores)} QA pairs with score 0.0\n"
            f"  Question patterns: {analysis['question_patterns']}\n"
            f"  Missing fact types: {analysis['common_missing_facts']}\n"
            f"  Recommendation: Focus on extracting {self._get_top_missing_category(analysis)}"
        )

        # Save detailed analysis if intermediate results are enabled
        if self.save_intermediate_results:
            self._results_saver.save(
                sub_dir_name="analysis",
                data=analysis,
                file_prefix="zero_score_analysis",
            )

        return analysis

    def _get_top_missing_category(self, analysis: Dict[str, Any]) -> str:
        """Get the most common category of missing data from analysis.

        Args:
            analysis: Zero-score analysis results

        Returns:
            Recommendation string for the top missing category
        """
        patterns = analysis.get("question_patterns", {})
        if not patterns:
            return "general entity extraction"

        top_pattern = max(patterns.items(), key=lambda x: x[1])
        recommendations = {
            "character_questions": "character entities and work_character relations",
            "award_questions": "award entities and person_award relations",
            "date_questions": "date attributes (birth_date, publication_date, etc.)",
            "location_questions": "location entities and birth_place attributes",
            "relationship_questions": "parent info (father_name, mother_name, etc.)",
            "work_detail_questions": "work attributes (themes, genres, influences)",
        }
        return recommendations.get(top_pattern[0], "comprehensive entity extraction")

    def _save_verification_summary(
        self,
        results: List[VerificationResult],
        iteration: int = 0,
    ) -> None:
        """Save a summary of all verification results.

        Args:
            results: List of VerificationResult objects
            iteration: Verification iteration number (0 = initial)
        """
        if not self.save_intermediate_results:
            return

        total: int = len(results)
        needs_fix_count: int = sum(1 for r in results if r.needs_fix)
        qa_inconsistency_count: int = sum(
            1 for r in results if r.has_qa_inconsistency
        )
        passed_count: int = total - needs_fix_count - qa_inconsistency_count
        avg_score: float = (
            sum(r.similarity_score for r in results) / total if total > 0 else 0.0
        )
        performance_stats = self._snapshot_verification_stats()

        # Analyze zero-score failures for insights
        zero_score_analysis = self._analyze_zero_score_failures(results)

        self._results_saver.save(
            sub_dir_name="summaries",
            data={
                "iteration": iteration,
                "total_qa_pairs": total,
                "passed_count": passed_count,
                "needs_fix_count": needs_fix_count,
                "qa_inconsistency_count": qa_inconsistency_count,
                "pass_rate": passed_count / total if total > 0 else 0.0,
                "average_score": avg_score,
                "score_distribution": self._compute_score_distribution(results),
                "performance": performance_stats,
                "zero_score_analysis": zero_score_analysis,
                "failed_qa_details": self._build_failed_qa_details(results),
            },
            suffix=f"iteration_{iteration}",
            file_prefix="summary",
        )

        logger.info(
            f"Verification summary saved (iteration {iteration}): "
            f"{passed_count}/{total} passed, {needs_fix_count} need fixes, "
            f"{qa_inconsistency_count} inconsistent QA, avg score={avg_score:.3f}"
        )
        logger.info(
            "Round-trip verification performance: "
            f"plans={performance_stats['lookup_plans_executed']}, "
            f"plan_items={performance_stats['lookup_plan_items']}, "
            f"metadata_queries={performance_stats['metadata_queries']}, "
            f"metadata_cache_hits={performance_stats['metadata_cache_hits']}, "
            f"table_queries={performance_stats['table_queries']}, "
            f"substring_queries={performance_stats['substring_lookup_queries']}, "
            f"lookup_time={performance_stats['total_lookup_time_seconds']:.3f}s, "
            f"query_time={performance_stats['total_query_time_seconds']:.3f}s"
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def verify_all(
        self,
        qa_pairs: List[Tuple[str, str]],
        pg_client,
        schema_registry: SchemaRegistry,
    ) -> List[VerificationResult]:
        """Verify all QA pairs against database content.

        For each QA pair:
        1. Generate lookup plan
        2. Query database records
        3. Synthesize answer
        4. Compare with original using semantic similarity
        5. Return discrepancies

        Uses async parallel processing when enabled for improved performance.

        Args:
            qa_pairs: List of original (question, answer) tuples
            pg_client: PostgreSQL client for querying data
            schema_registry: Current database schema

        Returns:
            List of VerificationResult objects
        """
        if not self.enable_round_trip_verification:
            logger.info("Round-trip verification disabled")
            return []

        logger.info(f"Verifying {len(qa_pairs)} QA pairs via intelligent round-trip")
        self._reset_verification_run_state(schema_registry)
        try:
            # Use async parallel processing if enabled
            if self.enable_async_round_trip and len(qa_pairs) > 1:
                logger.info(
                    f"Using async parallel processing (concurrency={self.max_concurrency}, "
                    f"rate_limit={self.requests_per_second} RPS, cache-aware)"
                )
                return asyncio.run(
                    self._verify_all_async(qa_pairs, pg_client, schema_registry)
                )

            # Fallback to sequential processing
            return self._verify_all_sequential(qa_pairs, pg_client, schema_registry)
        finally:
            self._clear_verification_run_state()

    def _verify_all_sequential(
        self,
        qa_pairs: List[Tuple[str, str]],
        pg_client,
        schema_registry: SchemaRegistry,
    ) -> List[VerificationResult]:
        """Verify all QA pairs sequentially (original implementation).

        Args:
            qa_pairs: List of original (question, answer) tuples
            pg_client: PostgreSQL client for querying data
            schema_registry: Current database schema

        Returns:
            List of VerificationResult objects
        """
        results: List[VerificationResult] = []
        schema_sql = schema_registry.to_sql_list()

        for idx, (question, answer) in enumerate(qa_pairs):
            # Initialize intermediate data for logging
            intermediate_data = VerificationIntermediateData(
                qa_index=idx,
                question=question,
                answer=answer,
            )

            try:
                # Step 1-2: Generate lookup plan and execute queries
                records_text, intermediate_data = (
                    self._reconstruct_from_db_with_logging(
                        question, answer, pg_client, schema_registry, intermediate_data
                    )
                )

                # Step 3: Judge record sufficiency (replaces synthesis + comparison)
                judgment_result, intermediate_data = self._judge_with_logging(
                    question, answer, records_text, schema_sql, idx, intermediate_data
                )

                # Step 4: For failed QA pairs, check for native QA inconsistency
                if judgment_result.needs_fix:
                    has_inconsistency, details = self._check_qa_inconsistency(
                        question, answer
                    )
                    if has_inconsistency:
                        judgment_result.has_qa_inconsistency = True
                        judgment_result.inconsistency_details = details
                        judgment_result.needs_fix = False  # Exclude from fix pipeline
                        intermediate_data.has_qa_inconsistency = True
                        intermediate_data.inconsistency_details = details
                        logger.info(
                            f"QA pair {idx}: Native QA inconsistency detected - {details[:100]}..."
                        )

                results.append(judgment_result)

                if judgment_result.needs_fix:
                    logger.warning(
                        f"QA pair {idx} needs fix (score: {judgment_result.similarity_score:.2f})"
                    )

            except Exception as e:
                # Log error and create failed result
                intermediate_data.error = str(e)
                logger.error(f"QA pair {idx} verification failed with error: {e}")
                results.append(
                    VerificationResult(
                        qa_index=idx,
                        original_qa=(question, answer),
                        reconstructed_text="ERROR",
                        similarity_score=0.0,
                        needs_fix=True,
                    )
                )

            # Save intermediate data for debugging
            self._save_verification_intermediate(intermediate_data)

        # Summary
        needs_fix_count = sum(1 for r in results if r.needs_fix)
        inconsistent_count = sum(1 for r in results if r.has_qa_inconsistency)
        passed_count = len(results) - needs_fix_count - inconsistent_count
        logger.info(
            f"Verification complete: {passed_count}/{len(results)} passed, "
            f"{needs_fix_count} need fixes, {inconsistent_count} inconsistent QA"
        )

        # Save verification summary
        self._save_verification_summary(results, iteration=0)

        return results

    async def _verify_all_async(
        self,
        qa_pairs: List[Tuple[str, str]],
        pg_client,
        schema_registry: SchemaRegistry,
        max_retries: int = 3,
    ) -> List[VerificationResult]:
        """Verify all QA pairs using async parallel processing with retry logic.

        Uses a semaphore to limit concurrency and a rate limiter to respect
        API rate limits. Failed verifications are retried with exponential backoff.

        Args:
            qa_pairs: List of original (question, answer) tuples
            pg_client: PostgreSQL client for querying data
            schema_registry: Current database schema
            max_retries: Maximum number of retry attempts per QA pair (default: 3)

        Returns:
            List of VerificationResult objects (in original order)
        """
        schema_sql = schema_registry.to_sql_list()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Create rate limiter
        rate_limiter = AsyncRateLimiter(rate=self.requests_per_second)

        # Create thread pool for running sync LLM calls
        executor = ThreadPoolExecutor(max_workers=self.max_concurrency)

        # Track retry statistics (thread-safe via asyncio single-threaded event loop)
        retry_stats = {"total_retries": 0, "failed_after_retries": 0}

        def reconstruct_with_call_count(question: str, answer: str) -> Tuple[str, int]:
            """Wrapper that returns records result with actual API call count.

            This runs in a thread pool and tracks non-cached API calls for rate limiting.

            Args:
                question: Original question
                answer: Original answer

            Returns:
                Tuple of (records_text, actual_api_call_count)
            """
            self._reset_api_call_count()
            result = self._reconstruct_from_db(
                question, answer, pg_client, schema_registry
            )
            return result, self._get_api_call_count()

        def judge_with_call_count(
            question: str,
            answer: str,
            records_text: str,
            schema: List[str],
            qa_idx: int,
        ) -> Tuple[VerificationResult, int]:
            """Wrapper that returns judgment result with actual API call count.

            This runs in a thread pool and tracks non-cached API calls for rate limiting.

            Args:
                question: Original question
                answer: Original answer
                records_text: Text representation of database records
                schema: Schema SQL statements
                qa_idx: Index of QA pair

            Returns:
                Tuple of (VerificationResult, actual_api_call_count)
            """
            self._reset_api_call_count()
            result = self._judge(question, answer, records_text, schema, qa_idx)
            return result, self._get_api_call_count()

        async def verify_single_async(
            idx: int, question: str, answer: str
        ) -> VerificationResult:
            """Verify a single QA pair asynchronously with retry on failure.

            Rate limiting is applied only for actual API calls, not cached results.
            """
            last_error: Optional[Exception] = None
            should_retry: bool = False

            for attempt in range(max_retries + 1):
                # Wait for backoff BEFORE acquiring semaphore (not while holding it)
                if should_retry:
                    backoff_time = 1.0 * attempt  # 1s, 2s, 3s for attempts 1, 2, 3
                    await asyncio.sleep(backoff_time)
                    should_retry = False

                async with semaphore:
                    try:
                        loop = asyncio.get_running_loop()

                        # Reserve one token before dispatch so rate limiting is preventive,
                        # not purely reactive after the outbound LLM work has already run.
                        await rate_limiter.acquire()

                        # Run lookup in thread pool (tracks actual API calls)
                        records_text, reconstruct_api_calls = (
                            await loop.run_in_executor(
                                executor,
                                reconstruct_with_call_count,
                                question,
                                answer,
                            )
                        )

                        # Acquire only any remaining tokens beyond the pre-reserved one.
                        await rate_limiter.acquire(max(reconstruct_api_calls - 1, 0))

                        await rate_limiter.acquire()

                        # Run judgment in thread pool (tracks actual API calls)
                        judgment_result, judge_api_calls = await loop.run_in_executor(
                            executor,
                            judge_with_call_count,
                            question,
                            answer,
                            records_text,
                            schema_sql,
                            idx,
                        )

                        # Acquire only any remaining tokens beyond the pre-reserved one.
                        await rate_limiter.acquire(max(judge_api_calls - 1, 0))

                        # For failed QA pairs, check for native QA inconsistency
                        if judgment_result.needs_fix:
                            has_inconsistency, details = await loop.run_in_executor(
                                executor,
                                self._check_qa_inconsistency,
                                question,
                                answer,
                            )
                            if has_inconsistency:
                                judgment_result.has_qa_inconsistency = True
                                judgment_result.inconsistency_details = details
                                judgment_result.needs_fix = False  # Exclude from fix pipeline
                                logger.info(
                                    f"QA inconsistency detected: {details[:150]}..."
                                )

                        # Success - return immediately (progress shown via tqdm)
                        return judgment_result

                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            retry_stats["total_retries"] += 1
                            logger.warning(
                                f"QA pair {idx} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                f"Retrying in {attempt + 1}s..."
                            )
                            # Signal retry - sleep happens OUTSIDE semaphore on next iteration
                            should_retry = True
                            continue

            # All retries exhausted
            retry_stats["failed_after_retries"] += 1
            logger.error(
                f"QA pair {idx} failed after {max_retries + 1} attempts: {last_error}"
            )
            return VerificationResult(
                qa_index=idx,
                original_qa=(question, answer),
                reconstructed_text=f"ERROR: {last_error}",
                similarity_score=0.0,
                needs_fix=True,
            )

        # Create tasks for all QA pairs (wrap in indexed task for result ordering)
        async def indexed_task(
            idx: int, q: str, a: str
        ) -> Tuple[int, VerificationResult]:
            result = await verify_single_async(idx, q, a)
            return idx, result

        tasks = [
            asyncio.create_task(indexed_task(idx, question, answer))
            for idx, (question, answer) in enumerate(qa_pairs)
        ]

        # Track live stats for progress bar
        live_stats = {"passed": 0, "needs_fix": 0, "errors": 0, "inconsistent": 0}
        results_dict: Dict[int, VerificationResult] = {}

        # Run all tasks with live progress updates
        with tqdm(
            total=len(tasks),
            desc="Verifying QA pairs",
            unit="qa",
            ncols=120,
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results_dict[idx] = result

                # Update live stats
                if result.reconstructed_text.startswith("ERROR:"):
                    live_stats["errors"] += 1
                elif result.has_qa_inconsistency:
                    live_stats["inconsistent"] += 1
                elif result.needs_fix:
                    live_stats["needs_fix"] += 1
                else:
                    live_stats["passed"] += 1

                # Update progress bar with live stats
                pbar.set_postfix(
                    passed=live_stats["passed"],
                    fix=live_stats["needs_fix"],
                    incon=live_stats["inconsistent"],
                    err=live_stats["errors"],
                )
                pbar.update(1)

        # Reconstruct results in original order
        results = [results_dict[i] for i in range(len(qa_pairs))]

        # Shutdown thread pool
        executor.shutdown(wait=False)

        # Final summary
        logger.info(
            f"Verification complete: {live_stats['passed']}/{len(results)} passed, "
            f"{live_stats['needs_fix']} need fixes, "
            f"{live_stats['inconsistent']} inconsistent QA, {live_stats['errors']} errors"
        )

        # Save verification summary (async mode saves summary only, not per-QA details)
        # For detailed per-QA intermediate results, use sequential mode (disable async)
        self._save_verification_summary(list(results), iteration=0)

        return list(results)

    def generate_fixes(
        self,
        discrepancies: List[VerificationResult],
        schema_registry: SchemaRegistry,
        pg_client: Optional[Any] = None,
    ) -> List[str]:
        """Generate UPDATE/INSERT statements to fix missing data.

        Validates fixes against schema before generating SQL if validation is enabled.
        If pg_client is provided, resolves string FK values to integer IDs before validation.

        Args:
            discrepancies: List of VerificationResult objects with needs_fix=True
            schema_registry: Current database schema
            pg_client: Optional PostgreSQL client for FK value resolution

        Returns:
            List of SQL statements to fix the data
        """
        fixes: List[str] = []
        skipped_invalid: int = 0
        skipped_generation: int = 0

        for result in discrepancies:
            if not result.needs_fix:
                continue

            for fix in result.suggested_fixes:
                # Resolve FK string values to IDs if pg_client is available
                if pg_client is not None:
                    fix = self._resolve_fk_values(fix, schema_registry, pg_client)

                # Validate fix if validation is enabled
                if self.enable_fix_validation:
                    is_valid, error_msg = self._validate_fix(fix, schema_registry)
                    if not is_valid:
                        logger.warning(
                            f"Skipping invalid fix for QA {result.qa_index}: {error_msg}"
                        )
                        skipped_invalid += 1
                        continue

                sql = self._generate_fix_sql(fix, schema_registry)
                if sql:
                    fixes.append(sql)
                else:
                    skipped_generation += 1

        logger.info(
            f"Generated {len(fixes)} fix statements "
            f"(skipped: {skipped_invalid} invalid, {skipped_generation} generation failed)"
        )
        return fixes

    def _get_relevant_tables_for_fix(
        self,
        result: VerificationResult,
        schema_registry: SchemaRegistry,
    ) -> Set[str]:
        """Extract relevant table names for a verification result.

        Uses suggested_fixes from the comparison step to identify which tables
        are likely needed for generating fixes. Also includes related tables
        via foreign key relationships.

        Args:
            result: VerificationResult containing suggested_fixes and missing_facts
            schema_registry: Schema registry for FK relationship lookup

        Returns:
            Set of relevant table names (or all tables if none identified)
        """
        relevant: Set[str] = set()
        all_tables: Set[str] = set(schema_registry.get_table_names())

        # Priority 1: Extract entity_type from suggested_fixes
        for fix in result.suggested_fixes:
            entity_type = fix.get("entity_type", "")
            if entity_type and entity_type in all_tables:
                relevant.add(entity_type)

        # Priority 2: Look for table names mentioned in missing_facts
        for fact in result.missing_facts:
            fact_text: str = fact.get("fact", "").lower()
            for table_name in all_tables:
                # Check if table name or its singular form appears in the fact
                if table_name.lower() in fact_text:
                    relevant.add(table_name)
                # Also check without trailing 's' (crude singularization)
                if table_name.endswith("s") and table_name[:-1].lower() in fact_text:
                    relevant.add(table_name)

        # Priority 3: Add related tables via FK relationships
        expanded: Set[str] = set(relevant)
        for table_name in relevant:
            table = schema_registry.get_table(table_name)
            if table:
                # Add referenced tables (tables this table points to)
                for fk in table.foreign_keys:
                    if fk.references_table in all_tables:
                        expanded.add(fk.references_table)

        # Fallback: If no relevant tables found, return a core subset
        if not expanded:
            # Return common core tables instead of all tables
            core_tables: List[str] = [
                "person",
                "work",
                "location",
                "award",
                "occupation",
            ]
            for t in core_tables:
                if t in all_tables:
                    expanded.add(t)

        # Safety: If still empty, return all tables
        if not expanded:
            return all_tables

        logger.debug(
            f"Schema filtering: {len(expanded)}/{len(all_tables)} tables "
            f"for QA {result.qa_index}"
        )
        return expanded

    def _filter_schema_for_tables(
        self,
        schema_registry: SchemaRegistry,
        relevant_tables: Set[str],
    ) -> List[str]:
        """Filter schema SQL statements to only include relevant tables.

        Args:
            schema_registry: Full schema registry
            relevant_tables: Set of table names to include

        Returns:
            List of CREATE TABLE SQL statements for relevant tables only
        """
        full_schema: List[str] = schema_registry.to_sql_list()
        filtered: List[str] = []

        for sql in full_schema:
            # Extract table name from CREATE TABLE statement
            # Format: "CREATE TABLE table_name (...)"
            match = re.match(r"CREATE TABLE\s+(\w+)", sql, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                if table_name in relevant_tables:
                    filtered.append(sql)

        # If filtering produced nothing, return original (safety)
        return filtered if filtered else full_schema

    def get_relevant_tables_from_extraction(
        self,
        qa_index: int,
        qa_extractions: Optional[QAExtractionRegistry],
        schema_registry: SchemaRegistry,
    ) -> Set[str]:
        """Get relevant tables from QAExtractionRegistry (exact mapping).

        This method uses the per-QA extraction data to get the exact set of
        tables that a QA pair touches, rather than inferring from fix suggestions.

        Args:
            qa_index: Index of the QA pair
            qa_extractions: Optional QAExtractionRegistry with per-QA data
            schema_registry: Schema registry for validation

        Returns:
            Set of relevant table names
        """
        if qa_extractions is None:
            return set(schema_registry.get_table_names())

        # Get exact tables from extraction
        exact_tables: Set[str] = qa_extractions.get_relevant_tables(qa_index)

        if not exact_tables:
            # Fallback to all tables if no mapping available
            return set(schema_registry.get_table_names())

        # Expand with related tables via FK
        all_tables: Set[str] = set(schema_registry.get_table_names())
        expanded: Set[str] = set(exact_tables)

        for table_name in exact_tables:
            table: Optional[TableSchema] = schema_registry.get_table(table_name)
            if table:
                for fk in table.foreign_keys:
                    if fk.references_table in all_tables:
                        expanded.add(fk.references_table)

        return expanded

    def generate_fixes_with_qa_mapping(
        self,
        failed_results: List["VerificationResult"],
        schema_registry: SchemaRegistry,
        qa_extractions: Optional[QAExtractionRegistry] = None,
        pg_client: Optional[Any] = None,
    ) -> List[str]:
        """Generate fixes using exact QA-table mapping from extraction phase.

        This method is similar to generate_fixes_with_dedicated_prompt but
        uses the exact table mapping from QAExtractionRegistry instead of
        inferring from fix suggestions.

        Args:
            failed_results: List of VerificationResult objects with needs_fix=True
            schema_registry: Current database schema
            qa_extractions: Optional QAExtractionRegistry for exact table mapping
            pg_client: Optional PostgreSQL client for FK value resolution

        Returns:
            List of SQL statements to fix the data
        """
        if not failed_results:
            return []

        logger.info(
            f"Generating fixes for {len(failed_results)} failed QA pairs "
            f"using {'exact QA mapping' if qa_extractions else 'inference'}"
        )

        # Use async parallel processing for fix generation
        logger.info(
            f"Using async parallel processing (concurrency={self.max_concurrency})"
        )
        return asyncio.run(
            self._generate_fixes_async(
                failed_results, schema_registry, qa_extractions, pg_client
            )
        )

    async def _generate_fixes_async(
        self,
        failed_results: List["VerificationResult"],
        schema_registry: SchemaRegistry,
        qa_extractions: Optional[QAExtractionRegistry] = None,
        pg_client: Optional[Any] = None,
    ) -> List[str]:
        """Generate fixes using async parallel processing.

        Args:
            failed_results: List of VerificationResult objects with needs_fix=True
            schema_registry: Current database schema
            qa_extractions: Optional QAExtractionRegistry for exact table mapping
            pg_client: Optional PostgreSQL client for FK value resolution

        Returns:
            List of SQL statements to fix the data
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Create rate limiter (consistent with other LLM-calling code)
        rate_limiter = AsyncRateLimiter(rate=self.requests_per_second)

        # Create thread pool for running sync LLM calls
        executor = ThreadPoolExecutor(max_workers=self.max_concurrency)

        # Results storage
        all_fixes: List[str] = []
        skipped_invalid: int = 0

        async def generate_fix_for_single(
            result: "VerificationResult",
        ) -> Tuple[List[str], int]:
            """Generate fix for a single result asynchronously."""
            async with semaphore:
                await rate_limiter.acquire()
                loop = asyncio.get_running_loop()

                # Get relevant tables (sync operation, fast)
                if qa_extractions:
                    relevant_tables = self.get_relevant_tables_from_extraction(
                        result.qa_index, qa_extractions, schema_registry
                    )
                else:
                    relevant_tables = self._get_relevant_tables_for_fix(
                        result, schema_registry
                    )

                # Filter schema to relevant tables
                filtered_schema = self._filter_schema_for_tables(
                    schema_registry, relevant_tables
                )

                # Run LLM call in thread pool
                result_fixes, skipped = await loop.run_in_executor(
                    executor,
                    self._generate_fixes_for_single_result,
                    result,
                    filtered_schema,
                    schema_registry,
                    pg_client,
                )

                return result_fixes, skipped

        # Create tasks for all results
        tasks = [
            asyncio.create_task(generate_fix_for_single(result))
            for result in failed_results
        ]

        # Process results with progress bar (consistent with other LLM-calling code)
        with tqdm(
            total=len(tasks),
            desc="Generating fixes",
            unit="qa",
            dynamic_ncols=True,
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result_fixes, skipped = await coro
                    all_fixes.extend(result_fixes)
                    skipped_invalid += skipped
                except Exception as e:
                    logger.warning(f"Fix generation task failed: {e}")
                pbar.update(1)
                pbar.set_postfix(
                    fixes=len(all_fixes), skip=skipped_invalid, refresh=False
                )

        executor.shutdown(wait=False)

        logger.info(
            f"Generated {len(all_fixes)} fixes (skipped {skipped_invalid} invalid)"
        )

        return all_fixes

    def _generate_fixes_for_single_result(
        self,
        result: "VerificationResult",
        filtered_schema: List[str],
        schema_registry: SchemaRegistry,
        pg_client: Optional[Any],
    ) -> Tuple[List[str], int]:
        """Generate fixes for a single verification result with retry on failure.

        Uses dynamic schema introspection to provide valid columns to LLM.
        If validation fails, retries with error feedback.

        Args:
            result: VerificationResult to generate fixes for
            filtered_schema: Filtered schema SQL statements
            schema_registry: Full schema registry for validation
            pg_client: Optional PostgreSQL client for FK resolution

        Returns:
            Tuple of (List of SQL statements, count of skipped invalid fixes)
        """
        # Get retry config
        max_retries = self.global_cfg.model.aligned_db.get("max_fix_retries", 2)
        enable_retry = self.global_cfg.model.aligned_db.get(
            "enable_fix_retry_with_feedback", True
        )

        # Build column info dynamically from schema (domain-agnostic)
        table_names = self._extract_table_names_from_schema(filtered_schema)
        column_info = self._build_column_info_for_prompt(schema_registry, table_names)

        schema_text = "\n".join(filtered_schema)
        question, answer = result.original_qa

        missing_facts_text = "\n".join(
            f"- {fact.get('fact', str(fact))}" for fact in result.missing_facts
        )

        previous_error: Optional[str] = None
        max_attempts = (max_retries + 1) if enable_retry else 1

        for attempt in range(max_attempts):
            # Build error feedback section if retrying
            error_section = ""
            if previous_error and attempt > 0:
                error_section = f"""
## PREVIOUS ATTEMPT FAILED - PLEASE CORRECT
Error: {previous_error}
Use ONLY the columns listed in "VALID COLUMNS" above.
"""

            prompt_text = f"""Generate fixes to add missing data to the database.

## VALID COLUMNS (use ONLY these - extracted from schema)
{column_info}

## Database Schema (relevant tables only)
{schema_text}

## Original QA Pair
Question: {question}
Answer: {answer}

## Missing Facts
{missing_facts_text}
{error_section}
## Rules
- Use ONLY columns listed in "VALID COLUMNS" above
- Do NOT use SQL keywords as column names (set, where, select, etc.)
- Do NOT use placeholder values (<id>, $var$, {{{{var}}}}, PENDING)
- If unsure of a value, OMIT that column
- Return [] if no valid fix possible

Output JSON array of fixes:
```json
[
  {{"operation": "INSERT", "entity_type": "table_name", "data": {{"column1": "value1"}}}}
]
```

Generate fixes:"""

            try:
                from src.generator.base import SimpleTextPrompt

                prompt = SimpleTextPrompt(prompt_text)
                response = self._call_with_fallback(prompt, prefix="fix_generation")
                fixes_data = self._parse_json_response_for_fixes(response)

                sql_fixes: List[str] = []
                skipped_count: int = 0
                validation_failed = False
                first_error: Optional[str] = None

                for fix in fixes_data:
                    if not isinstance(fix, dict):
                        continue

                    # Validate fix
                    if self.enable_fix_validation:
                        is_valid, error_msg = self._validate_fix(fix, schema_registry)
                        if not is_valid:
                            if enable_retry and attempt < max_retries:
                                # Save first error for retry feedback
                                if not first_error:
                                    first_error = error_msg
                                validation_failed = True
                                logger.debug(
                                    f"Fix validation failed (attempt {attempt + 1}): {error_msg}"
                                )
                                break
                            else:
                                logger.warning(
                                    f"Skipping invalid fix for QA {result.qa_index}: {error_msg}"
                                )
                                skipped_count += 1
                                continue

                    sql = self._generate_fix_sql(fix, schema_registry)
                    if sql:
                        sql_fixes.append(sql)

                # If validation failed and we have retries left, continue loop
                if validation_failed and first_error and attempt < max_retries:
                    previous_error = first_error
                    continue

                return sql_fixes, skipped_count

            except Exception as e:
                logger.warning(f"Fix generation failed for QA {result.qa_index}: {e}")
                return [], 0

        return [], 0

    def _parse_json_response_for_fixes(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response for fix generation.

        Args:
            response: Raw LLM response

        Returns:
            List of fix dictionaries
        """
        json_str = self._extract_json(response)
        data, _ = safe_json_loads(
            json_str, repairer=self.json_repairer, repair_on_error=True
        )

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "fixes" in data:
                return data["fixes"]
            return [data]
        return []

    def _validate_fix(
        self,
        fix: Dict[str, Any],
        schema_registry: SchemaRegistry,
    ) -> Tuple[bool, Optional[str]]:
        """Validate a fix against the schema before execution.

        Checks that the entity type (table) exists, all columns in the fix
        data exist in the table, and data values are valid.

        Args:
            fix: Fix dictionary with operation, entity_type, data
            schema_registry: Database schema

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        entity_type = fix.get("entity_type", "")
        operation = fix.get("operation", "INSERT")  # Default to INSERT if not provided
        data = fix.get("data", {})

        # Update the fix dict with the default operation for downstream use
        if "operation" not in fix or not fix["operation"]:
            fix["operation"] = operation

        # Check entity_type is provided
        if not entity_type:
            return False, "Missing entity_type in fix"

        # Check operation is valid
        if operation.upper() not in ("INSERT", "UPDATE"):
            return False, f"Invalid operation '{operation}', must be INSERT or UPDATE"

        # Check data is provided and is a dictionary
        if not data:
            return False, "Missing or empty data in fix"

        if not isinstance(data, dict):
            return False, f"Fix data must be a dict, got {type(data).__name__}"

        # Check table exists in schema
        table = schema_registry.get_table(entity_type)
        if not table:
            available_tables = list(schema_registry.get_table_names())
            return False, (
                f"Table '{entity_type}' does not exist. "
                f"Available tables: {available_tables[:10]}..."
            )

        # Check all columns exist in the table
        valid_columns = set(table.get_column_names())
        invalid_columns = [col for col in data.keys() if col not in valid_columns]
        if invalid_columns:
            return False, (
                f"Invalid columns {invalid_columns} for table '{entity_type}'. "
                f"Valid columns: {list(valid_columns)}"
            )

        # Check for malformed subqueries in data
        for col, val in data.items():
            if isinstance(val, str):
                # Check for quoted subquery strings (malformed SQL)
                if val.startswith("(SELECT") or val.startswith("'(SELECT"):
                    return False, (
                        f"Subquery string in column '{col}' - use plain values"
                    )

        # Check for non-integer values in SERIAL/INTEGER PK columns
        pk = table.get_primary_key()
        for col, val in data.items():
            col_info = table.get_column(col)
            if col_info and col_info.data_type.upper() in ("INTEGER", "SERIAL"):
                # Allow integers and numeric strings
                if isinstance(val, str) and not val.lstrip("-").isdigit():
                    # Check if it's a string ID like 'doc_something' for a SERIAL column
                    if col == pk or col.endswith("_id"):
                        return False, (
                            f"Non-integer value '{val}' for integer column '{col}'"
                        )

        # Check for SQL reserved words used as column names (dynamic check)
        sql_keywords = self._get_sql_reserved_words()
        keyword_cols = [col for col in data.keys() if col.lower() in sql_keywords]
        if keyword_cols:
            return False, f"SQL keywords used as columns: {keyword_cols}"

        # Check for placeholder values (pattern-based, not value-specific)
        for col, val in data.items():
            if isinstance(val, str) and self._looks_like_placeholder(val):
                return False, f"Placeholder value '{val}' in column '{col}'"

        return True, None

    def _get_sql_reserved_words(self) -> Set[str]:
        """Return set of SQL reserved words that cannot be column names.

        These are universal SQL keywords, not domain-specific.
        """
        return {
            "select",
            "from",
            "where",
            "insert",
            "update",
            "delete",
            "set",
            "values",
            "into",
            "create",
            "drop",
            "alter",
            "table",
            "index",
            "and",
            "or",
            "not",
            "null",
            "true",
            "false",
            "join",
            "on",
            "order",
            "by",
            "group",
            "having",
            "limit",
            "offset",
            "union",
            "references",
            "foreign",
            "key",
            "primary",
            "constraint",
            "unique",
            "check",
            "default",
            "as",
            "is",
            "in",
            "like",
        }

    def _looks_like_placeholder(self, value: str) -> bool:
        """Check if a value looks like a placeholder (pattern-based).

        This is domain-agnostic - detects common placeholder patterns.
        """
        placeholder_patterns = [
            r"^<[^>]+>$",  # <anything>
            r"^\$[^$]+\$$",  # $anything$
            r"^\{\{[^}]+\}\}$",  # {{anything}}
            r"^PLACEHOLDER",  # PLACEHOLDER_*
            r"_PENDING$",  # *_PENDING
            r"^<[^>]+>",  # starts with <tag>
            r"<[^>]+>$",  # ends with <tag>
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    def _build_column_info_for_prompt(
        self,
        schema_registry: SchemaRegistry,
        table_names: Set[str],
    ) -> str:
        """Build explicit column listing dynamically from schema.

        This is domain-agnostic - works for any schema.
        """
        lines = []
        for table_name in sorted(table_names):
            table = schema_registry.get_table(table_name)
            if table:
                # Dynamically categorize columns from schema
                pk_col = table.get_primary_key()
                data_columns = []
                fk_columns = []

                for col in table.columns:
                    if col.is_primary_key:
                        continue  # Skip PK
                    elif col.name.endswith("_id"):
                        fk_columns.append(col.name)
                    else:
                        data_columns.append(col.name)

                lines.append(f"Table '{table_name}':")
                if pk_col:
                    lines.append(f"  Primary key: {pk_col} (auto-generated, DO NOT include)")
                lines.append(
                    f"  Data columns: {', '.join(data_columns) if data_columns else 'none'}"
                )
                if fk_columns:
                    lines.append(
                        f"  FK columns: {', '.join(fk_columns)} (use human-readable names)"
                    )
                lines.append("")
        return "\n".join(lines)

    def _extract_table_names_from_schema(self, schema_sql: List[str]) -> Set[str]:
        """Extract table names from CREATE TABLE statements."""
        table_names: Set[str] = set()
        for sql in schema_sql:
            match = re.match(r"CREATE TABLE\s+(\w+)", sql, re.IGNORECASE)
            if match:
                table_names.add(match.group(1))
        return table_names

    def _resolve_fk_values(
        self,
        fix: Dict[str, Any],
        schema_registry: SchemaRegistry,
        pg_client: Optional[Any],
    ) -> Dict[str, Any]:
        """Resolve string values in FK columns to integer IDs.

        When the LLM generates fixes with human-readable names (e.g., "Santiago")
        in FK columns (e.g., birth_place_id), this method queries the database
        to find the actual integer ID.

        Args:
            fix: Fix dictionary with entity_type, operation, and data
            schema_registry: Database schema for FK lookups
            pg_client: PostgreSQL client for querying IDs (if None, returns fix unchanged)

        Returns:
            Modified fix with string FK values resolved to integer IDs
        """
        if pg_client is None:
            return fix

        entity_type: str = fix.get("entity_type", "")
        data: Dict[str, Any] = fix.get("data", {})

        if not entity_type or not data:
            return fix

        # Get the table schema
        table = schema_registry.get_table(entity_type)
        if not table:
            return fix

        # Create a mutable copy of the fix
        resolved_fix: Dict[str, Any] = {
            "entity_type": fix.get("entity_type"),
            "operation": fix.get("operation"),
            "data": dict(data),
        }

        # Find all FK columns and their referenced tables
        fk_mapping: Dict[str, Tuple[str, str]] = {}  # column -> (ref_table, ref_column)
        for fk in table.foreign_keys:
            fk_mapping[fk.column_name] = (fk.references_table, fk.references_column)

        # Resolve string values in FK columns
        for col_name, value in list(resolved_fix["data"].items()):
            # Skip if not a string value
            if not isinstance(value, str):
                continue

            # Skip if already an integer string
            if value.lstrip("-").isdigit():
                continue

            # Check if this is an FK column (either explicit FK or ends with _id)
            ref_table: Optional[str] = None

            if col_name in fk_mapping:
                ref_table, _ = fk_mapping[col_name]
            elif col_name.endswith("_id"):
                # Infer referenced table from column name (e.g., birth_place_id -> location)
                base_name = col_name[:-3]  # Remove '_id'

                # Try direct table name match
                if schema_registry.get_table(base_name):
                    ref_table = base_name
                else:
                    # Try common mappings (e.g., birth_place -> location)
                    common_mappings = {
                        "birth_place": "location",
                        "residence": "location",
                        "death_place": "location",
                    }
                    mapped_table = common_mappings.get(base_name)
                    if mapped_table and schema_registry.get_table(mapped_table):
                        ref_table = mapped_table

            if not ref_table:
                continue

            # Query the referenced table to find the ID
            resolved_id = self._lookup_entity_id(
                pg_client, ref_table, value, schema_registry
            )

            if resolved_id is not None:
                resolved_fix["data"][col_name] = resolved_id
                logger.debug(
                    f"Resolved FK value: {col_name}='{value}' -> {resolved_id} "
                    f"(from {ref_table})"
                )

        return resolved_fix

    def _get_lookup_column(
        self,
        table_name: str,
        schema_registry: SchemaRegistry,
    ) -> Optional[str]:
        """Get the lookup column for an entity type (same logic as AlignedDB).

        Uses the following priority:
        1. Schema-based lookup:
           a. Column matching '{table}_name' pattern (e.g., 'occupation_name')
           b. Column matching table name (e.g., 'gender_identity')
           c. Common column names: 'name', 'title', 'full_name'
           d. First non-PK TEXT column in the table
        2. Hardcoded fallback mappings for known entity types

        Args:
            table_name: Table name to look up
            schema_registry: Schema for column information

        Returns:
            Column name to use for lookups, or None if not found
        """
        table = schema_registry.get_table(table_name)
        if table:
            column_names = table.get_column_names()
            pk_name = table.get_primary_key() or f"{table_name}_id"

            # Priority 1a: '{table}_name' pattern (most common convention)
            entity_name_col = f"{table_name}_name"
            if entity_name_col in column_names:
                return entity_name_col

            # Priority 1b: Column matching table name (e.g., 'gender_identity')
            if table_name in column_names:
                return table_name

            # Priority 1c: Common column names
            for candidate in ["name", "title", "full_name", "label"]:
                if candidate in column_names:
                    return candidate

            # Priority 1d: First non-PK TEXT/VARCHAR column
            for col in table.columns:
                if col.name != pk_name and col.data_type.upper() in ("TEXT", "VARCHAR"):
                    return col.name

        # Priority 2: Hardcoded fallback mappings (matches AlignedDB._get_entity_lookup_column)
        fallback_columns: Dict[str, str] = {
            "work": "title",
            "person": "full_name",
            "award": "award_name",
            "series": "series_name",
            "theme": "theme_name",
            "genre": "genre_name",
            "occupation": "occupation_name",
            "location": "city",  # location often uses 'city' column
            "language": "language_name",
            "nationality": "nationality_name",
            "identity": "identity_label",
            "character": "character_name",
            "culture": "culture_name",
            "mythology": "mythology_name",
        }

        return fallback_columns.get(table_name, f"{table_name}_name")

    def _lookup_entity_id(
        self,
        pg_client: Any,
        table_name: str,
        lookup_value: str,
        schema_registry: SchemaRegistry,
    ) -> Optional[int]:
        """Look up an entity ID by its name/title in the referenced table.

        Uses _get_lookup_column to find the appropriate column, then queries
        the database with case-insensitive matching.

        Args:
            pg_client: PostgreSQL client
            table_name: Table to search in
            lookup_value: Value to search for
            schema_registry: Schema for column information

        Returns:
            Integer ID if found, None otherwise
        """
        table = schema_registry.get_table(table_name)
        if not table:
            return None

        # Get primary key column
        pk_column = table.get_primary_key()
        if not pk_column:
            pk_column = f"{table_name}_id"

        # Get the lookup column using consistent logic with AlignedDB
        lookup_col = self._get_lookup_column(table_name, schema_registry)
        if not lookup_col or not table.has_column(lookup_col):
            # Fallback: try additional common columns
            fallback_cols = ["name", "full_name", "title", "label", "city"]
            lookup_col = next((c for c in fallback_cols if table.has_column(c)), None)
            if not lookup_col:
                logger.debug(
                    f"FK lookup: No lookup column found for {table_name}. "
                    f"Available: {list(table.get_column_names())}"
                )
                return None

        # Try to find the entity
        try:
            # First try exact match (case-insensitive)
            query = f"SELECT {pk_column} FROM {table_name} WHERE LOWER({lookup_col}) = LOWER(%s) LIMIT 1"
            with pg_client.conn.cursor() as cursor:
                cursor.execute(query, (lookup_value,))
                row = cursor.fetchone()
                if row:
                    logger.debug(
                        f"FK lookup: Found {table_name}.{pk_column}={row[0]} "
                        f"for {lookup_col}='{lookup_value}'"
                    )
                    return row[0]

                # For location-like values (e.g., "Santiago, Chile"), try the first part
                if "," in lookup_value:
                    first_part = lookup_value.split(",")[0].strip()
                    cursor.execute(query, (first_part,))
                    row = cursor.fetchone()
                    if row:
                        logger.debug(
                            f"FK lookup: Found {table_name}.{pk_column}={row[0]} "
                            f"for {lookup_col}='{first_part}' (from '{lookup_value}')"
                        )
                        return row[0]

                # Try normalized value (strip whitespace, handle case variations)
                normalized_value = lookup_value.strip()
                if normalized_value != lookup_value:
                    cursor.execute(query, (normalized_value,))
                    row = cursor.fetchone()
                    if row:
                        logger.debug(
                            f"FK lookup: Found {table_name}.{pk_column}={row[0]} "
                            f"for {lookup_col}='{normalized_value}' (normalized)"
                        )
                        return row[0]

        except Exception as e:
            logger.debug(f"FK lookup failed for {table_name}.{lookup_col}: {e}")

        return None

    def generate_fixes_with_dedicated_prompt(
        self,
        failed_results: List[VerificationResult],
        schema_registry: SchemaRegistry,
        pg_client: Optional[Any] = None,
    ) -> List[str]:
        """Generate fixes using dedicated LLM prompt with full schema context.

        This method aggregates missing facts from failed verifications and uses
        a dedicated prompt that emphasizes using only existing tables/columns.

        If async round-trip processing is enabled, uses parallel processing with
        semaphore-controlled concurrency and cache-aware rate limiting.
        If pg_client is provided, resolves string FK values to integer IDs before validation.

        Args:
            failed_results: List of VerificationResult objects with needs_fix=True
            schema_registry: Current database schema
            pg_client: Optional PostgreSQL client for FK value resolution

        Returns:
            List of SQL statements to fix the data
        """
        if not self.use_dedicated_fix_prompt:
            # Fall back to original method
            return self.generate_fixes(failed_results, schema_registry, pg_client)

        # Use async parallel processing if enabled
        if self.enable_async_round_trip:
            return asyncio.run(
                self._generate_fixes_async(failed_results, schema_registry, pg_client)
            )

        all_fixes: List[str] = []
        skipped_invalid: int = 0
        cache_hits: int = 0
        cache_misses: int = 0

        # Build table->columns mapping for the prompt
        table_columns: Dict[str, List[str]] = {}
        for table_name in schema_registry.get_table_names():
            table = schema_registry.get_table(table_name)
            if table:
                table_columns[table_name] = table.get_column_names()

        # Count results that actually need processing
        results_to_process = [
            r for r in failed_results if r.needs_fix and r.missing_facts
        ]
        total_to_process = len(results_to_process)

        logger.info(
            f"Generating fixes for {total_to_process} QA pairs with missing facts..."
        )

        for idx, result in enumerate(results_to_process):
            question, answer = result.original_qa

            # Log progress every 5 items or for first/last
            if idx == 0 or (idx + 1) % 5 == 0 or idx == total_to_process - 1:
                logger.info(
                    f"  Processing fix {idx + 1}/{total_to_process} (QA pair {result.qa_index})..."
                )

            # Filter schema to only relevant tables for this QA pair
            relevant_tables: Set[str] = self._get_relevant_tables_for_fix(
                result, schema_registry
            )
            filtered_schema: List[str] = self._filter_schema_for_tables(
                schema_registry, relevant_tables
            )

            # Create prompt with filtered schema context
            prompt = self.fix_generation_prompt(
                question=question,
                answer=answer,
                missing_facts=result.missing_facts,
                schema=filtered_schema,
                table_columns={
                    k: v for k, v in table_columns.items() if k in relevant_tables
                },
            )

            # Call LLM
            response = self._call_with_fallback(
                prompt, prefix="fix_generation", track_api_calls=True
            )

            # Track cache statistics
            if self.llm_api_caller.last_call_was_cached:
                cache_hits += 1
            else:
                cache_misses += 1

            # Parse response
            fixes_data = self._parse_fixes_response(response)

            # Validate and generate SQL for each fix
            for fix in fixes_data:
                # Resolve FK string values to IDs if pg_client is available
                if pg_client is not None:
                    fix = self._resolve_fk_values(fix, schema_registry, pg_client)

                if self.enable_fix_validation:
                    is_valid, error_msg = self._validate_fix(fix, schema_registry)
                    if not is_valid:
                        logger.warning(
                            f"Skipping invalid fix for QA {result.qa_index}: {error_msg}"
                        )
                        skipped_invalid += 1
                        continue

                sql = self._generate_fix_sql(fix, schema_registry)
                if sql:
                    all_fixes.append(sql)

        logger.info(
            f"Generated {len(all_fixes)} fixes via dedicated prompt "
            f"(skipped {skipped_invalid} invalid, "
            f"cache: {cache_hits} hits / {cache_misses} misses)"
        )
        return all_fixes

    def _parse_fixes_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse fix generation response from LLM.

        Handles various edge cases including truncated JSON from LLM responses.

        Args:
            response: Raw LLM response string

        Returns:
            List of fix dictionaries
        """
        try:
            json_str = self._extract_json(response)

            # Use safe_json_loads with repair capability
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )

            if was_repaired:
                logger.info("JSON was repaired successfully")

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Handle dict with "fixes" key
                if "fixes" in data:
                    return data["fixes"]
                # Handle single fix dict - wrap in list
                if "entity_type" in data and "operation" in data:
                    return [data]
                logger.warning(
                    f"Unexpected fixes dict format (keys: {list(data.keys())})"
                )
                return []

            logger.warning(f"Unexpected fixes response format: {type(data)}")
            return []

        except json.JSONDecodeError as e:
            # Handle truncated JSON from LLM - try to parse partial valid fixes
            json_str = self._extract_json(response)
            partial_fixes = self._try_parse_partial_json_array(json_str)
            if partial_fixes:
                logger.warning(
                    f"Recovered {len(partial_fixes)} fixes from truncated JSON"
                )
                return partial_fixes

            logger.error(f"Failed to parse fixes response JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            return []

    def _try_parse_partial_json_array(self, json_str: str) -> List[Dict[str, Any]]:
        """Try to parse a potentially truncated JSON array by extracting complete objects.

        Args:
            json_str: Potentially incomplete JSON array string

        Returns:
            List of successfully parsed dictionaries
        """
        fixes: List[Dict[str, Any]] = []

        # Find all complete JSON objects in the string
        # Pattern matches balanced braces for simple objects (no nested objects with braces)
        obj_pattern = re.compile(
            r'\{\s*"[^{}]*"\s*:\s*(?:"[^"]*"|[^{}]*)\s*(?:,\s*"[^{}]*"\s*:\s*(?:"[^"]*"|[^{}]*|\{[^{}]*\}))*\s*\}'
        )

        for match in obj_pattern.finditer(json_str):
            try:
                obj = json.loads(match.group(0))
                # Validate it looks like a fix object
                if isinstance(obj, dict) and "entity_type" in obj:
                    fixes.append(obj)
            except json.JSONDecodeError:
                continue

        return fixes

    def verify_single(
        self,
        question: str,
        answer: str,
        reconstructed_text: str,
        schema_registry: SchemaRegistry,
        qa_index: int = 0,
    ) -> VerificationResult:
        """Verify a single QA pair against reconstructed text.

        Args:
            question: Original question
            answer: Original answer
            reconstructed_text: Text reconstructed from database
            schema_registry: Current database schema
            qa_index: Index of this QA pair

        Returns:
            VerificationResult
        """
        schema_sql = schema_registry.to_sql_list()
        return self._compare(question, answer, reconstructed_text, schema_sql, qa_index)

    # =========================================================================
    # Protected Methods
    # =========================================================================

    def _reconstruct_from_db(
        self,
        question: str,
        answer: str,
        pg_client,
        schema_registry: SchemaRegistry,
    ) -> str:
        """Get database records as text using intelligent lookup.

        Args:
            question: Original question
            answer: Original answer
            pg_client: PostgreSQL client
            schema_registry: Database schema

        Returns:
            Records as text or error message
        """
        # Step 1: Generate lookup plan (with relationship documentation)
        lookup_plan = self._generate_lookup_plan(
            question, answer, schema_registry.to_sql_with_relationships()
        )

        if not lookup_plan:
            return "No records (failed to generate lookup plan)"

        # Step 2: Execute lookup plan
        raw_records = self._execute_lookup_plan(
            pg_client, lookup_plan, schema_registry=schema_registry
        )

        # Step 3: Convert to text (no more synthesis)
        records_text = self._data_to_text(raw_records)

        return records_text

    def _reconstruct_from_db_with_logging(
        self,
        question: str,
        answer: str,
        pg_client,
        schema_registry: SchemaRegistry,
        intermediate_data: VerificationIntermediateData,
    ) -> Tuple[str, VerificationIntermediateData]:
        """Get database records with intermediate data capture for debugging.

        This method generates a lookup plan, executes it, and returns the
        records as text. It no longer synthesizes an answer - that step is
        replaced by direct judgment.

        Args:
            question: Original question
            answer: Original answer
            pg_client: PostgreSQL client
            schema_registry: Database schema
            intermediate_data: Data object to populate with intermediate results

        Returns:
            Tuple of (records_text, updated_intermediate_data)
        """
        # Use schema with relationship documentation for better lookup generation
        schema_with_rels = schema_registry.to_sql_with_relationships()

        # Step 1: Generate lookup plan with raw response capture
        prompt = self.round_trip_lookup_prompt(
            schema=schema_with_rels,
            question=question,
            answer=answer,
        )
        lookup_response = self._call_with_fallback(
            prompt, prefix="round_trip_lookup", track_api_calls=True
        )
        intermediate_data.lookup_plan_raw_response = lookup_response

        # Parse lookup plan
        try:
            json_str = self._extract_json(lookup_response)
            # Use safe_json_loads with repair capability
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )
            if was_repaired:
                logger.info("JSON was repaired successfully")
            lookup_plan = data.get("lookups", [])
        except Exception as e:
            logger.error(f"Failed to parse lookup plan: {e}")
            lookup_plan = []
            intermediate_data.error = f"Lookup plan parse error: {e}"

        intermediate_data.lookup_plan = lookup_plan

        if not lookup_plan:
            intermediate_data.records_as_text = (
                "No records (failed to generate lookup plan)"
            )
            return intermediate_data.records_as_text, intermediate_data

        # Step 2: Execute lookup plan and capture records
        raw_records = self._execute_lookup_plan(
            pg_client, lookup_plan, schema_registry=schema_registry
        )
        intermediate_data.retrieved_records = raw_records

        # Step 3: Convert records to text format (no more synthesis step)
        records_text = self._data_to_text(raw_records)
        intermediate_data.records_as_text = records_text

        return records_text, intermediate_data

    def _compare_with_logging(
        self,
        question: str,
        answer: str,
        reconstructed: str,
        schema_sql: List[str],
        qa_index: int,
        intermediate_data: VerificationIntermediateData,
    ) -> Tuple[VerificationResult, VerificationIntermediateData]:
        """Compare with intermediate data capture for debugging.

        This method wraps _compare and captures all intermediate results
        for later analysis of verification failures.

        Args:
            question: Original question
            answer: Original answer
            reconstructed: Reconstructed text from database
            schema_sql: Schema as SQL statements
            qa_index: Index of this QA pair
            intermediate_data: Data object to populate with comparison results

        Returns:
            Tuple of (VerificationResult, updated_intermediate_data)
        """
        # Create prompt and call LLM
        prompt = self.round_trip_comparison_prompt(
            question=question,
            answer=answer,
            reconstructed_text=reconstructed,
            schema=schema_sql,
            qa_index=qa_index,
        )

        result_str = self._call_with_fallback(prompt, track_api_calls=True)

        # Parse result
        comparison = self._parse_comparison_result(result_str)
        intermediate_data.comparison_result = comparison

        # Build VerificationResult
        similarity = comparison.get("similarity_score", 0.5)
        needs_fix = similarity < self.similarity_threshold

        intermediate_data.final_score = similarity
        intermediate_data.needs_fix = needs_fix

        verification_result = VerificationResult(
            qa_index=qa_index,
            original_qa=(question, answer),
            reconstructed_text=reconstructed,
            similarity_score=similarity,
            missing_facts=comparison.get("missing_in_reconstruction", []),
            extra_facts=comparison.get("extra_in_reconstruction", []),
            needs_fix=needs_fix,
            suggested_fixes=comparison.get("suggested_fixes", []),
        )

        return verification_result, intermediate_data

    def _judge_with_logging(
        self,
        question: str,
        answer: str,
        records_text: str,
        schema_sql: List[str],
        qa_index: int,
        intermediate_data: VerificationIntermediateData,
    ) -> Tuple[VerificationResult, VerificationIntermediateData]:
        """Judge record sufficiency directly without intermediate synthesis.

        This method replaces the two-step synthesis+comparison with a single
        judgment step that directly evaluates whether records support the answer.

        Args:
            question: Original question
            answer: Original answer
            records_text: Text representation of retrieved database records
            schema_sql: Schema as SQL statements
            qa_index: Index of this QA pair
            intermediate_data: Data object to populate with judgment results

        Returns:
            Tuple of (VerificationResult, updated_intermediate_data)
        """
        # Create judgment prompt and call LLM
        prompt = self.round_trip_judgment_prompt(
            question=question,
            answer=answer,
            records=records_text,
            schema=schema_sql,
            qa_index=qa_index,
        )

        result_str = self._call_with_fallback(
            prompt, prefix="round_trip_judgment", track_api_calls=True
        )

        # Parse judgment result
        judgment = self._parse_judgment_result(result_str)
        intermediate_data.comparison_result = judgment

        # Extract score and determine if fix is needed
        similarity = judgment.get("similarity_score", 0.5)
        needs_fix = similarity < self.similarity_threshold

        intermediate_data.final_score = similarity
        intermediate_data.needs_fix = needs_fix
        # Store a summary in synthesized_answer for backward compatibility
        intermediate_data.synthesized_answer = (
            f"JUDGMENT: {judgment.get('match_quality', 'unknown')} "
            f"(score={similarity:.2f}, sufficient={judgment.get('sufficient', False)})"
        )

        verification_result = VerificationResult(
            qa_index=qa_index,
            original_qa=(question, answer),
            reconstructed_text=records_text,  # Use records as the "reconstruction"
            similarity_score=similarity,
            missing_facts=judgment.get("missing_facts", []),
            extra_facts=judgment.get("extra_facts", []),
            needs_fix=needs_fix,
            suggested_fixes=judgment.get("suggested_fixes", []),
        )

        return verification_result, intermediate_data

    def _parse_judgment_result(self, response: str) -> Dict[str, Any]:
        """Parse the judgment result from LLM response.

        Args:
            response: Raw LLM response containing JSON judgment

        Returns:
            Parsed judgment dictionary with fields:
            - sufficient: bool
            - similarity_score: float
            - match_quality: str
            - missing_facts: list
            - extra_facts: list
            - suggested_fixes: list
        """
        try:
            json_str = self._extract_json(response)
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )
            if was_repaired:
                logger.info("JSON was repaired successfully")

            # Handle case where LLM returns a list instead of dict
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    logger.debug("Judgment returned as list, extracting first element")
                    data = data[0]
                else:
                    raise ValueError(
                        "Judgment result is a list without valid dict element"
                    )

            # Validate and normalize the response
            result = {
                "sufficient": data.get("sufficient", False),
                "similarity_score": float(data.get("similarity_score", 0.5)),
                "match_quality": data.get("match_quality", "unknown"),
                "missing_facts": data.get("missing_facts", []),
                "extra_facts": data.get("extra_facts", []),
                "suggested_fixes": data.get("suggested_fixes", []),
            }

            # Ensure score is in valid range
            result["similarity_score"] = max(0.0, min(1.0, result["similarity_score"]))

            return result

        except Exception as e:
            logger.error(f"Failed to parse judgment result: {e}")
            return {
                "sufficient": False,
                "similarity_score": 0.0,
                "match_quality": "error",
                "missing_facts": [{"fact": f"Parse error: {e}", "severity": "high"}],
                "extra_facts": [],
                "suggested_fixes": [],
            }

    def _check_qa_inconsistency(
        self,
        question: str,
        answer: str,
    ) -> Tuple[bool, str]:
        """Check if a QA pair has native inconsistency.

        This method detects cases where the question asks about something
        that the answer doesn't actually address, which is a data quality
        issue rather than a pipeline extraction issue.

        Args:
            question: The question text
            answer: The answer text

        Returns:
            Tuple of (has_inconsistency, details_string)
        """
        from src.llm.tracker import llm_call_tracker
        from src.prompt.db_construction.qa_consistency_check import (
            QAConsistencyCheckPrompt,
        )

        try:
            # Build prompt from template
            prompt = QAConsistencyCheckPrompt(question=question, answer=answer)
            prompt_text = str(prompt)

            # Use LiteLLM directly for this simple text prompt. Prefer the
            # configured provider-qualified model name and keep the fallback on
            # the repo's nano default.
            model = self.api_cfg.get(
                "model_name",
                self.api_cfg.get("model", "openai/gpt-5.4-nano-2026-03-17"),
            )
            llm_response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0,
            )
            response = llm_response.choices[0].message.content
            llm_call_tracker.record_call("qa_consistency_check")

            json_str = self._extract_json(response)
            data, _ = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )

            is_consistent = data.get("is_consistent", True)
            logger.debug(
                f"QA consistency check: is_consistent={is_consistent}, "
                f"type={data.get('inconsistency_type', 'none')}"
            )

            if not is_consistent:
                inconsistency_type = data.get("inconsistency_type", "unknown")
                question_asks = data.get("question_asks_about", "")
                answer_provides = data.get("answer_provides", "")
                details = data.get("details", "")

                detail_str = (
                    f"Type: {inconsistency_type}. "
                    f"Question asks about: {question_asks}. "
                    f"Answer provides: {answer_provides}. "
                    f"{details}"
                )
                logger.info(f"QA inconsistency detected: {detail_str[:150]}...")
                return True, detail_str

            return False, ""

        except Exception as e:
            logger.warning(f"QA consistency check failed: {e}")
            return False, ""

    def _generate_lookup_plan(
        self, question: str, answer: str, schema: str
    ) -> List[Dict[str, Any]]:
        """Generate a structured lookup plan using LLM.

        Args:
            question: Original question
            answer: Original answer
            schema: Schema string (with relationship docs if available)

        Returns:
            List of lookup dictionaries
        """
        prompt = self.round_trip_lookup_prompt(
            schema=schema,
            question=question,
            answer=answer,
        )
        response = self._call_with_fallback(
            prompt, prefix="round_trip_lookup", track_api_calls=True
        )

        try:
            json_str = self._extract_json(response)
            # Use safe_json_loads with repair capability
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )
            if was_repaired:
                logger.info("JSON was repaired successfully")
            return data.get("lookups", [])
        except Exception as e:
            logger.error(f"Failed to parse lookup plan: {e}")
            return []

    def _execute_lookup_plan(
        self,
        pg_client,
        plan: List[Dict[str, Any]],
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute the lookup plan against the database with FK placeholder resolution.

        Uses a hybrid approach:
        1. Execute direct (name-based) lookups first and cache results
        2. Resolve FK placeholders using cached entity name → ID mappings
        3. For junction tables, use cached IDs from previous lookups

        This allows lookups like:
          lookup1: person WHERE full_name = "Jaime Vasquez" → caches person_id=1
          lookup2: work WHERE genre_id = <from_lookup3> → resolves using cached genre_id
          lookup3: person_work WHERE person_id = <from_lookup1> → uses cached person_id=1
        """
        plan_start = time.perf_counter()
        self._record_verification_stat("lookup_plans_executed")
        self._record_verification_stat("lookup_plan_items", len(plan))
        results: Dict[str, List[Dict[str, Any]]] = {}
        # Cache: table_name → {column_name → value} from first result row
        entity_cache: Dict[str, Dict[str, Any]] = {}
        # Cache: "lookup{N}" → {column_name → value} for placeholder resolution
        lookup_cache: Dict[str, Dict[str, Any]] = {}

        try:
            for idx, item in enumerate(plan):
                table_name = item.get("table")
                conditions = item.get("conditions", {})
                lookup_key = f"lookup{idx + 1}"

                if not table_name:
                    continue

                metadata = self._get_table_lookup_metadata(
                    table_name, pg_client, schema_registry=schema_registry
                )
                if metadata is None:
                    logger.debug(f"Table {table_name} not found or has no columns")
                    continue

                actual_columns = set(metadata.columns)

                # Separate direct conditions from placeholder conditions
                direct_conditions: Dict[str, Any] = {}
                placeholder_conditions: Dict[str, str] = {}

                for col, val in conditions.items():
                    if isinstance(val, str) and val.startswith("<") and val.endswith(">"):
                        placeholder_conditions[col] = val
                    else:
                        direct_conditions[col] = val

                # Resolve placeholders using caches
                resolved_conditions = self._resolve_placeholder_conditions(
                    placeholder_conditions, entity_cache, lookup_cache, actual_columns
                )

                # Merge direct and resolved conditions
                all_conditions = {**direct_conditions, **resolved_conditions}

                # Build and execute query
                rows = self._execute_table_query(
                    pg_client, table_name, all_conditions, actual_columns
                )
                logger.debug(
                    f"Lookup {lookup_key} ({table_name}): query with {all_conditions} "
                    f"returned {len(rows)} rows"
                )

                # If no results and we have FK conditions, try junction table fallback
                if not rows:
                    fk_conditions = {
                        k: v
                        for k, v in all_conditions.items()
                        if k.endswith("_id") and isinstance(v, int)
                    }
                    if fk_conditions:
                        rows = self._try_junction_table_fallback(
                            pg_client,
                            table_name,
                            fk_conditions,
                            entity_cache,
                            schema_registry=schema_registry,
                        )

                # Special fallback for work lookups when author name is provided
                if not rows and table_name == "work":
                    rows = self._try_work_author_fallback(
                        pg_client,
                        all_conditions,
                        entity_cache,
                        schema_registry=schema_registry,
                    )

                # Special fallback for person lookups - try alternative name columns
                if not rows and table_name == "person":
                    rows = self._try_person_name_fallback(
                        pg_client, all_conditions, schema_registry=schema_registry
                    )

                if rows:
                    # APPEND to existing results instead of overwriting
                    if table_name in results:
                        # Merge new rows, avoiding duplicates based on primary key
                        existing_ids = set()
                        pk_col = metadata.primary_key or f"{table_name}_id"
                        for existing_row in results[table_name]:
                            if pk_col in existing_row:
                                existing_ids.add(existing_row[pk_col])
                        new_count = 0
                        for row in rows:
                            if pk_col not in row or row[pk_col] not in existing_ids:
                                results[table_name].append(row)
                                new_count += 1
                                if pk_col in row:
                                    existing_ids.add(row[pk_col])
                        logger.debug(
                            f"Lookup {lookup_key}: Merged {new_count} new rows into "
                            f"{table_name} (total: {len(results[table_name])})"
                        )
                    else:
                        results[table_name] = list(rows)  # Make a copy
                    # Cache the first row's values for FK resolution in later lookups
                    first_row = rows[0]
                    entity_cache[table_name] = first_row
                    lookup_cache[lookup_key] = first_row
                    logger.debug(
                        f"Lookup {lookup_key} ({table_name}): found {len(rows)} rows, "
                        f"cached keys: {list(first_row.keys())[:5]}..."
                    )
                else:
                    if table_name not in results:
                        results[table_name] = []
                    logger.debug(
                        f"Lookup {lookup_key} ({table_name}): no results "
                        f"(conditions: {all_conditions})"
                    )

            return results
        finally:
            self._record_verification_stat(
                "total_lookup_time_seconds", time.perf_counter() - plan_start
            )

    def _try_junction_table_fallback(
        self,
        pg_client,
        target_table: str,
        fk_conditions: Dict[str, int],
        entity_cache: Dict[str, Dict[str, Any]],
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> List[Dict[str, Any]]:
        """Try to find records via junction table when direct FK lookup fails.

        This handles cases like:
        - Lookup: work WHERE genre_id = 1 (but work has no genre_id column)
        - Fallback: work_genre WHERE genre_id = 1, then work WHERE work_id IN (...)

        Args:
            pg_client: PostgreSQL client
            target_table: The table we want results from (e.g., "work")
            fk_conditions: FK conditions that failed (e.g., {"genre_id": 1})
            entity_cache: Cache of already-looked-up entities

        Returns:
            List of matching rows from target_table, or empty list
        """
        for fk_col, fk_val in fk_conditions.items():
            # Extract the referenced table from FK column (e.g., "genre_id" -> "genre")
            ref_table = fk_col.replace("_id", "")

            # Try common junction table naming conventions
            junction_candidates = [
                f"{target_table}_{ref_table}",  # work_genre
                f"{ref_table}_{target_table}",  # genre_work
            ]

            for junction_table in junction_candidates:
                try:
                    metadata = self._get_table_lookup_metadata(
                        junction_table, pg_client, schema_registry=schema_registry
                    )
                    jt_columns = set(metadata.columns) if metadata is not None else set()

                    if not jt_columns:
                        continue

                    # Check if junction table has both FK columns we need
                    target_fk = f"{target_table}_id"
                    if fk_col not in jt_columns or target_fk not in jt_columns:
                        continue

                    # Query junction table to get target IDs
                    query = (
                        f"SELECT {target_fk} FROM {junction_table} "
                        f"WHERE {fk_col} = %s"
                    )
                    target_rows = self._fetch_rows(
                        pg_client,
                        query,
                        [fk_val],
                        is_fallback=True,
                    )
                    target_ids = [row[target_fk] for row in target_rows if target_fk in row]

                    if not target_ids:
                        continue

                    # Query target table for those IDs
                    placeholders = ", ".join(["%s"] * len(target_ids))
                    target_query = (
                        f"SELECT * FROM {target_table} "
                        f"WHERE {target_fk} IN ({placeholders}) LIMIT 20"
                    )
                    rows = self._fetch_rows(
                        pg_client,
                        target_query,
                        target_ids,
                        is_fallback=True,
                    )
                    if rows:
                        logger.debug(
                            f"Junction fallback success: {junction_table} -> "
                            f"{target_table} ({len(rows)} rows)"
                        )
                        return rows

                except Exception as e:
                    logger.debug(f"Junction fallback failed for {junction_table}: {e}")
                    continue

        return []

    def _try_person_name_fallback(
        self,
        pg_client,
        conditions: Dict[str, Any],
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> List[Dict[str, Any]]:
        """Try to find person when direct lookup fails by trying alternative columns.

        This handles cases where LLM uses different name fields than what's in the DB.

        Args:
            pg_client: PostgreSQL client
            conditions: Original conditions that failed

        Returns:
            List of matching person rows, or empty list
        """
        # Extract name from various condition keys
        person_name = None
        for key in ["name", "full_name", "person_name", "author", "author_name"]:
            if key in conditions:
                person_name = conditions[key]
                break

        if not person_name:
            return []

        metadata = self._get_table_lookup_metadata(
            "person", pg_client, schema_registry=schema_registry
        )
        actual_columns = set(metadata.columns) if metadata is not None else set()

        # Try different name columns
        name_columns = ["name", "full_name", "person_name"]
        for col in name_columns:
            if col not in actual_columns:
                continue
            try:
                rows = self._execute_text_lookup(
                    pg_client,
                    "person",
                    col,
                    person_name,
                    limit=10,
                    is_fallback=True,
                )
                if rows:
                    logger.debug(
                        f"Person name fallback success: {col}='{person_name}' "
                        f"({len(rows)} rows)"
                    )
                    return rows
            except Exception as e:
                logger.debug(f"Person name fallback failed for {col}: {e}")
                continue

        return []

    def _try_work_author_fallback(
        self,
        pg_client,
        conditions: Dict[str, Any],
        entity_cache: Dict[str, Dict[str, Any]],
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> List[Dict[str, Any]]:
        """Try to find works when direct lookup fails by checking author columns.

        This handles cases where LLM looks for works but doesn't know the exact
        column name for author (author_creator, author, creator) or needs to go
        through person_work junction table.

        Args:
            pg_client: PostgreSQL client
            conditions: Original conditions that failed
            entity_cache: Cache of already-looked-up entities

        Returns:
            List of matching work rows, or empty list
        """
        # Extract author name from various condition keys
        author_name = None
        for key in ["author", "author_name", "creator", "writer", "person_name", "name"]:
            if key in conditions:
                author_name = conditions[key]
                break

        if not author_name:
            # Check if we have a person in cache
            if "person" in entity_cache:
                author_name = entity_cache["person"].get("name")

        if not author_name:
            return []

        work_metadata = self._get_table_lookup_metadata(
            "work", pg_client, schema_registry=schema_registry
        )
        work_columns = set(work_metadata.columns) if work_metadata is not None else set()

        # Strategy 1: Try direct lookup on author-related TEXT columns
        author_columns = ["author_creator", "author", "creator"]
        for col in author_columns:
            if col not in work_columns:
                continue
            try:
                rows = self._execute_text_lookup(
                    pg_client,
                    "work",
                    col,
                    author_name,
                    limit=20,
                    is_fallback=True,
                )
                if rows:
                    logger.debug(
                        f"Work author fallback success: {col}='{author_name}' "
                        f"({len(rows)} rows)"
                    )
                    return rows
            except Exception as e:
                logger.debug(f"Work author fallback failed for {col}: {e}")
                continue

        # Strategy 2: Try via person_work junction table
        try:
            person_rows = self._execute_text_lookup(
                pg_client,
                "person",
                "name",
                author_name,
                limit=1,
                is_fallback=True,
                select_columns="person_id",
            )
            if person_rows and "person_id" in person_rows[0]:
                person_id = person_rows[0]["person_id"]

                # Now find works via junction table
                junction_query = """
                    SELECT w.* FROM work w
                    JOIN person_work pw ON w.work_id = pw.work_id
                    WHERE pw.person_id = %s
                    LIMIT 20
                """
                rows = self._fetch_rows(
                    pg_client,
                    junction_query,
                    [person_id],
                    is_fallback=True,
                )
                if rows:
                    logger.debug(
                        f"Work author fallback via person_work success: "
                        f"person_id={person_id} ({len(rows)} rows)"
                    )
                    return rows
        except Exception as e:
            logger.debug(f"Work author fallback via person_work failed: {e}")

        return []

    def _resolve_placeholder_conditions(
        self,
        placeholder_conditions: Dict[str, str],
        entity_cache: Dict[str, Dict[str, Any]],
        lookup_cache: Dict[str, Dict[str, Any]],
        actual_columns: Set[str],
    ) -> Dict[str, Any]:
        """Resolve placeholder conditions using cached lookup results.

        Handles placeholders like:
        - <person_id_from_lookup1> → lookup_cache["lookup1"]["person_id"]
        - <genre_id_from_lookup3> → lookup_cache["lookup3"]["genre_id"]
        - <work_id> → entity_cache["work"]["work_id"]

        Args:
            placeholder_conditions: Dict of column → placeholder string
            entity_cache: Cache of table_name → first_row_dict
            lookup_cache: Cache of "lookup{N}" → first_row_dict
            actual_columns: Set of actual column names in target table

        Returns:
            Dict of resolved column → value pairs
        """
        resolved: Dict[str, Any] = {}

        for col, placeholder in placeholder_conditions.items():
            # Parse placeholder: <column_from_lookupN> or <column>
            inner = placeholder[1:-1]  # Remove < and >

            resolved_value = None

            # Pattern 1: <column_from_lookupN> (e.g., <person_id_from_lookup1>)
            if "_from_lookup" in inner:
                parts = inner.rsplit("_from_lookup", 1)
                if len(parts) == 2:
                    target_col = parts[0]
                    lookup_num = parts[1]
                    lookup_key = f"lookup{lookup_num}"

                    if lookup_key in lookup_cache:
                        cached_row = lookup_cache[lookup_key]
                        # Try exact column match
                        if target_col in cached_row:
                            resolved_value = cached_row[target_col]
                        # Try with _id suffix
                        elif f"{target_col}_id" in cached_row:
                            resolved_value = cached_row[f"{target_col}_id"]
                        # Try table's primary key (e.g., person_id for person table)
                        else:
                            # Infer table from lookup (person_id usually from person table)
                            for key, val in cached_row.items():
                                if key.endswith("_id") and target_col in key:
                                    resolved_value = val
                                    break

            # Pattern 2: <table_id> or <column> - lookup in entity_cache
            if resolved_value is None:
                # Try to find in entity cache by table name
                if inner.endswith("_id"):
                    table_guess = inner[:-3]  # person_id → person
                    if table_guess in entity_cache:
                        cached_row = entity_cache[table_guess]
                        if inner in cached_row:
                            resolved_value = cached_row[inner]

            # Pattern 3: Direct column reference in any cached table
            if resolved_value is None:
                for cached_row in lookup_cache.values():
                    if inner in cached_row:
                        resolved_value = cached_row[inner]
                        break

            if resolved_value is not None:
                # Map the column name to actual column in target table
                mapped_col = self._map_lookup_column(col, actual_columns, "")
                if mapped_col:
                    resolved[mapped_col] = resolved_value
                    logger.debug(
                        f"Resolved placeholder {placeholder} → {mapped_col}={resolved_value}"
                    )
                else:
                    # Try using the column as-is if it exists
                    if col in actual_columns:
                        resolved[col] = resolved_value
                        logger.debug(
                            f"Resolved placeholder {placeholder} → {col}={resolved_value}"
                        )
            else:
                logger.debug(f"Could not resolve placeholder: {placeholder}")

        return resolved

    def _execute_table_query(
        self,
        pg_client,
        table_name: str,
        conditions: Dict[str, Any],
        actual_columns: Set[str],
    ) -> List[Dict[str, Any]]:
        """Execute a query against a table with the given conditions.

        Args:
            pg_client: PostgreSQL client
            table_name: Table to query
            conditions: Column → value conditions (already resolved)
            actual_columns: Set of actual column names in the table

        Returns:
            List of row dictionaries
        """
        base_clauses: List[str] = []
        base_params: List[Any] = []
        string_conditions: List[Tuple[str, str]] = []

        for col, val in conditions.items():
            # Map column name to actual column
            mapped_col = self._map_lookup_column(col, actual_columns, table_name)
            if not mapped_col:
                logger.debug(f"Skipping unknown column '{col}' for table {table_name}")
                continue

            # For integer values (FK IDs), use exact match
            if isinstance(val, int):
                base_clauses.append(f"{mapped_col} = %s")
                base_params.append(val)
            # For string values, try increasingly permissive matching
            elif isinstance(val, str):
                if self._normalize_lookup_string(val):
                    string_conditions.append((mapped_col, val))
            else:
                base_clauses.append(f"{mapped_col} = %s")
                base_params.append(val)

        if not string_conditions:
            query = f"SELECT * FROM {table_name}"
            if base_clauses:
                query += " WHERE " + " AND ".join(base_clauses)
            query += " LIMIT 20"
            return self._fetch_rows(pg_client, query, base_params)

        for mode in ("exact", "prefix", "substring"):
            where_clauses = list(base_clauses)
            params = list(base_params)
            for mapped_col, val in string_conditions:
                for candidate_mode, candidate_param in self._build_string_lookup_modes(val):
                    if candidate_mode == mode:
                        where_clauses.append(
                            self._build_string_where_clause(mapped_col, mode)
                        )
                        params.append(candidate_param)
                        break

            query = f"SELECT * FROM {table_name}"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " LIMIT 20"
            rows = self._fetch_rows(pg_client, query, params, match_mode=mode)
            if rows:
                return rows

        return []

    def _execute_text_lookup(
        self,
        pg_client,
        table_name: str,
        column_name: str,
        value: str,
        *,
        limit: int = 20,
        is_fallback: bool = False,
        select_columns: str = "*",
    ) -> List[Dict[str, Any]]:
        """Execute a single-column text lookup from exact to substring match."""
        for mode, param in self._build_string_lookup_modes(value):
            query = (
                f"SELECT {select_columns} FROM {table_name} "
                f"WHERE {self._build_string_where_clause(column_name, mode)} "
                f"LIMIT {limit}"
            )
            rows = self._fetch_rows(
                pg_client,
                query,
                [param],
                is_fallback=is_fallback,
                match_mode=mode,
            )
            if rows:
                return rows
        return []

    def _map_lookup_column(
        self, col: str, actual_columns: Set[str], table_name: str
    ) -> Optional[str]:
        """Map a lookup column name to an actual column in the table.

        Handles common naming mismatches like:
        - 'name' -> 'full_name', '{table}_name', 'title'
        - 'relationship' -> 'relationship_type_id'
        - 'author' -> 'person_id' (for junction tables)

        Args:
            col: Column name from lookup plan
            actual_columns: Set of actual column names in the table
            table_name: Name of the table being queried

        Returns:
            Mapped column name if found, None otherwise
        """
        # Direct match
        if col in actual_columns:
            return col

        # Common mappings
        col_lower = col.lower()

        # 'name' variations
        if col_lower == "name":
            for candidate in ["full_name", f"{table_name}_name", "title", "name"]:
                if candidate in actual_columns:
                    return candidate

        # 'relationship' -> 'relationship_type_id'
        if col_lower in ("relationship", "role", "type"):
            if "relationship_type_id" in actual_columns:
                return "relationship_type_id"
            # Try finding any column with the word
            for actual_col in actual_columns:
                if col_lower in actual_col.lower():
                    return actual_col

        # 'author' variations for work table
        if col_lower in ("author", "author_name", "writer", "creator_name"):
            for candidate in ["author_creator", "author", "creator"]:
                if candidate in actual_columns:
                    return candidate

        # 'title' variations
        if col_lower in ("title", "book_title", "work_title"):
            for candidate in ["title", "work_title", "name"]:
                if candidate in actual_columns:
                    return candidate

        # Try adding common suffixes
        for suffix in ["_id", "_name", "_type"]:
            candidate = f"{col}{suffix}"
            if candidate in actual_columns:
                return candidate

        # Case-insensitive match
        for actual_col in actual_columns:
            if col_lower == actual_col.lower():
                return actual_col

        return None

    def _synthesize_answer_from_records(
        self, question: str, raw_records: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Synthesize an answer from raw records using LLM."""
        # Convert records to text format
        records_text = self._data_to_text(raw_records)

        prompt = self.round_trip_answer_prompt(
            question=question,
            records=records_text,
        )

        return self._call_with_fallback(
            prompt, prefix="round_trip_answer", track_api_calls=True
        )

    def _data_to_text(self, data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Convert database records to natural language text."""
        if not data:
            return "No records found."

        lines: List[str] = []

        for table_name, records in data.items():
            if not records:
                lines.append(f"{table_name}: No matching records found.")
                continue

            for record in records:
                # Filter out None values and IDs for cleaner text
                filtered = {
                    k: v
                    for k, v in record.items()
                    if v is not None and not k.endswith("_id") and k != "id"
                }

                if filtered:
                    parts = [f"{k}={v}" for k, v in filtered.items()]
                    lines.append(f"{table_name}: {', '.join(parts)}")

        return "\n".join(lines) if lines else "No relevant data found."

    def _compare(
        self,
        question: str,
        answer: str,
        reconstructed: str,
        schema_sql: List[str],
        qa_index: int,
    ) -> VerificationResult:
        """Compare original QA with reconstructed text.

        Args:
            question: Original question
            answer: Original answer
            reconstructed: Reconstructed text from database
            schema_sql: Schema as SQL statements
            qa_index: Index of this QA pair

        Returns:
            VerificationResult
        """
        # Create prompt
        prompt = self.round_trip_comparison_prompt(
            question=question,
            answer=answer,
            reconstructed_text=reconstructed,
            schema=schema_sql,
            qa_index=qa_index,
        )

        # Call LLM
        result = self._call_with_fallback(prompt, track_api_calls=True)

        # Parse result
        comparison = self._parse_comparison_result(result)

        # Build VerificationResult
        similarity = comparison.get("similarity_score", 0.5)
        needs_fix = similarity < self.similarity_threshold

        return VerificationResult(
            qa_index=qa_index,
            original_qa=(question, answer),
            reconstructed_text=reconstructed,
            similarity_score=similarity,
            missing_facts=comparison.get("missing_in_reconstruction", []),
            extra_facts=comparison.get("extra_in_reconstruction", []),
            needs_fix=needs_fix,
            suggested_fixes=comparison.get("suggested_fixes", []),
        )

    def _judge(
        self,
        question: str,
        answer: str,
        records_text: str,
        schema_sql: List[str],
        qa_index: int,
    ) -> VerificationResult:
        """Judge record sufficiency directly without intermediate synthesis.

        Args:
            question: Original question
            answer: Original answer
            records_text: Text representation of retrieved database records
            schema_sql: Schema as SQL statements
            qa_index: Index of this QA pair

        Returns:
            VerificationResult
        """
        # Create judgment prompt
        prompt = self.round_trip_judgment_prompt(
            question=question,
            answer=answer,
            records=records_text,
            schema=schema_sql,
            qa_index=qa_index,
        )

        # Call LLM
        result_str = self._call_with_fallback(
            prompt, prefix="round_trip_judgment", track_api_calls=True
        )

        # Parse judgment result
        judgment = self._parse_judgment_result(result_str)

        # Build VerificationResult
        similarity = judgment.get("similarity_score", 0.5)
        needs_fix = similarity < self.similarity_threshold

        return VerificationResult(
            qa_index=qa_index,
            original_qa=(question, answer),
            reconstructed_text=records_text,
            similarity_score=similarity,
            missing_facts=judgment.get("missing_facts", []),
            extra_facts=judgment.get("extra_facts", []),
            needs_fix=needs_fix,
            suggested_fixes=judgment.get("suggested_fixes", []),
        )

    def _call_with_fallback(
        self, prompt, prefix: str = "round_trip", track_api_calls: bool = False
    ) -> str:
        """Call the LLM with fallback on TooMuchThinkingError.

        Args:
            prompt: The prompt to send to the LLM.
            prefix: Logging prefix for tracking.
            track_api_calls: If True, increment _actual_api_calls_count for non-cached calls.

        Returns:
            The LLM response string.
        """
        try:
            result = self.llm_api_caller(
                prompt,
                post_process_fn=None,
                prefix=prefix,
            )
            was_cached: bool = self.llm_api_caller.last_call_was_cached
            # Log cache status for debugging
            logger.debug(f"[{prefix}] Cache {'HIT' if was_cached else 'MISS'}")
            # Track actual API calls (not cached) if requested
            if track_api_calls and not was_cached:
                self._increment_api_call_count()
            return result
        except TooMuchThinkingError as e:
            logger.warning(f"Too much thinking: {e}")
            logger.warning("Using fallback model...")
            result = self.fallback_llm_api_caller(
                prompt,
                post_process_fn=None,
                prefix=f"{prefix}_fallback",
            )
            was_cached = self.fallback_llm_api_caller.last_call_was_cached
            # Log cache status for debugging
            logger.debug(f"[{prefix}_fallback] Cache {'HIT' if was_cached else 'MISS'}")
            # Track fallback API calls if requested
            if track_api_calls and not was_cached:
                self._increment_api_call_count()
            return result

    def _parse_comparison_result(self, response: str) -> Dict[str, Any]:
        """Parse comparison result from LLM response.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with comparison results
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)

            # Use safe_json_loads with repair capability
            data, was_repaired = safe_json_loads(
                json_str, repairer=self.json_repairer, repair_on_error=True
            )

            if was_repaired:
                logger.info("JSON was repaired successfully")

            if isinstance(data, dict):
                return data

            logger.warning(f"Unexpected response format: {type(data)}")
            return {"similarity_score": 0.5}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse comparison result JSON: {e}")
            logger.error(f"Response was: {response}")
            return {"similarity_score": 0.5}

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        extracted: str = extract_json_from_response(text)
        return fix_js_string_concat(extracted)

    def _sanitize_fix_data(
        self,
        fix: Dict[str, Any],
        schema_registry: SchemaRegistry,
    ) -> Dict[str, Any]:
        """Sanitize fix data by removing problematic columns and values.

        Removes:
        - Primary key columns (let SERIAL auto-generate)
        - Columns with placeholder values like <new_id>
        - Columns with quoted subquery strings

        Args:
            fix: Fix dictionary with operation, entity_type, data
            schema_registry: Database schema

        Returns:
            Sanitized fix dictionary (or original if no changes needed)
        """
        entity_type = fix.get("entity_type", "")
        data = fix.get("data", {})
        operation = fix.get("operation", "").upper()

        table = schema_registry.get_table(entity_type)
        if not table or not data:
            return fix

        pk = table.get_primary_key()
        sanitized_data: Dict[str, Any] = {}

        for col, val in data.items():
            # Skip primary key columns for INSERT operations
            if operation == "INSERT" and col == pk:
                logger.debug(f"Stripping PK column '{col}' from INSERT fix")
                continue

            # Skip placeholder values
            if isinstance(val, str) and re.match(r"^<[^>]+>$", val):
                logger.debug(f"Stripping placeholder value in column '{col}'")
                continue

            # Skip quoted subquery strings (malformed)
            if isinstance(val, str) and ("(SELECT" in val or "'(SELECT" in val):
                logger.debug(f"Stripping malformed subquery in column '{col}'")
                continue

            # Skip string values for SERIAL/INTEGER PK columns
            if col == pk or (col.endswith("_id") and col != pk):
                col_info = table.get_column(col)
                if col_info and col_info.data_type.upper() in ("INTEGER", "SERIAL"):
                    if isinstance(val, str) and not val.lstrip("-").isdigit():
                        logger.debug(
                            f"Stripping non-integer value '{val}' for column '{col}'"
                        )
                        continue

            sanitized_data[col] = val

        return {
            "entity_type": entity_type,
            "operation": fix.get("operation", "INSERT"),
            "data": sanitized_data,
        }

    def _get_lookup_column_for_table(
        self,
        table_name: str,
        schema_registry: SchemaRegistry,
    ) -> str:
        """Get the column to use for looking up records in a table.

        Priority:
        1. 'name' column
        2. 'title' column
        3. 'full_name' column
        4. 'label' column
        5. First TEXT column that's not PK
        6. Fallback to 'name'

        Args:
            table_name: Name of the table
            schema_registry: Database schema

        Returns:
            Column name to use for lookups
        """
        table = schema_registry.get_table(table_name)
        if not table:
            return "name"

        columns = table.get_column_names()
        pk = table.get_primary_key()

        # Priority order for lookup columns
        for candidate in ["name", "title", "full_name", "label"]:
            if candidate in columns:
                return candidate

        # First TEXT column that's not PK
        for col in table.columns:
            if col.name != pk and col.data_type.upper() in ("TEXT", "VARCHAR"):
                return col.name

        return "name"

    def _generate_fix_sql(
        self,
        fix: Dict[str, Any],
        schema_registry: SchemaRegistry,
    ) -> Optional[str]:
        """Generate SQL UPSERT statement for a single fix.

        Uses ON CONFLICT clause to handle duplicates gracefully, updating
        existing records instead of failing or creating duplicates.

        Key improvements:
        - Sanitizes fix data first (removes PK columns, placeholders, etc.)
        - Never includes PK columns in INSERT statements
        - Never includes PK columns in ON CONFLICT DO UPDATE SET clause
        - Handles FK columns by resolving string values to subqueries

        Args:
            fix: Fix dictionary with operation, entity_type, data
            schema_registry: Database schema

        Returns:
            SQL statement or None if invalid
        """
        # Sanitize fix data first (removes PK columns, placeholders, etc.)
        fix = self._sanitize_fix_data(fix, schema_registry)

        operation = fix.get("operation", "").upper()
        entity_type = fix.get("entity_type", "")
        data = fix.get("data", {})

        if not entity_type or not data:
            return None

        table = schema_registry.get_table(entity_type)
        if not table:
            return None

        # Get PK to exclude from updates
        pk = table.get_primary_key()

        # Build FK column map for resolving string values to subqueries
        fk_column_map: Dict[str, Any] = {}
        for fk in table.foreign_keys:
            fk_column_map[fk.column_name] = fk

        if operation == "INSERT":
            columns: List[str] = []
            values: List[str] = []

            for col, val in data.items():
                # Skip primary key columns entirely
                if col == pk:
                    continue

                columns.append(col)

                # Handle FK columns - use subquery to resolve string to ID
                if col in fk_column_map and isinstance(val, str):
                    fk = fk_column_map[col]
                    ref_table = fk.references_table
                    ref_col = fk.references_column
                    lookup_col = self._get_lookup_column_for_table(
                        ref_table, schema_registry
                    )
                    escaped_val = self._escape_value(val)
                    subquery = (
                        f"(SELECT {ref_col} FROM {ref_table} "
                        f"WHERE {lookup_col} = {escaped_val} LIMIT 1)"
                    )
                    values.append(subquery)
                else:
                    values.append(self._escape_value(val))

            if not columns:
                return None

            # Get conflict key for UPSERT operation
            conflict_key: Optional[str] = self._get_conflict_key(entity_type, table)

            # Generate UPSERT if we have a valid conflict key in the columns
            if conflict_key and conflict_key in columns:
                # Build ON CONFLICT DO UPDATE SET clause
                # IMPORTANT: Exclude both conflict key AND primary key from updates
                update_cols = [c for c in columns if c != conflict_key and c != pk]
                if update_cols:
                    update_clause = ", ".join(
                        f"{col} = EXCLUDED.{col}" for col in update_cols
                    )
                    return (
                        f"INSERT INTO {entity_type} ({', '.join(columns)}) "
                        f"VALUES ({', '.join(values)}) "
                        f"ON CONFLICT ({conflict_key}) DO UPDATE SET {update_clause};"
                    )
                else:
                    # Only conflict key column, use DO NOTHING
                    return (
                        f"INSERT INTO {entity_type} ({', '.join(columns)}) "
                        f"VALUES ({', '.join(values)}) "
                        f"ON CONFLICT ({conflict_key}) DO NOTHING;"
                    )
            else:
                # No conflict key - fallback to plain INSERT
                return (
                    f"INSERT INTO {entity_type} ({', '.join(columns)}) "
                    f"VALUES ({', '.join(values)});"
                )

        elif operation == "UPDATE":
            # For UPDATE, find the record by natural key (conflict key), not PK
            conflict_key = self._get_conflict_key(entity_type, table)

            # Make a copy to avoid modifying original
            data_copy = dict(data)

            if conflict_key and conflict_key in data_copy:
                lookup_value = self._escape_value(data_copy.pop(conflict_key))
                set_clauses: List[str] = []

                for k, v in data_copy.items():
                    # Skip PK in SET clause
                    if k == pk:
                        continue
                    set_clauses.append(f"{k} = {self._escape_value(v)}")

                if set_clauses:
                    return (
                        f"UPDATE {entity_type} SET {', '.join(set_clauses)} "
                        f"WHERE {conflict_key} = {lookup_value};"
                    )
            elif pk and pk in data_copy:
                # Fallback to PK if no conflict key
                pk_value = self._escape_value(data_copy.pop(pk))
                set_clauses = []
                for k, v in data_copy.items():
                    set_clauses.append(f"{k} = {self._escape_value(v)}")

                if set_clauses:
                    return (
                        f"UPDATE {entity_type} SET {', '.join(set_clauses)} "
                        f"WHERE {pk} = {pk_value};"
                    )

        return None

    def _get_conflict_key(
        self, entity_type: str, table: Optional[TableSchema]
    ) -> Optional[str]:
        """Get the conflict key column for UPSERT operations.

        Determines which column to use for ON CONFLICT clause. Priority order:
        1. Schema-defined unique column (from table's get_conflict_key())
        2. Default mapping from DEFAULT_CONFLICT_KEYS

        Args:
            entity_type: Entity/table type name
            table: TableSchema object if available

        Returns:
            Column name to use for conflict resolution, or None if no suitable key
        """
        # Priority 1: Try schema-defined conflict key from table
        if table:
            schema_key: Optional[str] = table.get_conflict_key()
            # Check that the key is not just the primary key (which is always unique)
            if schema_key and schema_key != table.get_primary_key():
                return schema_key

        # Priority 2: Fall back to default mapping
        return DEFAULT_CONFLICT_KEYS.get(entity_type)

    def _escape_value(self, value: Any) -> str:
        """Escape a value for SQL.

        Args:
            value: Value to escape

        Returns:
            Escaped SQL string
        """
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            # Escape single quotes
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"
