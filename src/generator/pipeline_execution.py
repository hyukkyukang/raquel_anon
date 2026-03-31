"""Execution helpers for aligned DB pipeline stages."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry
from src.aligned_db.type_registry import AttributeType, EntityType, RelationType, TypeRegistry
from src.generator.relation_normalization import build_schema_backed_relation_registry

logger = logging.getLogger("AlignedDBPipeline")


def discover_attributes_and_relations_parallel(
    *,
    qa_pairs: List[Tuple[str, str]],
    entity_types: List[EntityType],
    attribute_discoverer: Any,
    relation_discoverer: Any,
    attribute_batch_size: int,
    max_workers: int,
) -> Tuple[Dict[str, List[AttributeType]], List[RelationType]]:
    """Run attribute and relation discovery in parallel."""
    attributes: Dict[str, List[AttributeType]] = {}
    relations: List[RelationType] = []

    def discover_attributes() -> Dict[str, List[AttributeType]]:
        return attribute_discoverer.discover_all(
            qa_pairs,
            entity_types,
            batch_size=attribute_batch_size,
            max_workers=max_workers,
        )

    def discover_relations() -> List[RelationType]:
        return relation_discoverer.discover_all(
            qa_pairs,
            entity_types,
            max_workers=max_workers,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        attr_future = executor.submit(discover_attributes)
        rel_future = executor.submit(discover_relations)

        try:
            attributes = attr_future.result()
        except Exception as exc:
            logger.error("Attribute discovery failed: %s", exc)
            attributes = {}

        try:
            relations = rel_future.result()
        except Exception as exc:
            logger.error("Relation discovery failed: %s", exc)
            relations = []

    return attributes, relations


def run_extraction_validation(
    *,
    qa_pairs: List[Tuple[str, str]],
    extractions: QAExtractionRegistry,
    schema_registry: SchemaRegistry,
    type_registry: TypeRegistry,
    extraction_validator: Any,
    validation_max_iterations: int,
    validation_coverage_threshold: float,
    extraction_max_concurrency: int,
) -> QAExtractionRegistry:
    """Run fact-based extraction validation with async parallelism."""
    return asyncio.run(
        run_extraction_validation_async(
            qa_pairs=qa_pairs,
            extractions=extractions,
            schema_registry=schema_registry,
            type_registry=type_registry,
            extraction_validator=extraction_validator,
            validation_max_iterations=validation_max_iterations,
            validation_coverage_threshold=validation_coverage_threshold,
            extraction_max_concurrency=extraction_max_concurrency,
        )
    )

async def run_extraction_validation_async(
    *,
    qa_pairs: List[Tuple[str, str]],
    extractions: QAExtractionRegistry,
    schema_registry: SchemaRegistry,
    type_registry: TypeRegistry,
    extraction_validator: Any,
    validation_max_iterations: int,
    validation_coverage_threshold: float,
    extraction_max_concurrency: int,
) -> QAExtractionRegistry:
    """Async implementation of fact-based extraction validation."""
    max_iterations = validation_max_iterations
    coverage_threshold = validation_coverage_threshold
    max_concurrency = extraction_max_concurrency

    logger.info(
        "  Running fact-based validation (threshold=%0.f%%, max_iterations=%d, concurrency=%d)",
        coverage_threshold * 100,
        max_iterations,
        max_concurrency,
    )
    schema_backed_type_registry = build_schema_backed_relation_registry(
        schema_registry,
        type_registry,
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    executor = ThreadPoolExecutor(max_workers=max_concurrency)

    try:
        for iteration in range(max_iterations):
            logger.info("  Validation iteration %d/%d", iteration + 1, max_iterations)

            async def extract_facts_for_qa(
                qa_idx: int,
                question: str,
                answer: str,
            ) -> Tuple[int, List[Any], Any]:
                extraction = extractions.get(qa_idx)
                if extraction is None:
                    return qa_idx, [], None

                async with semaphore:
                    loop = asyncio.get_running_loop()
                    facts = await loop.run_in_executor(
                        executor,
                        extraction_validator.extract_answer_facts,
                        question,
                        answer,
                    )
                return qa_idx, facts, extraction

            fact_tasks = [
                extract_facts_for_qa(idx, question, answer)
                for idx, (question, answer) in enumerate(qa_pairs)
            ]
            fact_results = await asyncio.gather(*fact_tasks)

            validation_results = []
            total_facts = 0
            total_found = 0

            for qa_idx, facts, extraction in fact_results:
                if extraction is None:
                    continue

                total_facts += len(facts)

                if not facts:
                    extraction.validation_status = "valid"
                    validation_results.append((qa_idx, 1.0, []))
                    continue

                coverage = extraction_validator.check_fact_coverage(facts, extraction)
                total_found += len(coverage.found_facts)
                validation_results.append(
                    (qa_idx, coverage.coverage_score, coverage.missing_facts)
                )

                if coverage.coverage_score >= coverage_threshold:
                    extraction.validation_status = "valid"
                else:
                    extraction.validation_status = "invalid"
                    extraction.missing_facts = [
                        str(fact) for fact in coverage.missing_facts
                    ]

            avg_coverage = total_found / max(total_facts, 1)
            logger.info(
                "    Coverage: %.1f%% (%d/%d facts found)",
                avg_coverage * 100,
                total_found,
                total_facts,
            )

            if avg_coverage >= coverage_threshold:
                logger.info("    Coverage meets threshold - validation complete")
                break

            if iteration < max_iterations - 1:
                gaps_to_fix = [
                    (qa_idx, missing_facts)
                    for qa_idx, coverage_score, missing_facts in validation_results
                    if missing_facts and coverage_score < coverage_threshold
                ]

                if gaps_to_fix:

                    async def extract_gap_for_qa(
                        qa_idx: int,
                        missing_facts: List[Any],
                    ) -> Tuple[int, Any]:
                        question, answer = qa_pairs[qa_idx]
                        async with semaphore:
                            loop = asyncio.get_running_loop()
                            gap_extraction = await loop.run_in_executor(
                                executor,
                                extraction_validator.extract_missing_facts,
                                question,
                                answer,
                                missing_facts,
                                schema_backed_type_registry,
                            )
                        return qa_idx, gap_extraction

                    gap_tasks = [
                        extract_gap_for_qa(qa_idx, missing_facts)
                        for qa_idx, missing_facts in gaps_to_fix
                    ]
                    gap_results = await asyncio.gather(*gap_tasks)

                    gaps_fixed = 0
                    for qa_idx, gap_extraction in gap_results:
                        if gap_extraction and not gap_extraction.is_empty:
                            extraction = extractions.get(qa_idx)
                            if extraction:
                                extraction_validator.merge_extractions(
                                    extraction,
                                    gap_extraction,
                                )
                                gaps_fixed += 1

                    logger.info(
                        "    Re-extracted for %d QA pairs with gaps",
                        gaps_fixed,
                    )
        else:
            logger.warning(
                "    Validation did not converge after %d iterations",
                max_iterations,
            )
    finally:
        executor.shutdown(wait=False)

    valid_count = sum(
        1 for extraction in extractions if extraction.validation_status == "valid"
    )
    logger.info(
        "  Validation complete: %d/%d valid",
        valid_count,
        extractions.count,
    )
    return extractions


def run_verification(
    *,
    qa_pairs: List[Tuple[str, str]],
    qa_extractions: QAExtractionRegistry,
    schema_registry: SchemaRegistry,
    round_trip_verifier: Any,
    pg_client: Any,
    verification_max_iterations: int,
) -> None:
    """Run round-trip verification iterations for the pipeline."""
    for iteration in range(verification_max_iterations):
        logger.info("  Verification iteration %d", iteration + 1)

        results = round_trip_verifier.verify_all(qa_pairs, pg_client, schema_registry)
        failed_results = [result for result in results if result.needs_fix]

        if not failed_results:
            logger.info("  All QA pairs verified successfully!")
            break

        logger.info("  %d QA pairs need fixes", len(failed_results))

        fixes = round_trip_verifier.generate_fixes_with_qa_mapping(
            failed_results,
            schema_registry,
            qa_extractions=qa_extractions,
            pg_client=pg_client,
        )

        if not fixes:
            logger.warning("  No fixes generated, stopping verification")
            break

        logger.info("  Generated %d fix statements", len(fixes))

        success_count = 0
        for fix in fixes:
            try:
                pg_client.execute(fix)
                pg_client.conn.commit()
                success_count += 1
            except Exception as exc:
                logger.debug("  Fix failed: %s", exc)

        logger.info("  Executed %d/%d fixes", success_count, len(fixes))
    else:
        logger.warning(
            "  Verification incomplete after %d iterations",
            verification_max_iterations,
        )
