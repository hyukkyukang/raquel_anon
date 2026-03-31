"""Verification and fix-application helpers for aligned DB builds."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import hkkang_utils.pg as pg_utils

from src.aligned_db.qa_extraction import QAExtractionRegistry
from src.aligned_db.schema_registry import SchemaRegistry

logger = logging.getLogger("AlignedDB")


def execute_fixes(
    *,
    pg_client: pg_utils.PostgresConnector,
    fixes: List[str],
) -> int:
    """Execute verification fix statements and summarize outcomes."""
    logger.info("Executing %d fix statements...", len(fixes))
    success_count = 0
    failure_count = 0
    duplicate_skipped = 0
    fk_violation_skipped = 0

    for idx, sql in enumerate(fixes, start=1):
        try:
            pg_client.execute(sql)
            pg_client.conn.commit()
            success_count += 1
            logger.info("  [%d/%d] Applied fix successfully", idx, len(fixes))
        except Exception as exc:
            error_str = str(exc).lower()

            if "duplicate key" in error_str:
                duplicate_skipped += 1
                logger.debug("  [%d/%d] Skipped (duplicate key exists)", idx, len(fixes))
                success_count += 1
            elif "violates foreign key constraint" in error_str:
                fk_violation_skipped += 1
                logger.debug(
                    "  [%d/%d] Skipped (FK reference not found)",
                    idx,
                    len(fixes),
                )
                failure_count += 1
            else:
                failure_count += 1
                logger.warning(
                    "  [%d/%d] Failed to execute fix: %s",
                    idx,
                    len(fixes),
                    f"{sql[:200]}{'...' if len(sql) > 200 else ''}",
                )
                logger.warning("  Error: %s", exc)

    summary_parts: List[str] = [f"{success_count} succeeded"]
    if duplicate_skipped > 0:
        summary_parts.append(f"{duplicate_skipped} duplicate-skipped")
    if fk_violation_skipped > 0:
        summary_parts.append(f"{fk_violation_skipped} FK-skipped")
    if failure_count > 0:
        summary_parts.append(f"{failure_count - fk_violation_skipped} failed")

    logger.info("Fix execution complete: %s", ", ".join(summary_parts))
    return success_count


def run_iterative_verification(
    *,
    verifier: Any,
    pg_client: pg_utils.PostgresConnector,
    qa_pairs: List[Tuple[str, str]],
    schema_registry: SchemaRegistry,
    qa_extractions: Optional[QAExtractionRegistry] = None,
    enable_iterative: bool = True,
    max_iterations: int = 3,
    use_dedicated_prompt: bool = True,
) -> Tuple[List[Any], int]:
    """Run iterative round-trip verification with optional fix application."""
    total_fixes_applied = 0

    logger.info("Running initial verification...")
    all_results = verifier.verify_all(qa_pairs, pg_client, schema_registry)
    logger.info("  Verified %d QA pairs", len(all_results))

    if not enable_iterative:
        fixes = verifier.generate_fixes(
            [result for result in all_results if result.needs_fix],
            schema_registry,
            pg_client,
        )
        if fixes:
            logger.info("Applying %d fixes...", len(fixes))
            total_fixes_applied = execute_fixes(pg_client=pg_client, fixes=fixes)
        return all_results, total_fixes_applied

    for iteration in range(1, max_iterations + 1):
        failed_results = [result for result in all_results if result.needs_fix]
        failed_indices = [result.qa_index for result in failed_results]

        if not failed_results:
            logger.info(
                "All QA pairs verified successfully after %d iterations",
                iteration - 1,
            )
            break

        logger.info(
            "\nIteration %d/%d: %d/%d QA pairs need fixes",
            iteration,
            max_iterations,
            len(failed_results),
            len(all_results),
        )

        if qa_extractions and use_dedicated_prompt:
            fixes = verifier.generate_fixes_with_qa_mapping(
                failed_results,
                schema_registry,
                qa_extractions=qa_extractions,
                pg_client=pg_client,
            )
        elif use_dedicated_prompt:
            fixes = verifier.generate_fixes_with_dedicated_prompt(
                failed_results,
                schema_registry,
                pg_client,
            )
        else:
            fixes = verifier.generate_fixes(
                failed_results,
                schema_registry,
                pg_client,
            )

        if fixes:
            logger.info("  Applying %d fixes...", len(fixes))
            applied = execute_fixes(pg_client=pg_client, fixes=fixes)
            total_fixes_applied += applied
        else:
            logger.info("  No valid fixes generated")

        failed_qa_pairs = [qa_pairs[idx] for idx in failed_indices]
        logger.info("  Re-verifying %d QA pairs...", len(failed_qa_pairs))
        new_results = verifier.verify_all(failed_qa_pairs, pg_client, schema_registry)

        for new_result, orig_idx in zip(new_results, failed_indices):
            new_result.qa_index = orig_idx
            all_results[orig_idx] = new_result

        still_failing = sum(1 for result in all_results if result.needs_fix)
        improved = len(failed_results) - still_failing
        logger.info(
            "  Iteration %d complete: %d QA pairs fixed, %d still need fixes",
            iteration,
            improved,
            still_failing,
        )
    else:
        logger.warning(
            "  Verification incomplete after %d iterations",
            max_iterations,
        )

    final_failed = sum(1 for result in all_results if result.needs_fix)
    inconsistent = sum(1 for result in all_results if result.has_qa_inconsistency)
    passed = len(all_results) - final_failed - inconsistent
    avg_score = (
        sum(result.similarity_score for result in all_results) / len(all_results)
        if all_results
        else 0
    )
    logger.info(
        "\nIterative verification complete:\n"
        "  Total fixes applied: %d\n"
        "  Final results: %d passed, %d need fixes, %d inconsistent QA\n"
        "  Average similarity score: %.3f",
        total_fixes_applied,
        passed,
        final_failed,
        inconsistent,
        avg_score,
    )

    return all_results, total_fixes_applied
