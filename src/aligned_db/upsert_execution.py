"""Execution helpers for aligned DB upsert statements."""

from __future__ import annotations

import logging
from typing import Any, List

import tqdm

logger = logging.getLogger("AlignedDB")


def execute_upserts(
    *,
    pg_client: Any,
    upserts: List[str],
    max_retries: int = 5,
) -> None:
    """Execute upsert statements with retry logging."""
    logger.info(
        "Executing %d upsert statements (max_retries=%d)...",
        len(upserts),
        max_retries,
    )
    success_count = 0
    failure_count = 0
    retry_count = 0

    for idx, sql in enumerate(tqdm.tqdm(upserts, desc="Executing upserts")):
        for attempt in range(max_retries):
            try:
                pg_client.execute(sql)
                pg_client.conn.commit()
                success_count += 1
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    logger.error("Failed to execute: %s", sql)
                    logger.error("Error: %s", exc)
                    failure_count += 1
                else:
                    retry_count += 1
                    logger.debug("Retry %d for: %s...", attempt + 1, sql[:100])

        if (idx + 1) % 100 == 0:
            logger.info(
                "  Progress: %d/%d (%d success, %d failed)",
                idx + 1,
                len(upserts),
                success_count,
                failure_count,
            )

    success_rate = (
        f"\n  Success rate: {(success_count / len(upserts) * 100):.1f}%"
        if upserts
        else ""
    )
    logger.info(
        "Upsert execution complete:\n"
        "  Succeeded: %d\n"
        "  Failed: %d\n"
        "  Total retries: %d%s",
        success_count,
        failure_count,
        retry_count,
        success_rate,
    )
