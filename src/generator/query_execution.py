"""Shared query-execution helpers for aligned/null database comparison."""

from __future__ import annotations

import datetime
import decimal
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type

import hkkang_utils.pg as pg_utils
import tqdm

logger = logging.getLogger(__name__)

# Keep low to avoid DB connection exhaustion. Each worker keeps one aligned/null pair.
DEFAULT_MAX_WORKERS = 8
_THREAD_LOCAL_STATE = threading.local()


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-JSON-native DB values to stable string forms in a result dict."""
    formatted: Dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            formatted[key] = value.isoformat()
        elif isinstance(value, decimal.Decimal):
            formatted[key] = str(value)
        else:
            formatted[key] = value
    return formatted


def canonicalize_sql_results(results: List[Dict[str, Any]]) -> List[str]:
    """Canonicalize a result set for denotation comparison."""
    canonical_rows = [
        json.dumps(format_result(result), sort_keys=True, default=str, ensure_ascii=False)
        for result in results
    ]
    canonical_rows.sort()
    return canonical_rows


def format_sql_results_to_answer(results: List[Dict[str, Any]]) -> str:
    """Convert SQL query results to the string form used by result-to-text prompts."""
    if not results:
        return "No results found."
    if len(results) == 1:
        return str(results[0])
    return str(results)


def _db_connection_key(db_config: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build a stable cache key for thread-local connector reuse."""
    return (
        db_config["db_id"],
        db_config["null_db_id"],
        db_config["user_id"],
        db_config["passwd"],
        db_config["host"],
        db_config["port"],
        db_config["null_port"],
    )


def _close_thread_local_connector(connector: Any) -> None:
    """Best-effort connector cleanup when thread-local config changes."""
    if connector is None:
        return

    close = getattr(connector, "close", None)
    if callable(close):
        try:
            close()
            return
        except Exception:
            logger.debug("Failed to close connector via close()", exc_info=True)

    conn = getattr(connector, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            logger.debug("Failed to close connector.conn", exc_info=True)


def _build_pg_client(db_id: str, db_config: Dict[str, Any], port_key: str) -> Any:
    """Create a PostgreSQL connector for one database target."""
    return pg_utils.PostgresConnector(
        db_id=db_id,
        user_id=db_config["user_id"],
        passwd=db_config["passwd"],
        host=db_config["host"],
        port=db_config[port_key],
    )


def _get_thread_local_pg_clients(
    db_config: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Reuse one aligned/null connector pair per worker thread."""
    connection_key = _db_connection_key(db_config)
    current_key = getattr(_THREAD_LOCAL_STATE, "connection_key", None)

    if current_key != connection_key:
        _close_thread_local_connector(
            getattr(_THREAD_LOCAL_STATE, "pg_client_normal", None)
        )
        _close_thread_local_connector(
            getattr(_THREAD_LOCAL_STATE, "pg_client_null", None)
        )
        _THREAD_LOCAL_STATE.pg_client_normal = _build_pg_client(
            db_id=db_config["db_id"],
            db_config=db_config,
            port_key="port",
        )
        _THREAD_LOCAL_STATE.pg_client_null = _build_pg_client(
            db_id=db_config["null_db_id"],
            db_config=db_config,
            port_key="null_port",
        )
        _THREAD_LOCAL_STATE.connection_key = connection_key

    return _THREAD_LOCAL_STATE.pg_client_normal, _THREAD_LOCAL_STATE.pg_client_null


def process_single_query(
    index: int,
    sql_query: str,
    translated_query: str,
    db_config: Dict[str, Any],
    llm_api: Any,
    result_to_text_prompt_cls: Type[Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """Execute one SQL query on aligned/null DBs and convert the result to QA form."""
    pg_client_normal, pg_client_null = _get_thread_local_pg_clients(db_config)

    try:
        results_normal = pg_client_normal.execute_and_fetchall_with_col_names(sql_query)
    except Exception as exc:
        logger.error("Error executing query on Aligned DB %d: %s", index + 1, exc)
        return (index, "error", None)

    try:
        results_null = pg_client_null.execute_and_fetchall_with_col_names(sql_query)
    except Exception as exc:
        logger.error("Error executing query on null DB %d: %s", index + 1, exc)
        return (index, "error", None)

    if results_normal:
        results_normal = [format_result(result) for result in results_normal]
    if results_null:
        results_null = [format_result(result) for result in results_null]

    answer_normal = format_sql_results_to_answer(results_normal)
    answer_null = format_sql_results_to_answer(results_null)

    try:
        prompt_normal = result_to_text_prompt_cls(
            question=translated_query,
            result=answer_normal,
        )
        llm_sentence_normal = llm_api(prompt_normal, prefix="normalize_answer_to_nl")
    except Exception as exc:
        logger.error("Error converting result to text for query %d: %s", index + 1, exc)
        return (index, "error", None)

    qa_dict: Dict[str, Any] = {
        "question": translated_query,
        "answer": llm_sentence_normal,
    }
    if metadata is not None:
        qa_dict["metadata"] = metadata

    category = (
        "unaffected"
        if canonicalize_sql_results(results_normal)
        == canonicalize_sql_results(results_null)
        else "affected"
    )
    return (index, category, qa_dict)


def execute_queries(
    *,
    sql_queries: List[str],
    translated_queries: List[str],
    db_config: Dict[str, Any],
    llm_api: Any,
    result_to_text_prompt_cls: Type[Any],
    metadata_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Execute SQL queries in parallel and split them into affected/unaffected QA sets."""
    metadata_by_index = metadata_by_index or {}
    affected_data: List[Dict[str, Any]] = []
    unaffected_data: List[Dict[str, Any]] = []
    skip_count = 0

    logger.info(
        "Executing %d queries with %d parallel workers...",
        len(sql_queries),
        max_workers,
    )

    results_map: Dict[int, Tuple[str, Optional[Dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_query,
                index,
                sql_query,
                translated_query,
                db_config,
                llm_api,
                result_to_text_prompt_cls,
                metadata_by_index.get(index),
            ): index
            for index, (sql_query, translated_query) in enumerate(
                zip(sql_queries, translated_queries)
            )
        }

        for future in tqdm.tqdm(
            as_completed(futures),
            desc="Executing queries",
            total=len(sql_queries),
        ):
            try:
                index, category, qa_dict = future.result()
                results_map[index] = (category, qa_dict)
            except Exception as exc:
                index = futures[future]
                logger.error("Error processing query %d: %s", index + 1, exc)
                results_map[index] = ("error", None)

    for index in range(len(sql_queries)):
        category, qa_dict = results_map.get(index, ("error", None))
        if category == "error" or qa_dict is None:
            skip_count += 1
        elif category == "affected":
            affected_data.append(qa_dict)
        else:
            unaffected_data.append(qa_dict)

    return affected_data, unaffected_data, skip_count
