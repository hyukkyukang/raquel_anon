import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import hkkang_utils.pg as pg_utils
import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.generator.text_to_sql_translator import TextToSQLTranslator
from src.llm import LLMAPICaller
from src.prompt.registry import RESULT_TO_TEXT_PROMPT_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


def load_qa_dataset(cfg: DictConfig, name: str) -> List[Tuple[str, str]]:
    """Load the dataset with validation."""
    try:
        dataset = load_dataset(
            path=cfg.dataset.huggingface_path,
            name=name,
            split=cfg.dataset.split,
        )

        # Validate dataset structure
        required_fields = ["question", "answer"]
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Dataset missing required field: {field}")

        return [(item["question"], item["answer"]) for item in dataset]
    except Exception as e:
        logger.error(f"Failed to load dataset '{name}': {str(e)}")
        raise


def judge_equivalence(llm_api: LLMAPICaller, annotated: str, predicted: str) -> bool:
    """Judge if two answers are semantically equivalent using LLM."""
    try:
        system_instruction = (
            "You are a helpful assistant for evaluating answer equivalence. "
            "Your task is to determine if two answers are semantically equivalent, "
            "meaning they convey the same information even if worded differently."
        )
        user_prompt = (
            "Given the following two answers, are they semantically equivalent? "
            "Answer only with 'yes' or 'no'.\n\n"
            f"Annotated Answer:\n{annotated}\n\nPredicted Answer:\n{predicted}"
        )
        response = llm_api._call_custom_api(
            system_instruction, user_prompt, temperature=0.0
        )
        return "yes" in response.lower()
    except Exception as e:
        logger.error(f"Error in judge_equivalence: {str(e)}")
        return False


def format_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format database result to handle datetime objects."""
    import datetime

    if isinstance(result, dict):
        formatted_result = {}
        for key, value in result.items():
            if isinstance(value, datetime.date):
                formatted_result[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, datetime.datetime):
                formatted_result[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_result[key] = value
        return formatted_result
    return result


def format_sql_results_to_answer(results: List[Dict[str, Any]]) -> str:
    """Convert SQL results to a readable answer format."""
    if not results:
        return "No results found."

    try:
        if len(results) == 1:
            # Format single result more cleanly
            result = results[0]
            if isinstance(result, dict):
                return ", ".join([f"{k}: {v}" for k, v in result.items()])
            else:
                return str(result)
        else:
            # Format multiple results
            formatted_results = []
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    formatted_results.append(f"Row {i+1}: {result}")
                else:
                    formatted_results.append(f"Row {i+1}: {result}")
            return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error formatting SQL results: {str(e)}")
        return str(results)


def validate_sql_query(sql_query: str) -> Tuple[bool, str]:
    """Validate SQL query for safety and basic syntax."""
    if not sql_query or not sql_query.strip():
        return False, "Empty SQL query"

    sql_lower = sql_query.strip().lower()

    # Check for dangerous operations
    dangerous_operations = [
        "drop",
        "delete",
        "truncate",
        "alter",
        "update",
        "insert",
        "create",
        "grant",
        "revoke",
        "backup",
        "restore",
    ]

    for operation in dangerous_operations:
        if sql_lower.startswith(operation):
            return False, f"Dangerous SQL operation detected: {operation.upper()}"

    # Basic syntax validation
    if not sql_lower.startswith("select"):
        return False, "Only SELECT queries are allowed"

    return True, ""


def process_qa_pair(
    i: int,
    question: str,
    annotated_answer: str,
    text_to_sql_translator,
    pg_client,
    llm_api,
    result_to_text_prompt_cls,
    schema: str,
    insert_queries: List[str],
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Process a single QA pair with proper error handling."""
    test_result = {
        "index": i,
        "question": question,
        "annotated_answer": annotated_answer,
        "sql_query": None,
        "execution_result": None,
        "predicted_answer": None,
        "is_equivalent": False,
        "error": None,
        "error_step": None,  # Track where error occurred
    }

    try:
        # Step 1: Convert question to SQL
        logger.debug(f"Converting question {i} to SQL...")
        sql_query: str = text_to_sql_translator(
            schema=schema, insert_queries=insert_queries, question=question
        )

        # Validate SQL query
        is_valid, validation_error = validate_sql_query(sql_query)
        if not is_valid:
            test_result["error"] = validation_error
            test_result["error_step"] = "sql_validation"
            return test_result

        test_result["sql_query"] = sql_query

        # Step 2: Execute SQL query with timeout
        logger.debug(f"Executing SQL query {i}...")
        start_time = time.time()

        # Set a timeout for database operations
        results = pg_client.execute_and_fetchall_with_col_names(sql_query)

        if time.time() - start_time > timeout_seconds:
            test_result["error"] = (
                f"Database query timed out after {timeout_seconds} seconds"
            )
            test_result["error_step"] = "sql_execution"
            return test_result

        results = [format_result(r) for r in results]
        test_result["execution_result"] = results

        # Step 3: Convert SQL result to natural language answer
        logger.debug(f"Converting SQL result to text for question {i}...")
        answer_normal = format_sql_results_to_answer(results)
        prompt = result_to_text_prompt_cls(question=question, result=answer_normal)

        try:
            predicted_answer = llm_api(prompt, prefix="test_qa_alignment")
            test_result["predicted_answer"] = predicted_answer
        except Exception as e:
            test_result["error"] = f"LLM processing failed: {str(e)}"
            test_result["error_step"] = "llm_processing"
            return test_result

        # Step 4: Compare with original answer
        logger.debug(f"Comparing answers for question {i}...")
        try:
            is_equivalent = judge_equivalence(
                llm_api, annotated_answer, predicted_answer
            )
            test_result["is_equivalent"] = is_equivalent
        except Exception as e:
            test_result["error"] = f"Answer comparison failed: {str(e)}"
            test_result["error_step"] = "answer_comparison"
            return test_result

        return test_result

    except Exception as e:
        error_msg = f"Unexpected error processing question {i}: {str(e)}"
        logger.error(error_msg)
        test_result["error"] = error_msg
        test_result["error_step"] = "unknown"
        return test_result


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    """Test whether all information in QA pairs is correctly applied to the aligned DB."""

    # Initialize components
    logger.info("Initializing components...")

    text_to_sql_translator = None
    pg_client = None
    llm_api = None

    try:
        # Text-to-SQL translator
        text_to_sql_translator = TextToSQLTranslator(cfg.model.translator, cfg.llm, cfg)

        # Database client for aligned DB
        pg_client = pg_utils.PostgresConnector(
            db_id=cfg.database.db_id,
            user_id=cfg.database.user_id,
            passwd=cfg.database.passwd,
            host=cfg.database.host,
            port=cfg.database.port,
        )

        # LLM API for result-to-text conversion and equivalence checking
        llm_api = LLMAPICaller(
            model_name=cfg.llm.base.model_name,
            max_tokens=cfg.llm.base.max_tokens,
            temperature=cfg.llm.base.temperature,
            use_custom_api=cfg.llm.base.use_custom_api,
            global_cfg=cfg,
        )

        # Result-to-text prompt class
        result_to_text_prompt_cls = RESULT_TO_TEXT_PROMPT_REGISTRY["default"]

        # Load schema and insert queries with proper encoding
        schema_path: str = os.path.join(
            cfg.project_path, cfg.paths.data_dir, cfg.paths.schema
        )
        inserts_path: str = os.path.join(
            cfg.project_path, cfg.paths.data_dir, cfg.paths.cleaned_inserts
        )

        # Check if files exist
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        if not os.path.exists(inserts_path):
            raise FileNotFoundError(f"Inserts file not found: {inserts_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema: str = f.read()

        with open(inserts_path, "r", encoding="utf-8") as f:
            insert_queries: List[str] = [line.strip() for line in f if line.strip()]

        # Load QA pairs
        logger.info("Loading QA pairs from dataset...")
        retain_qa_pairs = load_qa_dataset(cfg, "retain90")
        forget_qa_pairs = load_qa_dataset(cfg, "forget10")

        # Use a subset for testing (configurable)
        sample_num: int = cfg.model.aligned_db.sample_num
        retain_sample = min(sample_num, len(retain_qa_pairs))
        forget_sample = min(sample_num, len(forget_qa_pairs))
        full_sampled_qa_pairs = (
            retain_qa_pairs[:retain_sample] + forget_qa_pairs[:forget_sample]
        )

        logger.info(f"Testing with {len(full_sampled_qa_pairs)} QA pairs")

        # Test results tracking with improved error classification
        test_results = {
            "total_tested": 0,
            "successful_conversions": 0,
            "successful_executions": 0,
            "correct_answers": 0,
            "failed_conversions": 0,
            "failed_executions": 0,
            "failed_llm_processing": 0,
            "failed_answer_comparison": 0,
            "detailed_results": [],
        }

        logger.info("Starting QA alignment testing...")

        # Process QA pairs with improved error handling
        for i, (question, annotated_answer) in enumerate(
            tqdm(full_sampled_qa_pairs, desc="Testing QA pairs")
        ):
            test_result = process_qa_pair(
                i=i,
                question=question,
                annotated_answer=annotated_answer,
                text_to_sql_translator=text_to_sql_translator,
                pg_client=pg_client,
                llm_api=llm_api,
                result_to_text_prompt_cls=result_to_text_prompt_cls,
                schema=schema,
                insert_queries=insert_queries,
                timeout_seconds=getattr(cfg, "query_timeout_seconds", 30),
            )

            test_results["total_tested"] += 1

            # Improved error classification based on error_step
            if test_result["error"]:
                error_step = test_result.get("error_step", "unknown")
                if error_step == "sql_validation":
                    test_results["failed_conversions"] += 1
                elif error_step == "sql_execution":
                    test_results["failed_executions"] += 1
                elif error_step == "llm_processing":
                    test_results["failed_llm_processing"] += 1
                elif error_step == "answer_comparison":
                    test_results["failed_answer_comparison"] += 1
                else:
                    # Default classification based on what was completed
                    if test_result["sql_query"] is None:
                        test_results["failed_conversions"] += 1
                    elif test_result["execution_result"] is None:
                        test_results["failed_executions"] += 1
                    elif test_result["predicted_answer"] is None:
                        test_results["failed_llm_processing"] += 1
                    else:
                        test_results["failed_answer_comparison"] += 1
            else:
                # Successful processing
                test_results["successful_conversions"] += 1
                test_results["successful_executions"] += 1
                if test_result["is_equivalent"]:
                    test_results["correct_answers"] += 1

            test_results["detailed_results"].append(test_result)

            # Log progress every 10 questions
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(full_sampled_qa_pairs)} questions")

        # Calculate and display results
        logger.info("Testing completed. Calculating results...")

        total = test_results["total_tested"]
        conversion_rate = (
            test_results["successful_conversions"] / total if total > 0 else 0
        )
        execution_rate = (
            test_results["successful_executions"] / total if total > 0 else 0
        )
        accuracy = test_results["correct_answers"] / total if total > 0 else 0

        logger.info("=" * 60)
        logger.info("QA ALIGNMENT TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total QA pairs tested: {total}")
        logger.info(
            f"Successful SQL conversions: {test_results['successful_conversions']} ({conversion_rate:.2%})"
        )
        logger.info(
            f"Successful SQL executions: {test_results['successful_executions']} ({execution_rate:.2%})"
        )
        logger.info(
            f"Correct answers: {test_results['correct_answers']} ({accuracy:.2%})"
        )
        logger.info(f"Failed conversions: {test_results['failed_conversions']}")
        logger.info(f"Failed executions: {test_results['failed_executions']}")
        logger.info(f"Failed LLM processing: {test_results['failed_llm_processing']}")
        logger.info(
            f"Failed answer comparison: {test_results['failed_answer_comparison']}"
        )
        logger.info("=" * 60)

        # Save detailed results to file
        results_path = os.path.join(
            cfg.project_path, cfg.paths.data_dir, "qa_alignment_test_results.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"Detailed results saved to: {results_path}")

        # Print some examples of failures for debugging
        failed_examples = [
            r
            for r in test_results["detailed_results"]
            if r["error"] or not r["is_equivalent"]
        ]
        if failed_examples:
            logger.info(f"\nSample failures ({min(5, len(failed_examples))} examples):")
            for i, example in enumerate(failed_examples[:5]):
                logger.info(f"\nFailure {i+1}:")
                logger.info(f"Question: {example['question']}")
                logger.info(f"Expected: {example['annotated_answer']}")
                if example["predicted_answer"]:
                    logger.info(f"Got: {example['predicted_answer']}")
                if example["error"]:
                    logger.info(f"Error: {example['error']}")
                if example.get("error_step"):
                    logger.info(f"Error step: {example['error_step']}")

        return test_results

    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if pg_client:
            try:
                pg_client.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logger.info("QA alignment testing completed!")
