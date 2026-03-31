import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import logging
from typing import List, Tuple

import hkkang_utils.pg as pg_utils
import hydra
from datasets import load_dataset
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from src.generator.text_to_sql_translator import TextToSQLTranslator
from src.llm import LLMAPICaller
from src.prompt.registry import RESULT_TO_TEXT_PROMPT_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)


def load_qa_dataset(cfg: DictConfig, name: str) -> List[Tuple[str, str]]:
    dataset = load_dataset(
        path=cfg.dataset.huggingface_path,
        name=name,
        split=cfg.dataset.split,
    )
    return [(item["question"], item["answer"]) for item in dataset]


def judge_equivalence(llm_api: LLMAPICaller, annotated: str, predicted: str) -> bool:
    system_instruction = (
        "You are a helpful assistant for evaluating answer equivalence."
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


def format_result(result):
    # If there are datetime.date objects, convert them to strings
    import datetime

    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, datetime.date):
                result[key] = value.strftime("%Y-%m-%d")
    return result


def format_sql_results_to_answer(results):
    if not results:
        return "No results found."
    if len(results) == 1:
        return str(results[0])
    return str(results)


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name=DEFAULT_CONFIG_FILE_NAME,
)
def main(cfg: DictConfig) -> None:
    # Load QA pairs
    retain_qa_pairs = load_qa_dataset(cfg, "retain90")
    forget_qa_pairs = load_qa_dataset(cfg, "forget10")
    full_sampled_qa_pairs = retain_qa_pairs[:50] + forget_qa_pairs[:50]

    # Instantiate TextToSQLTranslator
    text_to_sql_translator = TextToSQLTranslator(cfg.model.synthesizer, cfg)

    # Instantiate DB client
    pg_client = pg_utils.PostgresConnector(
        db_id=cfg.database.db_id,
        user_id=cfg.database.user_id,
        passwd=cfg.database.passwd,
        host=cfg.database.host,
        port=cfg.database.port,
    )
    # Instantiate LLM API for result-to-text and equivalence
    llm_api = LLMAPICaller(
        model_name=cfg.llm.api_cfg.model_name,
        max_tokens=cfg.llm.api_cfg.max_tokens,
        temperature=cfg.llm.api_cfg.temperature,
        use_custom_api=cfg.llm.api_cfg.use_custom_api,
        global_cfg=cfg,
    )
    result_to_text_prompt_cls = RESULT_TO_TEXT_PROMPT_REGISTRY["default"]

    # Load schema for synthesizer
    with open(cfg.paths.schema, "r") as f:
        schema: str = f.read()
    # Load insert queries for synthesizer
    with open(cfg.paths.cleaned_inserts, "r") as f:
        insert_queries: List[str] = [line.strip() for line in f if line.strip()]

    correct_num = 0
    total = 0
    for i, (question, annotated_answer) in enumerate(full_sampled_qa_pairs):
        # 1. Generate SQL from question
        sql_query = text_to_sql_translator(
            schema=schema, insert_queries=insert_queries, question=question
        )
        if (
            not sql_query
            or not sql_query.strip()
            or sql_query.strip()
            .lower()
            .startswith(("drop", "delete", "truncate", "alter"))
        ):
            logger.error(f"No valid SQL generated for Q{i}")
            continue

        # 2. Execute SQL
        results = pg_client.execute_and_fetchall_with_col_names(sql_query)
        results = [format_result(r) for r in results]

        # 3. Convert SQL result to text
        try:
            answer_normal = format_sql_results_to_answer(results)
            prompt = result_to_text_prompt_cls(question=question, result=answer_normal)
            predicted_answer = llm_api(prompt, prefix="evaluate_aligned_db")
        except Exception as e:
            logger.error(f"Failed to convert SQL result to text for Q{i}: {e}")
            continue

        # 4. Judge equivalence
        try:
            is_correct = judge_equivalence(llm_api, annotated_answer, predicted_answer)
        except Exception as e:
            logger.error(f"Failed to judge equivalence for Q{i}: {e}")
            is_correct = False

        if is_correct:
            correct_num += 1
        total += 1
        logger.info(f"Q{i}: {'Correct' if is_correct else 'Incorrect'}")

    accuracy = correct_num / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logging.info("Done!")
