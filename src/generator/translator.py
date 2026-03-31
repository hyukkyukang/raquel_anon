from functools import cached_property

from omegaconf import DictConfig

from src.llm import LLMAPICaller
from src.prompt import Prompt
from src.prompt.registry import SQL_TO_TEXT_PROMPT_REGISTRY


class SQLToTextTranslator:
    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig):
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg

    def __call__(self, sql_query: str) -> str:
        prompt: Prompt = self.sql_to_text_prompt(sql_query)
        text_query: str = self.sql_generator(prompt, prefix="sql_to_text_translation")
        return text_query

    @cached_property
    def sql_generator(self) -> LLMAPICaller:
        """Get or create the SQL generator instance."""
        return LLMAPICaller(
            model_name=self.api_cfg.model_name,
            max_tokens=self.api_cfg.max_tokens,
            temperature=self.api_cfg.temperature,
            use_custom_api=self.api_cfg.use_custom_api,
            global_cfg=self.global_cfg,
        )

    def sql_to_text_prompt(self, sql_query: str) -> Prompt:
        """Generate a prompt for query synthesis."""
        sql_to_text_prompt_name = self.global_cfg.prompt.sql_to_text
        return SQL_TO_TEXT_PROMPT_REGISTRY[sql_to_text_prompt_name](
            sql_query=sql_query,
        )
