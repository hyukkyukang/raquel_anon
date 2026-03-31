from typing import List

from omegaconf import DictConfig

from src.generator.sql import SQLGenerator


class TextToSQLTranslator:
    def __init__(self, cfg: DictConfig, api_cfg: DictConfig, global_cfg: DictConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.sql_generator = SQLGenerator(api_cfg, global_cfg)

    def __call__(self, schema: str, insert_queries: List[str], question: str) -> str:
        return self.sql_generator.text_to_sql(schema, insert_queries, question)
