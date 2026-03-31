"""LLM utilities and base classes."""

from functools import cached_property
from typing import Dict, Any

from omegaconf import DictConfig
from src.llm import LLMAPICaller


class LLMGeneratorMixin:
    """Mixin class providing LLM API caller management with fallback support."""

    def __init__(self, api_cfg: DictConfig, global_cfg: DictConfig):
        self.api_cfg = api_cfg
        self.global_cfg = global_cfg

    @cached_property
    def llm_api_caller(self) -> LLMAPICaller:
        """Primary LLM API caller."""
        return self._create_llm_caller(self.api_cfg.base)

    @cached_property
    def fallback_llm_api_caller(self) -> LLMAPICaller:
        """Fallback LLM API caller for when primary fails."""
        return self._create_llm_caller(self.api_cfg.smartest)

    def _create_llm_caller(self, cfg: DictConfig) -> LLMAPICaller:
        """Create an LLM API caller from configuration."""
        return LLMAPICaller(
            model_name=cfg.model_name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            use_custom_api=cfg.use_custom_api,
            global_cfg=self.global_cfg,
        )

    def call_llm_with_fallback(self, prompt_func, *args, **kwargs) -> Any:
        """Call LLM function with automatic fallback on failure."""
        try:
            return prompt_func(self.llm_api_caller, *args, **kwargs)
        except Exception as e:
            # Log the primary failure and try fallback
            from src.utils.logging import get_logger

            logger = get_logger(self.__class__.__module__)
            logger.warning(f"Primary LLM failed, trying fallback: {e}")
            return prompt_func(self.fallback_llm_api_caller, *args, **kwargs)
