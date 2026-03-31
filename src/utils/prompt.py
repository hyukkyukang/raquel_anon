"""Prompt utilities and registry helpers."""

from functools import cached_property
from typing import Dict, Any, Type
from omegaconf import DictConfig
from src.prompt import Prompt


class PromptRegistryMixin:
    """Mixin class providing simplified prompt registry access."""
    
    def __init__(self, global_cfg: DictConfig):
        self.global_cfg = global_cfg
    
    def _get_prompt_from_registry(self, registry: Dict[str, Type[Prompt]], config_key: str) -> Type[Prompt]:
        """Get a prompt class from registry using config key."""
        prompt_name = getattr(self.global_cfg.prompt, config_key)
        return registry[prompt_name]
    
    def create_cached_prompt_property(self, registry: Dict[str, Type[Prompt]], config_key: str):
        """Create a cached property for a prompt from registry."""
        def prompt_property(self):
            return self._get_prompt_from_registry(registry, config_key)
        return cached_property(prompt_property)


def simplify_prompt_properties(cls, prompt_mappings: Dict[str, tuple]):
    """
    Class decorator to automatically create cached prompt properties.
    
    Args:
        prompt_mappings: Dict mapping property names to (registry, config_key) tuples
    """
    for prop_name, (registry, config_key) in prompt_mappings.items():
        def make_property(reg, key):
            def prop_func(self):
                return self._get_prompt_from_registry(reg, key)
            return cached_property(prop_func)
        
        setattr(cls, prop_name, make_property(registry, config_key))
    
    return cls